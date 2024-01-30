#from:https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:26:25 2023

@author: admin
"""
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as at
import os

import whisper
from tqdm.notebook import tqdm
from pathlib import Path

import numpy as np

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import evaluate #Hugging Face提供的一个评估库

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)


#获取音频数据
def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:#导入数据,格式是tensor
    waveform, sr = torchaudio.load(str(wave_path), normalize=True)#用torchaudio读取声音
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

#测试获取音频数据
#mTest = load_wave("./data/aishell/test/wav/SSB0005/SSB00050353.wav")
#mTest = mTest.numpy()

#获取语音对应的文本信息并组装
LABEL_DIR = "./data/aishell/train/content.txt"#数据主目录，下面是语音子目录,这是一个中文数据
SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)
#这个已经改成读取中文aishell
def get_audio_file_list(label_dir, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000):
    
    #获取文件夹下所有文件路径,获取音频信号的路径
    audio_paths = []
    path = "./data/aishell/train/wav"
    for root, dirs, files in os.walk(path):
        for file in files:
            audio_paths.append(os.path.join(root, file))
    
    audio_transcript_pair_list = []
    # 从翻译文本中获取 AudioId 和文本
    with open(label_dir, "r", encoding='utf-8') as f:
        text_list = f.readlines()
    for text in text_list:
        audio_id, text = text.split("\t")
        text = text.split(" ")
        text = "".join([text[i] for i in range(0,len(text),2)])#由于aishell数据标签里面有拼音，只提取中文
        #print(audio_id, text)
        for ad in audio_paths:#通过查询包含关系获取audio路径
            if audio_id in ad:
                audio_path = ad
                break
        if Path(audio_path).exists():#如果语音存在
            # 音频信息核对，这个速度太慢
            """
            print(audio_path)
            audio = load_wave(audio_path, sample_rate=sample_rate)[0]#读取语音数据
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:#如果语音或文本过长则停止读取
                print(len(text), len(audio))
                continue
            """
            audio_transcript_pair_list.append((audio_id, str(audio_path), text))#list组装，语音名称，地址和文本
    return audio_transcript_pair_list


#数据格式转换为模型可以使用的格式
class JvsSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list#第93-94获取的数据信息
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]#对获取数据信息数据进行拆分

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)#多次导入wav数据，这种操作效率堪忧
        audio = whisper.pad_or_trim(audio.flatten())#语音数据进行对齐
        mel = whisper.log_mel_spectrogram(audio)#梅尔语谱图
        #这里以无需时间戳编码为起始点，推理时类似于rnn，逐字推理的，因此效率应该不高
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)# sot_sequence_including_notimestamps表示没有时间戳
        labels = text[1:] + [self.tokenizer.eot]#label和text差别是啥？？？，错位一位然后加self.tokenizer.eot（结束标志位）？
        #text和labels前后加的标志词是不是有问题，后期验证？？？
        return {
            "input_ids": mel,#梅尔音谱图
            "labels": labels, #标签编码，decoder输出的预期值
            "dec_input_ids": text # 文本编码，decoder输入
        }

#padding处理
class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []#初始化语音，标签和文字
        for f in features:#将之前list分别压入容器
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
        
        #之前的维度后移动一位，然后按第一维度拼接 如[2,3]变成[1,2,3]，5个拼接变成[5,2,3]
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]#label长度
        dec_input_ids_length = [len(e) for e in dec_input_ids]#文字长度
        max_label_len = max(label_lengths+dec_input_ids_length)#标签长度，文本编码长度+标签长度&&&&

        #文字和标签进行pading
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        # 50257是结束位的id
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] 

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


#数据获取测试
audio_transcript_pair_list = get_audio_file_list(LABEL_DIR, TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
train_num = int(len(audio_transcript_pair_list) * TRAIN_RATE)#80%用于训练
train_audio_transcript_pair_list, eval_audio_transcript_pair_list = audio_transcript_pair_list[:train_num],\
 audio_transcript_pair_list[train_num:]

print("TRAIN AUDIO DATASET NUM: ", len(train_audio_transcript_pair_list))
print("EVAL AUDIO DATASET NUM: ", len(eval_audio_transcript_pair_list))


woptions = whisper.DecodingOptions(language="zh", without_timestamps=True)#语言选择等
wmodel = whisper.load_model("base",download_root = "./model")#加载模型
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="zh", task=woptions.task)#多语言为true，其实这里应该是其他语言吧

dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)#数组组装
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())#数据加载


#测试数据加载功能
for b in loader:
    print(b["labels"].shape)
    print(b["input_ids"].shape)
    print(b["dec_input_ids"].shape)
    
    test_1 = b["labels"].numpy()#tensor转numpy
    test_2 = b["input_ids"].numpy()#tensor转numpy
    test_3 = b["dec_input_ids"].numpy()#tensor转numpy
    

    for token, dec in zip(b["labels"], b["dec_input_ids"]):
        token[token == -100] = wtokenizer.eot
        #text = wtokenizer.decode(token, skip_special_tokens=False)#没有这个字段
        text = wtokenizer.decode(token)
        print(text)

        dec[dec == -100] = wtokenizer.eot
        #text = wtokenizer.decode(dec, skip_special_tokens=False)#没有这个字段
        text = wtokenizer.decode(dec)
        print(text)
        print("\n")
    break


#训练配置
class Config:
    learning_rate = 0.0005#学习率
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2#预热步数
    batch_size = 16#每批数据
    num_worker = 2
    num_train_epochs = 10#循环次数
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE

#训练模型
class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="base", lang="ja", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)#配置decoder
        self.model = whisper.load_model(model_name)#加载模型
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="zh", task=self.options.task)#openAI 搞了一个词库tiktoken

        # only decoder training
        for p in self.model.encoder.parameters():#禁止encoder部分，只训练decoder部分，事实上，可以选择不禁用，训练全部
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)#损失函数选择
        #evaluate 是huggingface在2022年5月底搞的一个用于评估机器学习模型和数据集的库
        #https://blog.csdn.net/baobao3456810/article/details/107381052
        #https://zhuanlan.zhihu.com/p/449264305
        #https://zhuanlan.zhihu.com/p/114414797
        self.metrics_wer = evaluate.load("wer")#word error rate 词错率
        self.metrics_cer = evaluate.load("cer")#字错率

        self.cfg = cfg #配置参数
        self.__train_dataset = train_dataset #训练数据集
        self.__eval_dataset = eval_dataset #评估数据集
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):# 训练
        input_ids = batch["input_ids"]#语言处理后的结果
        labels = batch["labels"].long()#decoder输出
        dec_input_ids = batch["dec_input_ids"].long()#decoder输入

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)#编码语音数据

        out = self.model.decoder(dec_input_ids, audio_features)#解码语音编码和对应文字
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#计算损失函数
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)#打印日志
        return loss
    
    def validation_step(self, batch, batch_id): #验证
        input_ids = batch["input_ids"]#语言预处理后的结果
        labels = batch["labels"].long()#decoder输出
        dec_input_ids = batch["dec_input_ids"].long()#decoder输入


        audio_features = self.model.encoder(input_ids)#编码语音
        out = self.model.decoder(dec_input_ids, audio_features)#编码

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#计算loss

        out[out == -100] = self.tokenizer.eot#输出后添加eot标志
        labels[labels == -100] = self.tokenizer.eot#标签添加eot标志

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)#计算字错率，词错率和误差
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)#输出日志
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }#返回一个词典

    def configure_optimizers(self):
        """创建一个优化器和调度器"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """初始设置（加载数据集）"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):#加载训练数据
        """创建训练数据加载器"""
        dataset = JvsSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):#加载验证数据
        """创建验证数据加载器”"""
        dataset = JvsSpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
#参数设置
log_output_dir = "/content/logs"#日志输出位置
check_output_dir = "/content/artifacts"#模型保存地址

train_name = "whisper"#训练名称
train_id = "00001"#训练编号

model_name = "base"#模型名称
lang = "zh"#语言类型，这个是日语

cfg = Config()#配置参数

Path(log_output_dir).mkdir(exist_ok=True)#日志目录校验
Path(check_output_dir).mkdir(exist_ok=True)#模型目录校验

tflogger = TensorBoardLogger(#可视化配置
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

checkpoint_callback = ModelCheckpoint(#pytorch lightning模型保存设置
    dirpath=f"{check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1 # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
#模型加载
model = WhisperModelModule(cfg, model_name, lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

trainer = Trainer(
    precision=16,
    accelerator=DEVICE,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list
)

trainer.fit(model)#训练


#训练后验证
checkpoint_path = "/content/artifacts/checkpoint/checkpoint-epoch=0007.ckpt"
state_dict = torch.load(checkpoint_path)#加载torch模型
print(state_dict.keys())#打印keys做测试
state_dict = state_dict['state_dict']#获取权重数据数据
whisper_model = WhisperModelModule(cfg)#加载whisper模型
whisper_model.load_state_dict(state_dict)#将权重数据赋值给whisper模型

woptions = whisper.DecodingOptions(language="zh", without_timestamps=True)#设置decoder选项，这里选择中文
dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)#数据集初始化
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())#加载数据

refs = []
res = []
for b in tqdm(loader):
    input_ids = b["input_ids"].half().cuda()
    labels = b["labels"].long().cuda()
    with torch.no_grad():
        #audio_features = whisper_model.model.encoder(input_ids)
        #out = whisper_model.model.decoder(enc_input_ids, audio_features)
        #这个是推理，过程和训练差不多，输入是两部分，唯一的区别是文本部分全为1，具体再decoding.py中实现
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            res.append(r.text)
        
        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(l, skip_special_tokens=True)
            refs.append(ref)
            
            
cer_metrics = evaluate.load("cer")#评估矩阵，CER（字错率），WER（词错率）
cer_metrics.compute(references=refs, predictions=res)#这个评估矩阵用到啥地方了？？

for k, v in zip(refs, res):#打印推理结果和原始结果
    print("-"*10)
    print(k)
    print(v)

#测试语音转文字的效果如何
with torch.no_grad():
    input_ids = b["input_ids"]# 梅尔语普图
    audio_features = wmodel.encoder(input_ids)#no cuda,声音数据编码，梅尔语谱图
    labels = b["labels"].long()
    dec_input_ids = b["dec_input_ids"].long()

    test_4 = audio_features.numpy()#看看decoder之后是个啥样子
        
    print(dec_input_ids)
    print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
    print(audio_features.shape)
    print()
    #out = wmodel.decoder(dec_input_ids.cuda(), audio_features)
    out = wmodel.decoder(dec_input_ids, audio_features)#这个是文字和语音编码结果进行组合decoder
    
    print("out:")
    print(out.shape)
    print(out.view(-1, out.size(-1)).shape)
    print(b["labels"].view(-1).shape)
    
    tokens = torch.argmax(out, dim=2)#输出结果处理测试
    test_5 = tokens.numpy()#转换为numpy
    print("eot的值为：",wtokenizer.eot)
    for i in range(len(tokens)):
        token[token == -100] = wtokenizer.eot#如果输出的值为-100 则替换为wtokenizer.eot
        text = wtokenizer.decode(tokens[i])#词典是从哪里来的，定位一下
        otext = wtokenizer.decode(labels[i])#词典来自openai定义，这个建议替换成简体中文重新计算
        print("推理后的值为:",text)
        print("原始值为：",otext)
        print()
#测试结束


"""
with open(LABEL_DIR, "r", encoding='utf-8') as f:
    text_list = f.readlines()
for text in text_list:
    audio_id, text = text.split("\t")
    print(audio_id)
    text = text.split(" ")
    text = "".join([text[i] for i in range(0,len(text),2)])
    print(text)

#路径判断测试
path = "./data/aishell/train/wav\SSB1956\SSB19560481.wav"
if Path(path).exists():
    print(1)
    audio1 = load_wave(path, sample_rate=16000)[0].numpy()
else:
    print(0)
    
#获取文件夹下所有文件路径
wav_roads = []
path = "./data/aishell/train/wav"
for root, dirs, files in os.walk(path):
    for file in files:
        wav_roads.append(os.path.join(root, file))


        
#判断语音是否满足要求
sample_rate = 16000
text_max_length = 120
audio_max_sample_length = 480000
for wav_road  in wav_roads:

    # 资料核对
    #print(audio_path)
    audio = load_wave(wav_road, sample_rate=sample_rate)[0]#读取语音数据
    if len(audio) > audio_max_sample_length:#如果语音或文本过长则停止读取
        print(len(text), len(audio))
        continue
    print(wav_road)
"""
