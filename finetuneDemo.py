# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:40:35 2023

@author: admin
"""
#https://colab.research.google.com/drive/1-8pogK4MKr4mG4zU3yaII5Lg2OlorTIP#scrollTo=K-rYPxEF2XEy


import IPython.display
from pathlib import Path

import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
from torch import nn
import pandas as pd
import whisper
import torchaudio
import torchaudio.transforms as at

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm.notebook import tqdm
import pyopenjtalk#这个由于numpy版本问题会报错，目测在这里只是用来规范化日语的故没有必要
import evaluate

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

DATASET_DIR = "D:/work/资料/chatGPT/NLP/code/whisper-main/data/jvs/jvs_ver1"#数据主目录，下面是语音子目录
SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:#导入数据
    waveform, sr = torchaudio.load(str(wave_path), normalize=True)#用torchaudio读取声音
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform
test = load_wave("D:/work/资料/chatGPT/NLP/code/whisper-main/data/jvs/speaker_1.wav")
test = load_wave(Path('D:/work/资料/chatGPT/NLP/code/whisper-main/data/jvs/jvs_ver1/jvs001/falset10/wav24kHz16bit/VOICEACTRESS100_001.wav'))

dataset_dir = Path(DATASET_DIR)#地址
#glob获取该文件夹下面所有transcripts_utf8.txt，属于递归获取
transcripts_path_list = list(dataset_dir.glob("*/*/transcripts_utf8.txt"))#语音对应的文字，是一个txt
print(transcripts_path_list)

def get_audio_file_list(transcripts_path_list, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000):
    audio_transcript_pair_list = []
    for transcripts_path in tqdm(transcripts_path_list):
        # 检查音频文件目录
        audio_dir = transcripts_path.parent / "wav24kHz16bit" #检查语音数据是否存在
        if not audio_dir.exists():
            print(f"{audio_dir}不存在")#语音文件夹是否存在
            continue

        # 从翻译文本中获取 AudioId 和文本
        with open(transcripts_path, "r", encoding='utf-8') as f:
            text_list = f.readlines()
        for text in text_list:
            audio_id, text = text.replace("\n", "").split(":")#按：分割，前面语音名称，后面为对应文本
            #print(audio_id, text)
            audio_path = audio_dir / f"{audio_id}.wav"#语音文件路径
            if audio_path.exists():#如果语音存在
                # 资料核对
                #print(audio_path)
                audio = load_wave(audio_path, sample_rate=sample_rate)[0]#读取语音数据
                if len(text) > text_max_length or len(audio) > audio_max_sample_length:#如果语音或文本过长则停止读取
                    print(len(text), len(audio))
                    continue
                audio_transcript_pair_list.append((audio_id, str(audio_path), text))#list组装，语音名称，地址和文本
    return audio_transcript_pair_list

train_num = int(len(transcripts_path_list) * TRAIN_RATE)#80%用于训练
train_transcripts_path_list, eval_transcripts_path_list = transcripts_path_list[:train_num], transcripts_path_list[train_num:]
train_audio_transcript_pair_list = get_audio_file_list(train_transcripts_path_list, TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
eval_audio_transcript_pair_list = get_audio_file_list(eval_transcripts_path_list, TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
print("TRAIN AUDIO DATASET NUM: ", len(train_audio_transcript_pair_list))
print("EVAL AUDIO DATASET NUM: ", len(eval_audio_transcript_pair_list))


def text_kana_convert(text):#文字标准化
    text = pyopenjtalk.g2p(text, kana=True)
    #text = pyopenjtalk.g2p(text)#文字到拼音
    return text
text=text_kana_convert("オこんにちは")

woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)#语言选择等
wmodel = whisper.load_model("base",download_root = "./model")#加载模型
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=woptions.task)

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

        #text = text_kana_convert(text)#文本归一化，这个中文用不到
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)# sot_sequence_including_notimestamps表示没有时间戳
        labels = text[1:] + [self.tokenizer.eot]#label和text差别是啥？？？，错位一位然后加self.tokenizer.eot（结束标志位）？
        #text和labels前后加的标志词是不是有问题，后期验证？？？
        return {
            "input_ids": mel,#梅尔音谱图
            "labels": labels, #标签，decoder输出
            "dec_input_ids": text # 文本，decoder输入
        }
    
#padding处理
class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []#初始化语音，标签和文字
        for f in features:#将之前list分别压入容器
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])#抓换为2维？
        
        label_lengths = [len(lab) for lab in labels]#label长度
        dec_input_ids_length = [len(e) for e in dec_input_ids]#文字长度
        max_label_len = max(label_lengths+dec_input_ids_length)#标签长度

        #文字和标签进行pading
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257是结束位的id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch

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

#写的太烂，主要是用于测试语音转文字的效果如何
with torch.no_grad():
    #audio_features = wmodel.encoder(b["input_ids"].cuda())
    audio_features = wmodel.encoder(b["input_ids"])#no cuda,声音数据编码，梅尔语谱图
    test_4 = audio_features.numpy()
    input_ids = b["input_ids"]
    labels = b["labels"].long()
    dec_input_ids = b["dec_input_ids"].long()

        
    #audio_features = wmodel.encoder(input_ids.cuda())
    audio_features = wmodel.encoder(input_ids)#梅尔语谱图声音进行编码
    print(dec_input_ids)
    print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
    print(audio_features.shape)
    print()
#out = wmodel.decoder(dec_input_ids.cuda(), audio_features)
out = wmodel.decoder(dec_input_ids, audio_features)#这个是文字和语音编码结果进行组合decoder

print(out.shape)
print(out.view(-1, out.size(-1)).shape)
print(b["labels"].view(-1).shape)

tokens = torch.argmax(out, dim=2)#输出结果处理测试
test_5 = tokens.numpy()
print(wtokenizer.eot)
for token in tokens:
    token[token == -100] = wtokenizer.eot
    #text = wtokenizer.decode(token, skip_special_tokens=True)#没有skip_special_tokens字段
    text = wtokenizer.decode(token)#词典是从哪里来的，定位一下
    print(text)
#测试结束

#微调训练配置
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
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=self.options.task)#openAI 搞了一个词库tiktoken

        # only decoder training
        for p in self.model.encoder.parameters():#禁止encoder部分，只训练decoder部分，事实上，可以选择不禁用，训练全部
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)#损失函数选择
        self.metrics_wer = evaluate.load("wer")#是啥？？？
        self.metrics_cer = evaluate.load("cer")#是啥？？？

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
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
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


log_output_dir = "/content/logs"#日志输出位置
check_output_dir = "/content/artifacts"#模型保存地址

train_name = "whisper"#训练名称
train_id = "00001"#训练编号

model_name = "base"#模型名称
lang = "ja"#语言类型，这个是日语

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


#训练后验证？？
checkpoint_path = "/content/artifacts/checkpoint/checkpoint-epoch=0007.ckpt"
state_dict = torch.load(checkpoint_path)#加载torch模型
print(state_dict.keys())#打印keys做测试
state_dict = state_dict['state_dict']#获取权重数据数据
whisper_model = WhisperModelModule(cfg)#加载whisper模型
whisper_model.load_state_dict(state_dict)#将权重数据赋值给whisper模型

woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)#设置decoder选项，这里选择日语
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
        results = whisper_model.model.decode(input_ids, woptions)#这个是推理，问题是如何实现的 后期查看
        for r in results:
            res.append(r.text)
        
        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(l, skip_special_tokens=True)
            refs.append(ref)
            
            
cer_metrics = evaluate.load("cer")#评估矩阵？？
cer_metrics.compute(references=refs, predictions=res)

for k, v in zip(refs, res):
    print("-"*10)
    print(k)
    print(v)

            