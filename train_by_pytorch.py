# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:14:09 2023

@author: admin
"""

#代码是从https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing修改而来
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
from pathlib import Path

import numpy as np



import evaluate #Hugging Face提供的一个评估库



#1、设置音频数据加载对应参数
#LABEL_DIR = "/ai/trian/dataset/aishell/train/content.txt"#数据主目录，下面是语音子目录
LABEL_DIR = "/ai/trian/dataset/aishell/train/content_mini.txt"#数据主目录，下面是语音子目录
SAMPLE_RATE = 16000#音频频率
TRAIN_RATE = 0.8#100%多少用于训练

AUDIO_MAX_LENGTH = 480000#语音最大产犊
TEXT_MAX_LENGTH = 120#最大字符长度
SEED = 3407#随机数种子
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")#设备,这个是linghtning的不同设置

#seed_everything(SEED, workers=True)#设置随机数
torch.manual_seed(SEED)#设置随机数


#2、获取音频数据
def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:#导入数据,格式是tensor
    waveform, sr = torchaudio.load(str(wave_path), normalize=True)#用torchaudio读取声音
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


#3、这个函数主要是将语音编号，语音路径和文本进行组装，其中音频校验部分太慢，建议关闭
#文字最大长度可以限制为224，即whisper输入的一半
def get_audio_file_list(label_dir, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000):
    
    #获取aishel文件夹下所有文件路径,获取音频信号的路径
    audio_paths = []
    path = "/ai/trian/dataset/aishell/train/wav"#这个在不同的平台记得改
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
        audio_path = '0'#保证目录不存在
        for ad in audio_paths:#通过查询包含关系获取audio路径
            if audio_id in ad:
                audio_path = ad
                break
        if Path(audio_path).exists():#如果语音存在
            # 音频信息和对应文字核对，这个速度太慢
            """
            print(audio_path)
            audio = load_wave(audio_path, sample_rate=sample_rate)[0]#读取语音数据
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:#如果语音或文本过长则停止读取
                print(len(text), len(audio))
                continue
            """
            audio_transcript_pair_list.append((audio_id, str(audio_path), text))#list组装，语音名称，地址和文本
    return audio_transcript_pair_list


#4、获取训练音频及标签数据（不是加载数据到内存）
audio_transcript_pair_list = get_audio_file_list(LABEL_DIR, TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
train_num = int(len(audio_transcript_pair_list) * TRAIN_RATE)#80%用于训练
train_audio_transcript_pair_list, eval_audio_transcript_pair_list = audio_transcript_pair_list[:train_num],\
 audio_transcript_pair_list[train_num:]

print("用于训练的样本量: ", len(train_audio_transcript_pair_list))
print("用于评估的样本量: ", len(eval_audio_transcript_pair_list))


#5、数据格式转换为模型可以使用的格式，用get_audio_file_list获取的信息audio_info_list作为输入,这个函数可以训练的时候一点一点读取数据
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
        #sot_sequence_including_notimestamps为(50258, 50259, 50359, 50363)，对应<|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        #其中50260为<|zh|>，推理处理这个变成设置的语言
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]#由于whisper输出时是整体向前推移一位并添加结束位，因此这里的标签也做这样的处理
        return {
            "input_ids": mel,#梅尔音谱图
            "labels": labels, #标签，decoder输出
            "dec_input_ids": text # 文本，decoder输入
        }

#6、padding处理：并按bachsize进行dict包装，这时候labels和dec_input的长度是两者之和中最大的长度。
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

        #文字和标签进行pading，目测是两个长度之和作为输入，这样可以理解，为啥推理时要448(文本最大长度)/2作为推理可达的最大长度了
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



#7、训练配置
class Config:
    learning_rate = 0.0005#学习率
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2#预热步数
    batch_size = 16#每批数据
    num_worker = 32#原始的为2
    num_train_epochs = 10#循环次数
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE

#8、训练模型
class WhisperModel():
    def __init__(self, cfg:Config, model_name="base", lang="zh", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)#配置decoder
        self.model = whisper.load_model(model_name)#加载模型
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="zh", task=self.options.task)#openAI 搞了一个词库tiktoken

        # only decoder training
       # for p in self.model.encoder.parameters():#禁止encoder部分，只训练decoder部分，事实上，可以选择不禁用，训练全部
       #     p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)#损失函数：交叉熵损失函数
        #evaluate 是huggingface在2022年5月底搞的一个用于评估机器学习模型和数据集的库
        #https://blog.csdn.net/baobao3456810/article/details/107381052
        #https://zhuanlan.zhihu.com/p/449264305
        #https://zhuanlan.zhihu.com/p/114414797
        #这个下载很慢，这个下载不成功，先翻墙，再下载
        #self.metrics_wer = evaluate.load("wer")#word error rate 词错率
        #self.metrics_cer = evaluate.load("cer")#字错率
        #修改为
        self.metrics_wer = evaluate.load(path = "./model/evaluate/metrics/wer",name = "wer")#word error rate 词错率
        self.metrics_cer = evaluate.load(path = "./model/evaluate/metrics/cer",name = "cer")#字错率   

        self.cfg = cfg #配置参数
        self.__train_dataset = train_dataset #训练数据集
        self.__eval_dataset = eval_dataset #评估数据集

        #定义优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, #优化器
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        
    

        """创建训练数据加载器"""
        train_dataset = JvsSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )


        """创建验证数据加载器”"""
        test_dataset = JvsSpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        self.test_dataloader=  torch.utils.data.DataLoader(test_dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )


    def training(self):# 训练
    
        size = len(self.train_dataloader.dataset)#正常
        self.model.train()#标志模型用于训练
        for batch, data in enumerate(self.train_dataloader):
            labels = data['labels'].long()#decoder输出预期值
            dec_input_ids = data['dec_input_ids'].long()#decoder输入
            input_ids, labels,dec_input_ids = data['input_ids'].to(DEVICE), labels.to(DEVICE),dec_input_ids.to(DEVICE)

            #在该模式下，所有计算得出的tensor的requires_grad都自动设置为False，不会反向自动求导，因此如果要训练encoder，这个不能这么写，直接写到decode里面
            #with torch.no_grad():
            #    audio_features = self.model.encoder(input_ids)#编码语音数据
            audio_features = self.model.encoder(input_ids)#编码语音数据
            out = self.model.decoder(dec_input_ids, audio_features)#解码语音编码和对应文字
            loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#计算损失函数
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()#优化器迭代
            self.optimizer.zero_grad()#优化器清零
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(labels)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def validation(self): #验证
        size = len(self.test_dataloader.dataset)#总数据量
        num_batches = len(self.test_dataloader)#总batch量
        print("size",size)
        print("num_batches",num_batches)
        self.model.eval()
        loss, correct_all = 0, 0
        test_loss = 0.0
        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):
                labels = data['labels'].long()#decoder输出预期值
                dec_input_ids = data['dec_input_ids'].long()#decoder输入
                input_ids, labels,dec_input_ids = data['input_ids'].to(DEVICE), labels.to(DEVICE),dec_input_ids.to(DEVICE)
                
                audio_features = self.model.encoder(input_ids)#编码语音
                out = self.model.decoder(dec_input_ids, audio_features)#编码
        
                loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#计算loss
                test_loss += loss.item()#.item()获取其中的数值
                
                correct = (out.argmax(-1) == labels).type(torch.float).sum().item()/(labels.shape[-1]) #计算推理的单词准确率
                
                correct_all += correct
                #correct = (out.argmax(-1) == labels).type(torch.float)
                #correct = out.argmax(-1).type(torch.float)
                
                #print("out:",out.shape)
                #print("labels:",labels.shape)
                #print("correct:",correct)
                #print()
        
                out[out == -100] = self.tokenizer.eot#输出后添加eot标志
                labels[labels == -100] = self.tokenizer.eot#标签添加eot标志
                
                o_list, l_list = [], []
                for o, l in zip(out, labels):
                    o = torch.argmax(o, dim=1)
                    #o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
                    #l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
                    o_list.append(self.tokenizer.decode(o))
                    l_list.append(self.tokenizer.decode(l))            
                cer = self.metrics_cer.compute(references=l_list, predictions=o_list)#计算字错率，词错率和误差
                wer = self.metrics_wer.compute(references=l_list, predictions=o_list)
            
            test_loss /= num_batches
            correct_all /= size
            print(f"Test loss: \n Accuracy: {(100*correct_all):>0.1f}%, \
                  loss: {(test_loss):>8f}, \
                  cer: {cer:>8f}, wer: {wer:>8f} \n")
        
            #self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)#输出日志
            #self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
            #self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)
    

    
#9、参数设置
log_output_dir = "./content/logs"#日志输出位置
check_output_dir = "./content/artifacts"#模型保存地址

train_name = "whisper"#训练名称
train_id = "00001"#训练编号

model_name = "tiny"#模型名称
lang = "zh"#语言类型，这个是日语

cfg = Config()#配置参数

Path(log_output_dir).mkdir(exist_ok=True)#日志目录校验
Path(check_output_dir).mkdir(exist_ok=True)#模型目录校验

model = WhisperModel(cfg, model_name, lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

for t in range(cfg.num_train_epochs):#按循环次数进行训练
    print(f"Epoch {t+1}\n-------------------------------")
    model.training()
    model.validation()
print("Done!")
#torch.save(model.state_dict(), "model.pth") #保存模型

