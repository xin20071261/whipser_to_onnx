# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:53:29 2023

@author: admin
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


from torch import nn
import torchaudio
import torchaudio.transforms as at
from pathlib import Path
import numpy as np
import time

import whisper


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
        #max_label_len = 128

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
    batch_size = 32#每批数据
    num_worker = 32#原始的为2
    num_train_epochs = 10#循环次数
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE



def ddp_setup(rank, world_size):#ddp初始化
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if rank !=0:#保证第一个启动的是master节点
        time.sleep(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        tokenizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.tokenizer = tokenizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)#损失函数：交叉熵损失函数

    def _run_batch(self,input_ids, labels, dec_input_ids,batch,size):
        self.optimizer.zero_grad()
        #https://blog.csdn.net/qq_43332629/article/details/125322182
        audio_features = self.model.module.encoder(input_ids)#编码语音数据
        out = self.model.module.decoder(dec_input_ids, audio_features)#解码语音编码和对应文字
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#计算损失函数
        loss.backward()
        self.optimizer.step()
        if self.gpu_id == 0  and batch % 100 == 0:#等于0和100的倍数时打印
            loss, current = loss.item(), (batch + 1) * len(labels)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _run_epoch(self, epoch):
        #len(self.train_data.dataset)是总数据，len(self.train_data)是batch数据
        b_sz = len(next(iter(self.train_data))['labels'])#迭代
        size = len(self.train_data.dataset)#正常
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)#打乱次序
        for batch, data in enumerate(self.train_data):
            labels = data['labels'].long()#decoder输出预期值
            dec_input_ids = data['dec_input_ids'].long()#decoder输入
            input_ids, labels,dec_input_ids = \
            data['input_ids'].to(self.gpu_id), labels.to(self.gpu_id),dec_input_ids.to(self.gpu_id)
            #print("batch:",batch)
            self._run_batch(input_ids, labels,dec_input_ids,batch,size)
            
    def _run_valid(self):
        size = len(self.val_data.dataset)#总数据量
        num_batches = len(self.val_data)#总batch量
        print("size",size)
        print("num_batches",num_batches)
        loss, correct_all = 0, 0
        test_loss = 0.0
        with torch.no_grad():
            for batch, data in enumerate(self.val_data):
                labels = data['labels'].long()#decoder输出预期值
                dec_input_ids = data['dec_input_ids'].long()#decoder输入
                input_ids, labels,dec_input_ids = data['input_ids'].to(self.gpu_id), labels.to(self.gpu_id),dec_input_ids.to(self.gpu_id)
                
                audio_features = self.model.module.encoder(input_ids)#编码语音
                out = self.model.module.decoder(dec_input_ids, audio_features)#编码
        
                loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#计算loss
                test_loss += loss.item()#.item()获取其中的数值
                
                correct = (out.argmax(-1) == labels).type(torch.float).sum().item()/(labels.shape[-1]) #计算推理的单词准确率
                
                correct_all += correct
        
                out[out == -100] = self.tokenizer.eot#输出后添加eot标志
                labels[labels == -100] = self.tokenizer.eot#标签添加eot标志
            
            
            test_loss /= num_batches
            correct_all /= size
            print(f"Test loss: \n Accuracy: {(100*correct_all):>0.1f}%,loss: {(test_loss):>8f} \n")
                

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "./model/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0:
                self._run_valid()
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)


def load_train_objs():
    
    lang = 'zh'#语言选择
    model_name = 'tiny'#模型选择
    options = whisper.DecodingOptions(language=lang, without_timestamps=True)#配置decoder
    model = whisper.load_model(model_name)#加载模型
    tokenizer = whisper.tokenizer.get_tokenizer(True, language="zh", task=options.task)#openAI 搞了一个词库tiktoken
    """创建训练数据加载器"""
    train_set = JvsSpeechDataset(train_audio_transcript_pair_list, tokenizer, Config.sample_rate)
    """创建验证数据加载器”"""
    test_dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, tokenizer, Config.sample_rate)
    '''
    train_set = torch.utils.data.DataLoader(train_dataset, 
                          batch_size=Config.batch_size, 
                          drop_last=True, shuffle=True, num_workers=Config.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
    '''
    #定义优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                        if not any(nd in n for nd in no_decay)],
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                        if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, #优化器
                        lr=Config.learning_rate, 
                        eps=Config.adam_epsilon)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set,test_dataset, model, optimizer,tokenizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        collate_fn=WhisperDataCollatorWhithPadding()
    )
    
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    print(f"Running basic DDP example on rank {rank}.")
    ddp_setup(rank, world_size)
    dataset,test_dataset, model, optimizer,tokenizer  = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, batch_size)
    trainer = Trainer(model, train_data, test_data, optimizer,tokenizer , rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=20,type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=100,type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    #world_size = torch.cuda.device_count()#这个是一台服务器上多少张卡，这个设置为3，因为一张卡有问题
    world_size = 3
    #spawn启动进程，应该会初始化rank，
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)