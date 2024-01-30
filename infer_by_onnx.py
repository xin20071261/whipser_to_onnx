# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:26:55 2023

@author: admin
"""
# onnx runtime: https://onnxruntime.ai/docs/get-started/with-python

import torch

import torchaudio
import torchaudio.transforms as at

import whisper

import numpy as np

import onnx
import onnxruntime as ort
import numpy as np

from torch.distributions import Categorical #归一化函数

import torch.nn.functional as F#导入nn对应的函数

from whisper.tokenizer import Tokenizer, get_tokenizer# whisper定义的分词器，这个后期替换
import FunctionSet as fs


#加载wave,读取为啥是两份？？因为是双通道
def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:#导入数据,格式是tensor
    waveform, sr = torchaudio.load(str(wave_path), normalize=True)#用torchaudio读取声音
    if sample_rate != sr:#如果采样率不同，
        waveform = at.Resample(sr, sample_rate)(waveform)#修改采样率
    return waveform[0]#防止双通道wav，读取2维一样的声音数据


audio_path="./data/aishell/test4.wav"
audio_data = load_wave(audio_path,16000)
audio_data = whisper.pad_or_trim(audio_data.flatten())#语音数据进行对齐，这里会转化成480000，是设置的一个参数
mel = whisper.log_mel_spectrogram(audio_data)#梅尔语谱图，尺寸都是audio.py定义好的[80,3000]
input_ids =mel[None, :]




#decoder模型校验
with open("./model/small_encoder.onnx", "rb") as f:
    onnx_model = onnx.load(f)
onnx.checker.check_model(onnx_model)

#providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
#加载runtime
ort_sess = ort.InferenceSession('./model/small_encoder.onnx',providers=['CPUExecutionProvider'])
#推理输出语音encoder
outputs = ort_sess.run(None, {'VoiceEncoderInput': input_ids.numpy()})

#outputs = ort_sess.run(None, {'input': text.numpy(),
#                            'offsets':  torch.tensor([0]).numpy()})
#输入初始字符
TextInput = np.array([[50258, 50260, 50359, 50363]],dtype='int64')

#初始化总概率值和tokens
n_audio = 1 #每次只推理一条语句
sum_logprobs: torch.Tensor = torch.zeros(1, device="cpu")#xx概率值初始化为0,1为bach size，device要看用什么推理，cpu or gpu
tokens: torch.Tensor = torch.tensor(TextInput).repeat(n_audio, 1)#初始化tokens，就是四个数，repeat语音的条数次，这里就只有一条语音

#分词器
tokenizer = get_tokenizer(multilingual = True)#获取分词器，必须这么写，很多内容加载到缓存中了
print(tokenizer.eot)
#test
print(tokenizer.sot_sequence_including_notimestamps)
print(tokenizer.decode([50258, 50259, 50359, 50363]))

#校验模型
with open("./model/small_decoder.onnx", "rb") as f:
    onnx_model_decoder = onnx.load(f)

onnx.checker.check_model(onnx_model_decoder)
#加载模型
ort_sess_decoder = ort.InferenceSession('./model/small_decoder.onnx',providers=['CPUExecutionProvider'])
#推理

positional_slice = 0
kv_cache = np.zeros([24,1,4,768], dtype=np.float32)#kv_cache初始化
kv_cache_add = np.zeros([24,1,1,768], dtype=np.float32)#每次都要拼接一个维度
print(np.array([positional_slice],dtype='int64').shape)
for i in range(224): 
    if i>0:#如果不是第一次，则取最后一个token作为下一个编码输入，第一次为initial_tokens 
        TextInput = [tokens[:,-1].numpy()]
        #print("tokens[:,-1]:",[tokens[:,-1].numpy()])
    decoder_outputs,kv_cache = ort_sess_decoder.run(None, {'TextInput': TextInput,'voiceDecode':outputs[0],
                                                  'positional_slice':np.array(positional_slice,dtype='int64'),
                                                  'kv_cache':kv_cache})
    decoder_outputs = decoder_outputs[:, -1]#获取最后一维度，原始为[1,4,51865],变为[1,1,51865],因为每次推理都是推理都是获取最后一个编码
    next_tokens = torch.tensor(decoder_outputs).argmax(dim=-1)#如果temperature==0，见decoding.py 291行，取最大值的索引，其结果类似与分类
    #print(next_tokens)
    logprobs = F.log_softmax(torch.tensor(decoder_outputs).float(), dim=-1)#log_softmax归一化
    current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]#获取推理字符对应的概率
    sum_logprobs += current_logprobs * (tokens[:, -1] != tokenizer.eot)#如果是结尾则加0
    #print("1:",sum_logprobs)
    
    #next_tokens[tokens[:, -1] == tokenizer.eot] = tokenizer.eot#如果tokens最后一位是结束位，这next_tokens结束,如果tokens最后一位是结束位置，则next_tokens=eot，这个语句应该不可达
    #print("2:",next_tokens)
    tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)#这个是做拼接
    #print("3:",tokens)
    
    completed = (tokens[:, -1] == tokenizer.eot).all()#这个应该是算是否终止推理
    #print("4:",completed)
    
    if i==0:#第一次推理后解码器的文字输入位置编码变+4
        positional_slice = positional_slice + 4
    else:#其余位置+1
        positional_slice = positional_slice + 1
    if completed:
        break
    else:
        kv_cache = np.concatenate((kv_cache,kv_cache_add),axis=2)#如果没有结束则凭借kv缓存区域
#生成对于的文字
texts: list[str] = [tokenizer.decode(t).strip() for t in tokens]#strip去掉空格，这个是whisper原生tokenizer

tokenizer_whisper=fs.load_data("tokenizer_whisper","./data/")#读取处理后的

texts:str=''
for t in tokens[0].numpy().tolist():
   texts += tokenizer_whisper[t].strip()

print(texts)
print(tokens)
print(tokens.shape)


#获取字典
tokenizer_change = dict()
#tokenizer_word = dict()
for i in range(0,50364):
    #print(tokenizer.decode([i]).strip())
    tokenizer_change.update({i:str(tokenizer.decode([i]).strip())})
    #tokenizer_word.update({tokenizer.decode([i]).strip():i})
    
fs.save_data("tokenizer_whisper", tokenizer_change, "./data/")#保存所有wisper的词典，用于推理时快速使用
tokenizer_whisper=fs.load_data("tokenizer_whisper","./data/")

#print(tokenizer_word['是'])
print(tokenizer_change[11249,222])
print(tokenizer.decode([11249,222]))#中文有的字符是两个编号一个字，因此直接获取编号的方式是有问题的
print(tokenizer.encode("简叉"))
print(tokenizer.encode("unfortunately"))
print(tokenizer_change[11249]+tokenizer_change[222])

ttt = tokenizer_whisper[11249]
print(repr(ttt))

tokenizer_word = list(set(list(tokenizer_change.values())))


'''
#test
TextInput = np.array([[14245]],dtype='int64')

kv_cache = np.zeros([24,1,4,768], dtype=np.float32)

kv_cache_add = np.zeros([24,1,1,768], dtype=np.float32)

decoder_outputs,kv_cache = ort_sess_decoder.run(None, {'TextInput': TextInput,'voiceDecode':outputs[0],
                                              'positional_slice':np.array(0,dtype='int64'),'kv_cache':kv_cache})

kv_cache = np.concatenate((kv_cache,kv_cache_add),axis=2)
print(kv_cache.shape)
#print(kv_cache[:,0,1,26])

decoder_outputs = decoder_outputs[:, 0]#获取最后一维度，原始为[1,4,51865],变为[1,1,51865],因为每次推理都是推理都是获取最后一个编码
next_tokens = torch.tensor(decoder_outputs).argmax(dim=-1)#如果temperature==0，见decoding.py 291行，取最大值的索引，其结果类似与分类
print(next_tokens)

#测试temperature不为0的情况
temperature = 0.6#如果temperature不为零，有啥不同，后期解读
next_tokens_t = Categorical(logits=torch.tensor(decoder_outputs) / temperature).sample()#这里做了归一化，从归一化的结果中生成样本
print(next_tokens_t)

#求概率
logprobs = F.log_softmax(torch.tensor(decoder_outputs).float(), dim=-1)#log_softmax归一化
test = logprobs.numpy()
print(test[:,14245])
current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]#获取推理字符对应的概率
print(current_logprobs)

sum_logprobs += current_logprobs * (tokens[:, -1] != tokenizer.eot)#如果是结尾则加0
print(sum_logprobs)

next_tokens[tokens[:, -1] == tokenizer.eot] = tokenizer.eot#如果tokens最后一位是结束位，这next_tokens结束,如果tokens最后一位是结束位置，则next_tokens=eot，这个语句应该不可达
#next_tokens[True] = tokenizer.eot#如果tokens最后一位是结束位，这next_tokens结束 测试用
print(next_tokens)
tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)#这个是做拼接
print(tokens)

completed = (tokens[:, -1] == tokenizer.eot).all()#这个应该是算是否终止推理
print(completed)


for i in range(224):
    print(i)
result = outputs[0].argmax(axis=1)+1
print("This is a %s news" %result)

print(onnx_model)
'''