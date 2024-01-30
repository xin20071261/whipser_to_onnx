# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:28:35 2023

@author: admin
"""

import torch
from whisper.model_onnx import Whisper, ModelDimensions
import numpy as np


"""
由于encoder部分是用语音梅尔音谱图进行一次编码，而decoder是循环推理的，因此要单独生成
两个不同部分的onnx，而encoder结果是作为decoder的交叉注意力机制的key和value，因此要和
文字编码一起作为decoder的输入。
"""

def export_onnx(onnx_encoder_file_path,onnx_decoder_file_path,modelName:str="tiny",anchors_freq_path = './model'):
    
    #for test
    #onnx_encoder_file_path= './model/small_encoder.onnx'
    #onnx_decoder_file_path = './model/small_decoder.onnx'
   #modelName:str="small"
    #anchors_freq_path = './model'
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#建议导出cpu
    #device = 'cpu'
    batchSize = 1
    

    
    validModelNames = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v1", "large-v2"]
    if not modelName in validModelNames:
    	print("Error: model name must be one of {}".format(", ".join(validModelNames)))
    	exit(1)
        
    checkpoint = torch.load(anchors_freq_path+"/{}.pt".format(modelName), map_location=device)
    print(checkpoint["dims"])
    
    #训练后的模型
    #modelDims = ModelDimensions(**checkpoint["dims"])
    modelDims = checkpoint["dims"]
    whisper = Whisper(modelDims, modelName)
    whisper.load_state_dict(checkpoint["MODEL_STATE"])
    #原始模型  
    #modelDims = ModelDimensions(**checkpoint["dims"])
    #whisper = Whisper(modelDims, modelName)
    #whisper.load_state_dict(checkpoint["model_state_dict"])

    whisper.eval()
    

    whisper.to(device)
    
    
    
    
    #inputs = torch.randn(1, 80, 3000,requires_grad=True)#语音编码输入
    encoderRandomInputs = torch.randn(batchSize, modelDims.n_mels, modelDims.n_audio_ctx * 2)
    encoderRandomInputs = encoderRandomInputs.to(device)
    
    
    text_inputs = torch.randint(51865,(batchSize, 4))#文本编码输入,四个变量的情况只有 第一次推理 这一种
    text_inputs = text_inputs.to(device)
    kvCache = torch.from_numpy(whisper.new_kv_cache(batchSize, 4))#这里设置成了1，
    kvCache = kvCache.to(device)
    #text_inputs = torch.tensor([[50258, 50260, 50359, 50363]])
    #print("text_inputs:",text_inputs.shape)
    #print("text_inputs:",text_inputs)
    #text_inputs = text_inputs.long
    
    #解码器位置编码切片
    #positional_slice=torch.randint(0,448,(1,))#这个224是能推理字符的最大长度
    
    #positional_slice = 0#这样也可以的
    positional_slice=torch.tensor(0)#这个224是能推理字符的最大长度
    positional_slice = positional_slice.to(device)
    #print(positional_slice.numpy())
    #print(positional_slice.item())
    
    
    #encoder_results = torch.randn(1, 1500,768,requires_grad=True)#语音编码结果
    #encoder_results = encoder_results.to(device)
    #print("encoder_results:",encoder_results)
    
    #text_inputs = np.random.randint(0,10,(1,4))
    #inputs = inputs.cuda()
    #options = whisper.DecodingOptions(language="zh", without_timestamps=True)#语言选择等
    #dummy_input = dummy_input.to(device)
    with torch.no_grad():
        encoder_out = whisper.encoder(encoderRandomInputs)#encoder部分
        #print("encoder_out:",encoder_out.shape)
        decoder_out=whisper.decoder(text_inputs,encoder_out)#decoder部分

    #encoder
    torch.onnx.export(whisper.encoder,         # model being run 
        encoderRandomInputs,       # model input (or a tuple for multiple inputs) 
        onnx_encoder_file_path,       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=11,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['VoiceEncoderInput'],   # the model's input names 
        output_names = ['EncoderResults'],
        dynamic_axes ={"VoiceEncoderInput":{0:"batchSize"}}#设置变尺寸，输入TextInput的第二个维度设置成变量
        
        )
    
    #decoder
    #'''
    torch.onnx.export(whisper.decoder,         # model being run 
        (text_inputs,encoder_out,positional_slice,kvCache),       # model input (or a tuple for multiple inputs) 
        onnx_decoder_file_path,       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=11,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['TextInput','voiceDecode','positional_slice','kv_cache'],   # the model's input names 
        output_names = ['decoderResults', "output_kv_cache"],
        dynamic_axes ={"TextInput":{0:"batchSize",1:"word_size"},
                       "voiceDecode":{0:"batchSize"},
                       "kv_cache":{1:"batchSize",2:"kv_size"}}#设置变尺寸，输入TextInput的第二个维度设置成变量
        )
    


if __name__ == '__main__':
    export_onnx("./model/tiny_encoder.onnx","./model/tiny_decoder.onnx")
