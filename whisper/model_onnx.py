import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function#推理
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

import FunctionSet as fs


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):#正则化
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):#全链接
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(#全链接y=x*A^T+b
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):#一维卷积
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(#卷积运算
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):#这个应该是位置编码
    """Returns sinusoids for positional embedding"""
    """返回位置嵌入的正余弦曲线"""
    assert channels % 2 == 0#channel必须是偶数
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):#多头注意力
    layer_count = 0
    def __init__(self,n_ctx: int, n_state: int, n_head: int):#比原始的增加了字符可推理的最大长度
        super().__init__()
        self.n_head = n_head#头数
        self.query = Linear(n_state, n_state)#接下来就是多头注意力的key、value和query
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)#这个应该是输出
        
        self.layer_id = MultiHeadAttention.layer_count#后加1，起初为0
        MultiHeadAttention.layer_count += 1
        self.n_ctx = n_ctx#就是字符长度

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,#交叉注意力的输入，为空则为自多头注意力
        mask: Optional[Tensor] = None,#编码会有mask,448矩阵
        kv_cache: Optional[dict] = None,#这个是缓存上一次的结果，默认不缓存，这里其实并非dict 而是tensor
    ):
        q = self.query(x)#query一直和x有关，但是key和value要看情况
        
        k = self.key(x if xa is None else xa)#如果没有缓存，要不xa（交叉注意力）要不x（自注意力）
        v = self.value(x if xa is None else xa)#如果没有缓存，要不xa（交叉注意力）要不x（自注意力）

        #如下的函数的作用就是将decoder的mask attention的v和k值不断cat，（用k.shape[1] <= self.n_ctx限制cross attention）
        if kv_cache is not None and k.shape[1] <= self.n_ctx:#decoder的mask attention的k和v
            layer_count = kv_cache.shape[0] // 2 #是因为同时缓存了k和v，这个是总长度
            key_id = self.layer_id - layer_count#这个是从后往前存的，0、2，4，……，2n  -12；2的倍数是因为decoder包含cross attention和mask attention
            
            value_id = key_id + 1#vualue比key大1
            size = k.shape[1]
            kv_cache[key_id, :, -size:, :] = k#这里并没有拼接之意，需要onnx外部拼接，切记,如果要实现拼接，在onnx输入时在kv_cahche的低三维后面添加一维
            kv_cache[value_id, :, -size:, :] = v
            k = kv_cache[key_id]
            v = kv_cache[value_id]
            
 

        wv, qk = self.qkv_attention(q, k, v, mask)#
        return self.out(wv), qk

    #mask是可选项，是decode的mask注意力机制，就是前面多个推理后一个字或者词
    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape#获取batch、ctx长度，state长度，如果要知道值打印即可[1,1,768]n_ctx=1,应该是每次一个词
        #if n_ctx>1:
            #print("q.shape:",n_ctx)
        #print("\n")
        #fs.save_txt_p("./test.txt",q.shape)
        scale = (n_state // self.n_head) ** -0.25#除以头数然后**-0.25 n_state // self.n_head：512/8=64,这里完全是transformer的计算方式
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale#这一步是将768 按12个头拆分如[1,512,768]->[1,12,512,64]
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k#矩阵乘法
        #if mask is None:
        #    print("qk.shape:",qk.shape)
        #fs.save_txt_p("./test.txt",q.shape) #保存测试
        if mask is not None:#如果带mask
            #print("q.shape:",q.shape)
            #print("k.shape:",k.shape)
            #fs.save_txt_p("./test.txt",q.shape)
            #print("qk.shape:",qk.shape)
            qk = qk + mask[:n_ctx, :n_ctx]#加0，没有意义啊,是不是训练的时候才有意义
            #print("qk.shape1:",qk.shape)
            #print("mask[:n_ctx, :n_ctx]:",mask[:n_ctx, :n_ctx].cpu().shape)
            #print("mask[:n_ctx, :n_ctx]:",q @ k == qk)
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):#这个是对传统的MultiHeadAttention做处理，残差注意力单元
    def __init__(self, n_ctx: int, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_ctx,n_state, n_head)#多头注意力机制
        self.attn_ln = LayerNorm(n_state)#正则化

        self.cross_attn = (
            MultiHeadAttention(n_ctx, n_state, n_head) if cross_attention else None#是否cross_attention，如果是则加一层自多头注意力模型
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None#是否cross_attention

        n_mlp = n_state * 4#512*4 why?
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )#顺序模型，用于前后链接
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,#字符编码
        xa: Optional[Tensor] = None,#语音经过编码后的结果
        mask: Optional[Tensor] = None,#加mask，如果是decoder的话
        kv_cache: Optional[dict] = None,
    ):
        maskAttentionWeights, maskAttentionQK = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)#加的意义是残差机制
        x = x + maskAttentionWeights
        if self.cross_attn:#交叉注意力机制
            crossAttentionWeights, crossAttentionQK = self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
            x = x + crossAttentionWeights
        else:
            crossAttentionWeights = None
            crossAttentionQK = None
        x = x + self.mlp(self.mlp_ln(x))
        return x#,crossAttentionQK

class AudioEncoder(nn.Module):
    #n_mels梅尔音谱图维度如80、n_ctx mels的长度，n_state应该是隐藏层神经元个数，n_head是头数，n_layer层数
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)#卷积
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)#卷积
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))#注册buffer应该就是保留权值

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(#n_layer层残差多头注意力机制
            [ResidualAttentionBlock(n_ctx,n_state, n_head) for _ in range(n_layer)]#这个循环就是多头注意力单元的个数
        )
        self.ln_post = LayerNorm(n_state)#正则化

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))#gelu激活参数
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)#第三和第二维度对调

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"#判断语音维度是否正确
        x = (x + self.positional_embedding).to(x.dtype)#位置编码，

        for block in self.blocks:
            x = block(x)#多头自注意力机制

        x = self.ln_post(x)#正则化
        return x


class TextDecoder(nn.Module):
    def __init__(
        # n_vocab文本编码、n_ctx是权重，n_state应该是隐藏层神经元个数，n_head是头数，n_layer层数
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        
        self.n_head = n_head

        self.token_embedding = nn.Embedding(n_vocab, n_state)#Embedding
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))#位置编码,是随网络进行训练的

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(#
            [
                ResidualAttentionBlock(n_ctx,n_state, n_head, cross_attention=True)#cross_attention，交叉注意力，就是跨主力合并，这里合并的是之前的值
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)#正则化

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)#mask，就是掩码，只保证只推理下一个字或词,是一个矩阵从上到下，字符0逐渐变多,是一个448矩阵
        #print("mask:",mask.shape)
        self.register_buffer("mask", mask, persistent=False)#注册buffer
    #def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):   
    def forward(self, x: Tensor, xa: Tensor,offset: Optional[Tensor] = torch.tensor(0), kv_cache: Optional[dict] = None):#新增这变量目的是转成onnx时，可以作为输入
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens,字符编码 编解码的x是不同含义的,这个要循环自增，一个字一个字推理，n_ctx应该是推理句子的总长度
            但是训练的时候，是如何处理的？？
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on#语音编码，这个就是encoder的输出尺寸，[1,80,1500]
        """
        #这个值是0，4，5，6……最大推理的字数，自增的，0跳到4是因为第一个输入是1*4维度的。切记这个值推理过程中很重要，必须有
        #这是模型原来的设置切片位置，用pytorch时用这个,导出onnx模型时禁止
        #offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#应该是开端位置,这个也有缓存，而且很重要，和位置编码强相关，想想这个怎么在onnx中保留
        #这个是给导出onnx模型用的，转化模型的时候用这个
        #offset = offset_in.shape[-1]
        #offset = offset_in
        print("offset:",offset)
        #print("x.shape[-1] before:",x.shape)
        x = (
            self.token_embedding(x)#这个就是矩阵加操作，两个矩阵尺寸一样，按元素相加,x.shape[-1]这个是字的长度，一般为4或者1；positional_embedding为[448,768]
            + self.positional_embedding[offset : offset + x.shape[-1]]#对字进行编码(位置编码+字编码)，这里是一个变化的量总值小于n_ctx,x.shape[-1]为隐藏层个数
        )#embedding+位置编码
        #print("x.shape2:",offset)
        #print("x.shape[-1]:",x.shape)
        x = x.to(xa.dtype)#x和xa的数据格式要一致
        
        #crossAttentionQKs = torch.zeros((len(self.blocks), x.shape[0], self.n_head, x.shape[1], xa.shape[1]))#存cross attention的k和v
        
        i:int=0
        for block in self.blocks:#mask embredding
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
            #x,crossAttentionQK = block(x, xa, mask=self.mask, kv_cache=kv_cache)
            #crossAttentionQKs[i] = crossAttentionQK
            i += 1

        x = self.ln(x)#正则化
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits, kv_cache#, crossAttentionQKs


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, model: str):
        super().__init__()
        self.type = model
        
        self.dims = dims
        self.encoder = AudioEncoder(#语音编码
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(#文本解码
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )#所有头
        all_heads[self.dims.n_text_layer // 2 :] = True#部分值为true
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)#注册

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)#注册

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder.forward(tokens, audio_features, kv_cache=torch.from_numpy(kv_cache), offset=0)
        return output

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _= self.decoder(tokens, self.encoder(mel), kv_cache=torch.from_numpy(kv_cache), offset=0)#这里并非dict，而是tensor
        return output

    @property
    def device(self):#是cpu还是GPU
        return next(self.parameters()).device

    @property
    def is_multilingual(self):#是否为多语言，
        return self.dims.n_vocab == 51865
    
    def new_kv_cache(self, n_group: int, length: int):
        if self.type == "tiny.en" or self.type == "tiny":
            size = [8, n_group, length, 384]
        elif self.type == "base.en" or self.type == "base":
            size = [12, n_group, length, 512]
        elif self.type == "small.en" or self.type == "small":
            size = [24, n_group, length, 768]
        elif self.type == "medium.en" or self.type == "medium":
            size = [48, n_group, length, 1024]
        elif self.type == "large" or self.type == "large-v1" or self.type == "large-v2":
            size = [64, n_group, length, 1280]
        else:
            raise ValueError(f"Unsupported model type: {self.type}")
        return np.zeros(size, dtype=np.float32)

    detect_language = detect_language_function#语言判断
    transcribe = transcribe_function#语音转为文字，用的就是decoding.py的内容
    decode = decode_function#推理
    
