"""
推理态需要改写该函数
训练时用生成的中文词典
"""
import base64
import os
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Tuple

import tiktoken #openAI开发的字节对编码Python库

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {#编号和名称对调，和LANGUAGES的key value 互换，这个我之前也经常这么做
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

@dataclass
class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    encoding: tiktoken.Encoding#用tiktoken的Encoding
    language: Optional[str] = None #Optional为默认值，语言
    task: Optional[str] = None #任务
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)#特殊字符

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:#初始化special_token
            special_token = self.encoding.encode_single_token(special)#获取一个字或词的token
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]#开始位置token
        translate: int = self.special_tokens["<|translate|>"]#翻译标志token
        transcribe: int = self.special_tokens["<|transcribe|>"]#语音识别token

        langs = tuple(LANGUAGES.keys())#获取所有语言的key，即语言简称
        sot_sequence = [sot]#[开始位置token]
        if self.language is not None:#如果language不为空
            #下面这计算，就是语言对应的编码，好费解
            sot_sequence.append(sot + 1 + langs.index(self.language))#sot_sequence追加开始位置标志位+1+language的索引，就是整体的编码位置，不知道为啥要多此一举
        if self.task is not None:#如果任务不为空，这里是语音转文字transcribe
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)#追加任务token

        self.sot_sequence = tuple(sot_sequence)#压入self，这个变量，[开始位置编码，语言编码，任务编码]

    def encode(self, text, **kwargs):#编码，编码成token
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:#解码，解码成对应的文字或单词
        token_ids = [t for t in token_ids if t < self.timestamp_begin]#时间戳？？？
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:#带时间戳的decode
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)
    
    #这些值都缓存到cache里面
    @cached_property
    def eot(self) -> int:#处理应该是返回字对应的索引
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:#处理应该是返回字对应的索引
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:#返回语言编号
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        if token := self.special_tokens.get(f"<|{self.language}|>", None):
            return token

        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:#所有语音的token
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @cached_property
    def all_language_codes(self) -> Tuple[str]:#所有语言的单词
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:#没有时间戳的句子
        return tuple(list(self.sot_sequence) + [self.no_timestamps])#no_timestamps编码

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:#取消实际说不出的token，如音符，标点符号等
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')#符号
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")#音符
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens: List[int]):#按字分开，如中文，没有空格的
        if self.language in {"zh", "ja", "th", "lo", "my"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: List[int]):#按unicode分开
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: List[int]):#按空格分开如英文
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens


@lru_cache(maxsize=None)#中文选择多语言版本
def get_encoding(name: str = "gpt2"):#默认是纯英文版本，限定是str然后赋值，卧槽，秀技能呢？？
    #这个路径就是./assets/multilingual.tiktoken
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    #vocab_path = "./whisper/assets/multilingual.tiktoken" #测试用
    print("词典路径：",vocab_path)
    #multilingual.tiktoken是由空格分开，然后前面的字用base64进行了加密，因此正常看起来无法解读
    ranks = {#这个写法也挺反人类，是不是openai目的就是不想让别人读懂他们的代码，傻逼
        base64.b64decode(token): int(rank)
        #token: int(rank) #测试
        for token, rank in (line.split() for line in open(vocab_path) if line)#空格分开的，前面是字或者词的base64加密，后者是编号
    }
    #print(ranks)
    #rank是一个词典，形式为{字：编号}
    n_vocab = len(ranks)#词典总样本数
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],#将语言简编号，作为specials
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],#从0到1501按0.02新增作为specials于意何为？？是不是时间戳？？
    ]

    for token in specials:#目的是将special字符追加到后面，这里咋不装了，写的这么low
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(#tiktoken就是openAI定义的编码库，
        name=os.path.basename(vocab_path),#编码器名称，这里是multilingual
        explicit_n_vocab=n_vocab,#词汇量，普通分词+特殊分词的个数,用于校验
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",#正则表达文本，用于分割文本
        mergeable_ranks=ranks,#这个就是词典啊
        special_tokens=special_tokens,#特殊字符集合
    )

#@lru_cache 装饰在被执行的函数上，将其执行的结果缓存起来，当下次请求的时候，如果请求该函数的传参
#未变则直接返回缓存起来的结果而不再执行函数的一种缓存装饰器
"""这个函数就是设置tokenizer，太垃圾"""
@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,#是否为多语言
    *,
    language: Optional[str] = None,#语言选择
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]，任务选择
) -> Tokenizer:
    if language is not None:#指定语音名称，好麻烦
        language = language.lower()#归一为小写
        if language not in LANGUAGES:#虎丘语言编码
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:#如果是多种语言版本，如果选择中文，这个是true，其实就是用多语言版本
        encoding_name = "multilingual"#文件夹assets中的multilingual.tiktoken
        language = language or "en"#如果为空则选择"en"
        task = task or "transcribe"#如果没有指定任务，默认是语音转文字
    else:
        encoding_name = "gpt2"#文件夹assets中的gpt2.tiktoken
        language = None#语言为空？？
        task = None#任务为空？？
    print(language)
    print(task)
    print(encoding_name)

    encoding = get_encoding(name=encoding_name)

    return Tokenizer(encoding=encoding, language=language, task=task)
