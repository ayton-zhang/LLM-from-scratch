# ==========================================
# 组件：BPETokenizer —— BPE 字节级分词器
# ==========================================
# 包装 HuggingFace `tokenizers` 库的 ByteLevelBPETokenizer，
# 提供训练/保存/加载/编码/解码的统一接口。
#
# ==========================================
# BPE (Byte Pair Encoding) 算法直觉
# ==========================================
# 目标：把原始文本切分成"子词"（subword），既不像字符级那么碎片化，
#      也不像词级那样被生僻词卡住。
#
# 算法流程（以训练为例）：
#   1. 初始化：词表包含所有 256 个单字节 token
#   2. 遍历整个语料，统计"哪两个相邻 token 最常一起出现"
#   3. 把出现最频繁的那对合并为一个新 token（如 "t"+"h"→"th"）
#   4. 重复 2-3 直到词表达到 vocab_size
#
# 举例：英文句子 "the cat sat on the mat"
#   字节级（vocab=256）：t h e _ c a t _ s a t ... （每个字母占 1 token，序列极长）
#   BPE 后（vocab=8000）：the _ cat _ sat _ on _ the _ mat （常见词合并为 1 token）
#
# 关键优势：
#   - 未知词可被拆成已知子词（如 "unthinkable" → "un"+"think"+"able"）
#   - 字节级（byte-level）BPE 进一步做到：任何 Unicode 字符都不会是"未知"的，
#     因为所有字符最终都能拆成字节，而字节全在词表里
#
# 对比 Part 3 的 ByteTokenizer：
#   ByteTokenizer：vocab=256，每个字节一个 token，序列很长（等效信息密度低）
#   BPETokenizer ：vocab=8000~32000，常见子词合并为 1 token，序列更短
#   效果：相同 block_size 下 BPE 能"看到"的原文长度是字节级的 2-4 倍
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

# ==========================================
# 可选依赖导入：HuggingFace tokenizers 库
# ==========================================
# try/except 模式：tokenizers 是可选的第三方库（需 pip install tokenizers），
# 如果没安装，导入时不崩溃，只在实例化 BPETokenizer 时才报 ImportError。
# 这样 Part 4 的其他模块（如 train.py）可以 import tokenizer_bpe 而不触发异常，
# 只有真正尝试创建 BPETokenizer() 时才会报错提示安装。
try:
    # ByteLevelBPETokenizer：字节级 BPE 的具体实现。
    # 训练时用 ByteLevelBPETokenizer.train() 调教分词器，
    # 加载时用 Tokenizer.from_file() 恢复（兼容性更好）。
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
except Exception:
    # 设为 None 作为"未安装"的标记，__init__ 里检查并报友好错误。
    ByteLevelBPETokenizer = None


class BPETokenizer:
    """Minimal BPE wrapper (HuggingFace tokenizers).
    Trains on a text file or a folder of .txt files. Saves merges/vocab to out_dir.
    """

    # ==========================================
    # 初始化：配置词表大小和特殊 token
    # ==========================================
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] | None = None):
        # 依赖检查：如果 tokenizers 没安装，在这里报友好错误。
        # 不在文件顶部检查，因为其他模块可能只想 import 这个类而不实例化。
        if ByteLevelBPETokenizer is None:
            raise ImportError("Please `pip install tokenizers` for BPETokenizer.")
        self.vocab_size = vocab_size

        # 特殊 token 的含义（BPE 训练时会自动为它们预留词表位置）：
        #   <s>   (BOS)  : 序列开始标记，GPT 通常不用（因果注意力已经限定了方向）
        #   </s>  (EOS)  : 序列结束标记，训练时标记"到此为止"，推理时用于提前停止
        #   <pad> (PAD)  : 填充标记，把不等长的序列补到统一长度。
        #                   注意：GPT 训练通常不 pad（每个 batch 内序列等长），
        #                   但分词器保留这个 token 以备后续需要
        #   <unk> (UNK)  : 未知字符标记。字节级 BPE 通常不会产生 UNK（所有字节都在词表），
        #                   但预留它以防万一
        #   <mask> (MASK): 掩码标记，用于 BERT 式预训练（MLM）。GPT 不用，同样预留
        #
        # 语法：`special_tokens or [...]` 利用 Python 的短路求值：
        #   如果 special_tokens 是 None 或空列表（falsy），用默认值替换。
        #   注意：空字符串 "" 不能这样写（"" 也是 falsy，会错误回退）。
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]

        # _tok：底层的 HuggingFace Tokenizer 对象，在 train() 或 load() 后填充。
        # 初始为 None，调用 encode/decode 前必须 train 或 load。
        self._tok = None

    # ==========================================
    # train：在语料文件上训练 BPE 分词器
    # ==========================================
    def train(self, data_path: Union[str, Path]):
        """在文本文件或目录上训练 BPE 分词器。

        data_path 可以是单个 .txt 文件，也可以是一个目录（递归收集内部所有 .txt）。
        """
        # 收集所有要训练的 .txt 文件路径。
        files: List[str] = []
        p = Path(data_path)
        if p.is_dir():
            # 语法：p.glob("**/*.txt") 递归匹配目录下所有 .txt 文件。
            # ** 表示匹配任意层级的子目录（类似于 shell 的 `find . -name "*.txt"`）。
            # str(fp) 把 Path 对象转成字符串（HuggingFace tokenizers 需要 str 类型）。
            files = [str(fp) for fp in p.glob("**/*.txt")]
        else:
            # 单个文件：直接放入列表。
            files = [str(p)]

        # 创建底层分词器并训练。
        tok = ByteLevelBPETokenizer()
        # ByteLevelBPETokenizer.train() 的参数：
        #   files=：要训练的文本文件列表
        #   vocab_size=：目标词表大小（含特殊 token）
        #   min_frequency=2：子词合并的最小出现次数。设为 2 避免训练数据中只出现 1 次的
        #                    罕见字节组合被合并（它们不值得占词表位置）
        #   special_tokens=：特殊 token 列表，BPE 保证这些 token 一定在词表中
        tok.train(files=files, vocab_size=self.vocab_size, min_frequency=2, special_tokens=self.special_tokens)
        self._tok = tok

    # ==========================================
    # save：保存分词器到磁盘
    # ==========================================
    def save(self, out_dir: Union[str, Path]):
        """保存分词器的合并规则、词表和元数据到指定目录。

        产出文件：
          - vocab.json       : token → id 的映射表
          - merges.txt       : BPE 合并规则（按训练时的合并顺序排列）
          - tokenizer.json   : HuggingFace 格式的完整分词器（load 时优先用这个）
          - bpe_meta.json    : 额外元数据（vocab_size、special_tokens 等）
        """
        # 创建输出目录（递归，已存在不报错）。
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        # 防御：没 train 或 load 不能 save（否则 _tok 是 None，调用会崩）。
        assert self._tok is not None, "Train or load before save()."

        # ByteLevelBPETokenizer.save_model(str(out))：
        #   保存 vocab.json + merges.txt 到 out 目录。
        #   注意参数是 str（目录路径），不是 Path 对象。
        self._tok.save_model(str(out))
        # 额外保存 HuggingFace 格式的完整分词器（.json），加载时优先使用。
        # 语法：out / "tokenizer.json" 即 pathlib 的路径拼接（等价于 os.path.join）。
        self._tok.save(str(out / "tokenizer.json"))

        # 保存元数据：vocab_size 和 special_tokens 在加载时需要恢复。
        # json.dumps(meta) 把 Python dict 序列化为 JSON 字符串。
        # .write_text() 是 Path 的方法，一次性写入全部文本（内部自动 open+write+close）。
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        (out/"bpe_meta.json").write_text(json.dumps(meta))

    # ==========================================
    # load：从磁盘加载分词器
    # ==========================================
    def load(self, dir_path: Union[str, Path]):
        """从之前 save() 的目录恢复分词器。"""
        dirp = Path(dir_path)

        # 优先从 tokenizer.json（HuggingFace 完整格式）加载。
        # 这比分别加载 vocab.json + merges.txt 更可靠，因为 tokenizer.json
        # 包含了分词器的完整配置（特殊 token 的 ID、是否添加 prefix space 等）。
        # Prefer explicit filenames; fall back to glob if needed.
        vocab = dirp / "vocab.json"
        merges = dirp / "merges.txt"
        tokenizer = dirp / "tokenizer.json"

        # 防御检查：确认目录里有分词器文件（vocab.json/merges.txt 至少有一个）。
        # 如果标准文件名不存在（比如用户自定义了文件名），用 glob 兜底找。
        # 实际加载走的是下面的 Tokenizer.from_file(str(tokenizer))，不依赖
        # 这里找到的 vocab/merges 变量——它们只用于确认"目录不是空的分词器目录"。
        if not vocab.exists() or not merges.exists():
            # Fallback for custom basenames
            # dirp.glob("*.json") 返回目录下所有 .json 文件的生成器。
            # list() 把生成器转为列表（取具体文件路径）。
            vs = list(dirp.glob("*.json"))
            ms = list(dirp.glob("*.txt"))
            if not vs or not ms:
                raise FileNotFoundError(f"Could not find vocab.json/merges.txt in {dirp}")
            vocab = vs[0]
            merges = ms[0]

        # 注释掉的行是旧方式（分别指定 vocab 和 merges 路径加载）：
        # tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        # 现在直接用 Tokenizer.from_file() 加载完整的 tokenizer.json：
        tok = Tokenizer.from_file(str(tokenizer))
        self._tok = tok

        # 恢复元数据：如果 save 时写了 bpe_meta.json，就从中取回 vocab_size 和
        # special_tokens；如果文件不存在（老版本保存的），保持当前默认值。
        meta_file = dirp / "bpe_meta.json"
        if meta_file.exists():
            # 语法：meta_file.read_text() 读取文件全部文本内容。
            # json.loads(str) 把 JSON 字符串反序列化为 Python dict。
            meta = json.loads(meta_file.read_text())
            # .get(key, default)：如果 key 不在 dict 中，用 default 值代替。
            # 这样即使 bpe_meta.json 内容不完整也不会崩。
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)


    # ==========================================
    # encode / decode：分词器的主要功能
    # ==========================================
    def encode(self, text: str):
        """文本 → token ID 列表。

        例："Hello world" → [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
        （实际 ID 取决于 BPE 训练结果，常见子词可能只有 1-2 个 ID）
        """
        # self._tok.encode(text) 返回 Encoding 对象，.ids 属性是 token ID 的 Python list。
        ids = self._tok.encode(text).ids
        return ids

    def decode(self, ids):
        """token ID 列表 → 文本。

        例：[72, 101, 108, 108, 111] → "Hello"
        decode 是 encode 的逆操作。
        """
        return self._tok.decode(ids)
