# [kv_cache] rollingkv_step 代码解释笔记

## 原始代码
```python
from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class KVCache:
    k: torch.Tensor  # (B,H,T,D)
    v: torch.Tensor  # (B,H,T,D)

    @property
    def T(self):
        return self.k.size(2)

class RollingKV:
    """Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        # crop
        if self.k.size(2) > self.window + self.sink:
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)
        return self.k, self.v
```

## 语法用法
`from __future__ import annotations` 会延迟类型注解求值，能减少前向引用时的兼容问题。`@dataclass` 是 Python 标准库装饰器，会自动为 `KVCache` 生成初始化器等样板代码，适合这类“纯数据容器”。

`@property` 把 `T` 定义成属性访问形式，调用时写 `cache.T` 而不是 `cache.T()`，可读性更好。`self.k.size(2)` 读取第 3 个维度长度（从 0 开始计数），这里表示时间维长度。

`torch.cat([...], dim=2)` 是核心拼接操作：把历史缓存和新 token 在时间维上连接。切片语法 `:self.sink`、`-self.window:` 分别表示“前 sink 个”与“后 window 个”时间步。`self.k, self.v = ...` 是 Python 的并行赋值写法。

## 维度解读
本段核心张量统一采用 `(B, H, T, D)`：
- `B`：batch size
- `H`：注意力头数
- `T`：时间步/已缓存 token 数
- `D`：每个头的通道维（head_dim）

输入与输出：
- `k_new`, `v_new` 通常形状为 `(B,H,T_new,D)`，在自回归解码时常见 `T_new=1`
- `step` 返回 `(k_cache, v_cache)`，形状为 `(B,H,T_keep,D)`，其中 `T_keep <= sink + window`

关键 shape 变化：
1. 首次写入（`self.k is None`）：
   - `self.k = k_new`，`self.v = v_new`
   - shape 直接变为 `(B,H,T_new,D)`
2. 后续追加：
   - `torch.cat([self.k, k_new], dim=2)`
   - 若旧缓存为 `(B,H,T_old,D)`，新片段为 `(B,H,T_new,D)`，则拼接后 `(B,H,T_old+T_new,D)`
3. 触发裁剪条件：`self.k.size(2) > window + sink`
4. 保留两段：
   - `sink_part = self.k[:, :, :sink, :]` 形状 `(B,H,sink,D)`
   - `tail_k = self.k[:, :, -window:, :]` 形状 `(B,H,window,D)`
5. 重新拼接：
   - `torch.cat([sink_part, tail_k], dim=2)` -> `(B,H,sink+window,D)`
   - `v` 完全同样处理，保证 K/V 时间轴对齐

这里没有广播、reshape、transpose 或 matmul；关键是沿 `dim=2` 的时间轴拼接与切片，且 K 与 V 必须保持同构 shape。

## 代码逻辑
这段代码实现了一个“滚动 KV 缓存（Rolling KV Buffer）”。在 Transformer 推理中，每一步会产生新的 K/V。为了避免无限增长占用显存，`RollingKV` 采用“保留前缀 + 保留最近窗口”的策略：

- 前缀 `sink`：固定保留最早的几个 token（attention sink），帮助模型保持全局锚点。
- 窗口 `window`：保留最近 token，保证局部上下文质量。

每次 `step` 都先追加新 K/V，再检查是否超出上限 `sink + window`。若超出，就切出前缀段与尾部段重新拼接，丢弃中间过旧片段。这样可以把缓存长度上界控制在常数级（相对总生成长度），显著降低长序列推理显存压力。

## 关键知识点总结
`RollingKV` 的本质是“时间轴有损压缩缓存”：保留最前面的锚点和最新上下文，舍弃中间历史。实现上最关键的约束是 K 与 V 的时间维必须同步更新，否则注意力读取会错位。`torch.cat(dim=2)` + 切片重组是这类 KV cache 的标准实现模式，简单但非常高效。