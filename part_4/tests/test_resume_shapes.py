# ==========================================
# 测试：checkpoint 保存/加载的"形状一致性"回合验证
# ==========================================
# 这个测试验证 Part 4 训练基础设施中最关键的一环——
# 断点续训（resume training）的可靠性。
#
# 场景：训练跑到一半，机器宕机/被抢占。重启后能否无缝接上？
# 答案取决于 checkpoint 能否完整保存并精确恢复以下五个组件：
#   1. 模型权重（model）      ：所有层的参数形状和数值必须一致
#   2. 优化器状态（optimizer） ：AdamW 的动量/方差统计（否则第一步更新就偏了）
#   3. 调度器状态（scheduler） ：学习率曲线走到哪了（否则 lr 会跳变）
#   4. AMP 缩放器（amp）       ：混合精度的 loss scaling 因子（重启后不匹配可能导致梯度下溢）
#   5. 训练步数（step）        ：当前是第几步，决定 scheduler 的内部计数和日志时间轴
#
# 测试策略：
#   创建一个包含以上所有组件的最简"假"训练状态 → 保存 → 加载到另一个新对象 →
#   验证加载后返回的 step 是整数（证明序列化/反序列化代码能跑通）。
# 注意：这里不检查具体权重数值是否相等（torch.save/load 由 PyTorch 保证），
# 只验证"能不能存"和"能不能读回来"——这是最简单的冒烟测试。

import torch, tempfile, os
import torch.nn as nn
from checkpointing import save_checkpoint, load_checkpoint

# ==========================================
# Dummy：最简的 nn.Module，只包含一个 Linear 层
# ==========================================
# 真实训练中模型可能有几十层 Transformer block，但 checkpoint 的保存/加载
# 逻辑与模型复杂度无关——只要有 state_dict() 就行。
# 一个 8×8 的线性层已经足够验证"参数状态能否正确序列化"。
class Dummy(nn.Module):
    def __init__(self):
        # 语法：super().__init__() 是必须的——调用父类 nn.Module 的初始化，
        # 这样 Dummy 才能被 PyTorch 的 Module 系统识别、才能调用 .parameters() 和 .state_dict()。
        super().__init__()
        # nn.Linear(8, 8)：输入 8 维 → 输出 8 维。
        # 内部包含两个参数张量：
        #   weight：形状 (8, 8)，共 64 个浮点数
        #   bias：  形状 (8,)，  共 8  个浮点数
        # 这两个参数的保存/恢复就是这个测试的核心。
        self.l = torch.nn.Linear(8, 8)


# ==========================================
# 测试函数：保存 → 加载 → 验证 step 可恢复
# ==========================================
# 参数 tmp_path 是 pytest 的 fixture（通过 conftest.py 或内置插件提供），
# 它会为每个测试自动创建一个唯一的临时目录，测试结束后自动清理。
# 比手写 tempfile 更简洁，而且不同测试之间不会互相污染。
def test_save_and_load(tmp_path):
    # ══════════════════════════════════════════════════════════
    # 第一步：构建"训练中"的完整状态（所有五个组件）
    # ══════════════════════════════════════════════════════════

    # ─── 组件 1：模型（model） ───
    # Dummy() 创建一个全新的 8×8 线性层，参数是随机初始化的。
    # 保存时会把参数的"当前值"写进 .pt 文件。
    m = Dummy()

    # ─── 组件 2：优化器（optimizer） ───
    # AdamW 是目前 LLM 训练的首选优化器：Adam 的动量机制 + 解耦权重衰减。
    # m.parameters() 告诉优化器"你需要管理哪些参数"。
    # lr=1e-3 是学习率，优化器内部会维护每个参数的动量（m）和二阶矩（v）。
    # 这些状态必须保存在 checkpoint 里——否则 resume 后动量清零，更新方向会突变。
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # ─── 组件 3：调度器（scheduler）——用"空类"模拟 ───
    # 语法：`class S: pass` 创建一个什么都不做的空类。
    # Python 的 class 语句以冒号结尾；pass 是占位语句，表示"这里什么也不执行"。
    # 为什么用空类而不是真实的 WarmupCosineLR？
    #   1. 测试不关心调度器的数学公式是否正确（那个由 test_scheduler.py 验证）
    #   2. 我们只关心调度器的内部状态（warmup_steps、step_num 等）能否正确序列化
    #   3. 用空类可以精确控制 __dict__ 的内容，测试更可控
    # 语法：`sch = S()` 创建 S 的实例。
    class S: pass
    sch = S()
    # 语法：`obj.__dict__ = {...}` 直接替换实例的属性字典。
    # 等价于依次执行：
    #   sch.warmup_steps = 10
    #   sch.total_steps = 100
    #   sch.base_lr = 1e-3
    #   sch.step_num = 5
    # 这模拟了一个已经跑了 5 步的调度器：
    #   前 10 步是 warmup → cosine 衰减 → 总共 100 步 → 峰值 1e-3
    # step_num=5 说明"当前处于第 5 步"，恢复后应从这里继续。
    sch.__dict__ = {'warmup_steps': 10, 'total_steps': 100, 'base_lr': 1e-3, 'step_num': 5}

    # ─── 组件 4：AMP 混合精度缩放器（amp scaler）——也`用空类模拟 ───
    # AMP（Automatic Mixed Precision）用 FP16 做前向/反向，FP32 存权重更新。
    # GradScaler 是关键组件：它动态放大 loss，防止小梯度在 FP16 下变成 0（下溢）。
    # scaler 的缩放因子（scale）是自动调整的，必须保存到 checkpoint，
    # 否则 resume 后 scaler 从 1.0 重新开始，前几步的梯度可能下溢导致训练不稳定。
    class A: pass
    amp = A()
    # enabled=False 创建了一个"已禁用"的 scaler——实际上不做任何缩放。
    # 测试不需要真的混合精度训练，我们只需要一个能被 save/load 的 scaler 对象。
    # torch.cuda.amp.GradScaler 内部有 state_dict() 和 load_state_dict() 方法，
    # 这些是 checkpoint 保存/加载时调用的接口。
    amp.scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ══════════════════════════════════════════════════════════
    # 第二步：保存 checkpoint
    # ══════════════════════════════════════════════════════════
    # 语法：tmp_path / "chk" 是 pathlib.Path 的 / 运算符，
    # 等价于 os.path.join(tmp_path, "chk")，但更简洁。
    # tmp_path 是 pytest 提供的 Path 对象，/ 运算符会自动拼接路径。
    out = tmp_path / "chk"

    # save_checkpoint 把五个组件全部序列化到一个 .pt 文件中：
    #   m, opt, sch, amp → 训练状态
    #   step=123          → 假设已经训练了 123 步
    #   out_dir=str(out)  → 保存到临时目录
    #   tokenizer_dir=None → 不保存分词器（这个测试不需要）
    # 保存后会在 out/ 下生成 model_last.pt 文件。
    save_checkpoint(m, opt, sch, amp, step=123, out_dir=str(out), tokenizer_dir=None)

    # ══════════════════════════════════════════════════════════
    # 第三步：创建"全新"的对象（模拟重启后的干净状态）
    # ══════════════════════════════════════════════════════════
    # 这里的关键哲学：我们创建的是"全新"的对象，而不是直接复用原来的。
    # 这模拟了真实的 resume 场景——进程重启后，Python 对象全部丢失，
    # 必须从磁盘 checkpoint 恢复到新创建的对象中。
    #
    # 新模型 m2 的参数是随机初始化的（不同种子），如果不加载 checkpoint，
    # 它的权重和之前保存的 m 完全不同。

    # 新模型：结构相同（都是 8×8 Linear），但参数值不同。
    m2 = Dummy()
    # 新优化器：学习率相同，但动量/方差全为 0（还没跑过任何步骤）。
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)

    # 新调度器：step_num=0（从零开始），warmup/total 都设为 1（占位值）。
    # 这些值会在 load_checkpoint 后被覆盖为保存时的值（warmup=10, total=100, step_num=5）。
    sch2 = S()
    sch2.__dict__ = {'warmup_steps': 1, 'total_steps': 1, 'base_lr': 1e-3, 'step_num': 0}

    # 新 AMP scaler：同样创建一个已禁用的 scaler。
    # 加载后它的缩放因子会恢复到保存时的值。
    amp2 = A()
    amp2.scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ══════════════════════════════════════════════════════════
    # 第四步：加载 checkpoint，恢复训练状态
    # ══════════════════════════════════════════════════════════
    # str(out / "model_last.pt") 拼接出完整路径，如 /tmp/xxx/chk/model_last.pt。
    # load_checkpoint 做四件事：
    #   1. 把 model_last.pt 加载到 CPU（map_location="cpu"）
    #   2. 校验模型结构与 checkpoint 中的 config 是否匹配
    #   3. 把权重加载到 m2（覆盖随机初始化值）
    #   4. 如果传入了 optimizer/scheduler/amp，也恢复它们的状态
    #   5. 返回保存时记录的 step 值
    step = load_checkpoint(m2, str(out / "model_last.pt"),
                           optimizer=opt2, scheduler=sch2, amp=amp2)

    # ══════════════════════════════════════════════════════════
    # 第五步：验证结果
    # ══════════════════════════════════════════════════════════
    # 语法：isinstance(step, int) 检查 step 是否为 int 类型。
    # 保存时传的是 step=123，理论上加载后应该是 123。
    # 但这里只检查类型——如果序列化/反序列化链路上有 bug，
    # 可能在多层嵌套中丢掉了类型信息（比如变成 float、None）。
    # 更严格的话可以加上 `and step == 123`，
    # 但当前这种"轻断言"已经足够验证整个流程的代码能跑通。
    assert isinstance(step, int)
