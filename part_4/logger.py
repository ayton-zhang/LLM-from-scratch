from __future__ import annotations
import time
from pathlib import Path


# ==========================================
# NoopLogger：什么都不做的“空日志器”
# ==========================================
# 训练代码希望无论用户选择 tensorboard / wandb / none，都能调用同一套接口：
#   logger.log(...)
#   logger.close()
# NoopLogger 就是 "none" 模式下的占位实现。它不写磁盘、不连网络，只把调用吃掉。
# 这样主训练循环不用到处写 if logger is not None，代码会干净很多。
class NoopLogger:
    def log(self, **kwargs):
        # pass 是 Python 的空语句，表示“这里故意什么都不做”。
        pass

    def close(self):
        pass


# ==========================================
# TBLogger：TensorBoard 日志器
# ==========================================
# TensorBoard 适合本地训练可视化：loss 曲线、学习率曲线、参数分布、生成样例等
# 都会写成 event 文件，之后可用 tensorboard --logdir runs/part4 查看。
class TBLogger(NoopLogger):
    """
    Backward compatible:
      - logger.log(step=..., loss=..., lr=...)
    Extras you can optionally use:
      - logger.hist("params/wte.weight", tensor, step)
      - logger.text("samples/generation", text, step)
      - logger.image("attn/heatmap", HWC_or_CHW_tensor_or_np, step)
      - logger.graph(model, example_batch)
      - logger.hparams(dict_of_config, dict_of_metrics_once)
      - logger.flush()
    Auto-behavior:
      - If a value in .log(...) is a tensor/ndarray with >1 element, it logs a histogram.
      - If key starts with "text/", logs as text.
    """
    # logger.py
    def __init__(self, out_dir: str, flush_secs: int = 10, run_name: str | None = None):
        #   out_dir    : 日志根目录，例如 runs/part4。
        #                每次运行会在下面创建一个子目录，避免不同实验互相覆盖。
        #   flush_secs : SummaryWriter 多久把内存里的 event 数据刷到磁盘一次。
        #                间隔太短 I/O 更频繁；间隔太长，程序崩溃时可能丢少量最新日志。
        #   run_name   : 本次实验的名字。None 时自动用时间戳，方便区分多次运行。
        # self.w 保存真正的 TensorBoard SummaryWriter。
        # 先设为 None，后面 import/初始化成功后再赋值；如果失败，就保持 None 表示日志不可用。
        self.w = None

        # hparams_logged 是一个“只写一次”的开关。
        # TensorBoard 的超参数面板不适合每步重复写，写过一次后就把它置为 True。
        self.hparams_logged = False

        # 语法：`run_name or time.strftime(...)` 利用 or 的短路特性。
        # 如果用户传了非空 run_name，就用用户指定的；否则生成形如 20260623-201530 的时间戳。
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")

        # Path(out_dir) / run_name 用 pathlib 拼路径，比手写字符串拼接更安全。
        # 最终目录类似 runs/part4/20260623-201530。
        run_dir = Path(out_dir) / run_name

        # parents=True：父目录不存在时一起创建。
        # 例如 run_dir 是 runs/part4/20260623-201530，而 runs/part4 还不存在，
        # 它会把 runs 和 part4 这些父目录一并建好。
        # exist_ok=True：目录已存在也不报错。
        # 例如你手动指定 run_name="debug"，第二次写 runs/part4/debug 时会继续复用这个目录，
        # 方便重复调试同一个 run_name。
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 延迟导入 SummaryWriter：
            # 只有真的选择 TensorBoard 时才需要 import，避免没有安装 tensorboard 时影响其他日志模式。
            from torch.utils.tensorboard import SummaryWriter
            self.w = SummaryWriter(log_dir=str(run_dir), flush_secs=flush_secs)
        except Exception as e:
            # TensorBoard 不可用时降级为禁用日志，而不是让训练直接崩掉。
            print(f"[TBLogger] TensorBoard not available: {e}. Logging disabled.")

        # 自动记录直方图时的上限。
        # 小张量可以完整画分布；太大的张量直接画直方图会很慢、event 文件也会膨胀，
        # 所以后面会改成只记录 mean/std 这类摘要统计。
        self._auto_hist_max_elems = 2048

        # 保存实际运行目录，方便外部打印或 debug 时知道日志写到了哪里。
        self.run_dir = str(run_dir)  # handy for prints/debug



    # ---------- backwards-compatible ----------
    def log(self, step: Optional[int] = None, **kv: Any):
        # 如果 SummaryWriter 初始化失败，self.w 为 None。
        # 这里直接返回，让调用方无需关心 TensorBoard 是否可用。
        if not self.w: return

        # 语法：**kv 收集任意关键字参数为字典。
        # 调用 logger.log(step=10, loss=1.2, lr=3e-4) 时，
        # step 单独进入 step 参数，loss/lr 则进入 kv={"loss": 1.2, "lr": 0.0003}。
        for k, v in kv.items():
            # text channel (opt-in via key prefix)
            # 约定：key 以 "text/" 开头时，把 value 当作文本写入 TensorBoard。
            # 例如 text/samples 可以展示模型生成结果，而不是画成数值曲线。
            if isinstance(k, str) and k.startswith("text/"):
                try:
                    # k[5:] 去掉 "text/" 前缀。
                    # 语法：字符串切片 s[5:] 表示从下标 5 开始取到末尾。
                    self.w.add_text(k[5:], str(v), global_step=step)
                except Exception:
                    # 日志是辅助功能，写失败不应该中断训练。
                    pass
                continue

            # scalar vs histogram auto-route
            try:
                import torch, numpy as np  # lazy
                is_torch = isinstance(v, torch.Tensor)
                is_np = isinstance(v, np.ndarray)
                if is_torch or is_np:
                    # scalar?
                    # numel 表示元素个数：
                    #   torch.Tensor 用 .numel()
                    #   numpy.ndarray 用 .size
                    # numel==1 时，它是标量；numel>1 时，它是一组数，更适合看分布。
                    numel = int(v.numel() if is_torch else v.size)
                    if numel == 1:
                        # 单元素张量/数组转成 Python float，记录为标量曲线。
                        # .item() 会把形如 tensor(1.23) 的 0 维张量取出为普通数值。
                        val = (v.item() if is_torch else float(v))
                        self.w.add_scalar(k, float(val), global_step=step)
                    else:
                        # small-ish tensors => histogram
                        # 多元素且不太大时，记录直方图，适合观察参数/梯度分布是否爆炸或塌缩。
                        if numel <= self._auto_hist_max_elems:
                            # TensorBoard 写直方图前需要 CPU 数据。
                            # detach() 切断计算图，避免日志系统意外持有梯度历史；
                            # cpu() 把 GPU 张量搬回 CPU，SummaryWriter 才好序列化。
                            self.w.add_histogram(k, v.detach().cpu() if is_torch else v, global_step=step)
                        else:
                            # fall back to scalar summary stats
                            # 大张量不直接写完整分布，改写 mean/std 两个标量。
                            # flatten() 把任意形状压成一维，便于统一计算统计量。
                            arr = v.detach().cpu().flatten().numpy() if is_torch else v.flatten()
                            self.w.add_scalar(k + "/mean", float(arr.mean()), global_step=step)
                            self.w.add_scalar(k + "/std", float(arr.std()), global_step=step)
                    continue
            except Exception:
                # 如果 torch/numpy 不可用，或者某个奇怪对象处理失败，就落到下面按普通数字尝试。
                pass

            # number-like
            try:
                # 普通 int/float 会走这里，写成 TensorBoard 标量曲线。
                self.w.add_scalar(k, float(v), global_step=step)
            except Exception:
                # swallow non-numeric junk silently (same behavior as before)
                # 非数字对象既不是 text/，也不能转 float，就静默忽略。
                pass

    # ---------- nice-to-have helpers ----------
    def hist(self, tag: str, values: Any, step: Optional[int] = None, bins: str = "tensorflow"):
        # 显式记录直方图的辅助方法。
        # 比自动路由更直接，适合 checkpointing.py 里专门记录参数/梯度分布。
        if not self.w: return
        try:
            import torch
            if isinstance(values, torch.Tensor):
                # detach + cpu 的理由同上：日志只需要数值快照，不需要梯度图，也不能直接写 GPU 张量。
                values = values.detach().cpu()
            self.w.add_histogram(tag, values, global_step=step, bins=bins)
        except Exception:
            pass

    def text(self, tag: str, text: str, step: Optional[int] = None):
        # 显式记录文本，例如保存某一步的生成样例、checkpoint 信息。
        if not self.w: return
        try:
            self.w.add_text(tag, text, global_step=step)
        except Exception:
            pass

    def image(self, tag: str, img, step: Optional[int] = None):
        """
        img: torch.Tensor [C,H,W] or [H,W,C] or numpy array
        """
        # 显式记录图片，例如注意力热力图。
        # TensorBoard 需要知道图片通道维在哪里：CHW = 通道在前，HWC = 通道在后。
        if not self.w: return
        try:
            # getattr(img, "ndim", 0)：如果 img 有 ndim 属性就取它，否则返回 0。
            # 当 img 是 3 维且第 0 维是 1 或 3 时，通常代表 [C,H,W]；
            # 否则按 [H,W,C] 处理。
            self.w.add_image(tag, img, global_step=step, dataformats="CHW" if getattr(img, "ndim", 0) == 3 and img.shape[0] in (1,3) else "HWC")
        except Exception:
            pass

    def graph(self, model, example_input):
        # 记录模型计算图，方便在 TensorBoard 里查看模块连接关系。
        # 注意：动态图模型里如果有复杂控制流，add_graph 可能 trace 失败，所以这里吞掉异常。
        if not self.w: return
        try:
            # example_input: a Tensor batch or a tuple
            # add_graph 期望输入是 tuple；单个 Tensor 包一层 tuple，多个输入则原样使用。
            if not isinstance(example_input, tuple):
                example_input = (example_input,)
            self.w.add_graph(model, example_input)
        except Exception:
            pass  # graph tracing can fail depending on model control flow; don't crash

    def hparams(self, hparams: Dict[str, Any], metrics_once: Optional[Dict[str, float]] = None):
        # 记录超参数表，例如 lr、batch_size、n_layer。
        # self.hparams_logged 防止重复写入，避免 TensorBoard 左侧实验列表被刷屏。
        if not self.w or self.hparams_logged:
            return
        try:
            # Single, stable sub-run so it doesn’t spam the left pane
            # metrics_once 是配套的一次性指标，比如 total_steps。
            # run_name="_hparams" 固定子运行名，让超参数记录集中在一个位置。
            self.w.add_hparams(hparams, metrics_once or {}, run_name="_hparams")
            self.hparams_logged = True
        except Exception:
            pass

    def flush(self):
        # 手动把缓存中的 event 写到磁盘。
        # 长训练中通常自动 flush 就够了；保存 checkpoint 或退出前手动 flush 更稳。
        if self.w:
            try: self.w.flush()
            except Exception: pass

    def close(self):
        # 关闭 SummaryWriter，确保剩余日志落盘并释放文件句柄。
        if self.w:
            try: self.w.close()
            except Exception: pass


# ==========================================
# WBLogger：Weights & Biases 日志器
# ==========================================
# WandB 适合远程实验追踪和团队协作。这里保持一个很薄的封装，
# 让训练循环仍然只需要调用 logger.log(...)。
class WBLogger(NoopLogger):
    def __init__(self, project: str, run_name: str | None = None):
        #   project  : WandB 项目名，多个 run 会归到同一个项目下。
        #   run_name : 本次实验名字；None 时由 WandB 自动生成。
        try:
            # 延迟导入 wandb：只有用户选择 --log wandb 时才依赖它。
            import wandb
            wandb.init(project=project, name=run_name)
            self.wb = wandb
        except Exception:
            # WandB 未安装、未登录或网络不可用时，降级为不记录。
            self.wb = None

    def log(self, **kv):
        # WandB 的 log 接收一个字典。
        # 这里 **kv 会把 logger.log(step=..., loss=...) 收集成 {"step": ..., "loss": ...}。
        if self.wb: self.wb.log(kv)


# ==========================================
# init_logger：根据配置选择日志后端
# ==========================================
# 这是训练脚本唯一需要调用的入口。
# 它把 "tensorboard" / "wandb" / "none" 映射到具有相同 .log() 接口的对象，
# 这就是“鸭子类型”：不关心对象具体属于哪个类，只要它会 log/close 就能用。
def init_logger(which: str, out_dir: str = "runs/part4"):
    if which == 'tensorboard':
        tb = TBLogger(out_dir)
        # 如果 TensorBoard 初始化失败，返回 NoopLogger。
        # 训练继续跑，只是没有可视化日志。
        return tb if tb.w is not None else NoopLogger()
    if which == 'wandb':
        return WBLogger(project='llm-part4')
    # which == "none" 或其他兜底情况：使用空日志器。
    return NoopLogger()
