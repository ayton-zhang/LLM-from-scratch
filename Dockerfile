# 1. FROM：指定基础镜像
# 使用官方 Python 3.11 的 slim（精简）版本。
# 意义：它包含了运行 Python 必需的所有环境，但去掉了不常用的工具，体积比完整版小几百 MB。
FROM python:3.11-slim

# 2. WORKDIR：设置容器内部的工作目录
# 这相当于在容器启动后自动执行了：mkdir /app && cd /app。
# 意义：以后所有的操作（复制文件、安装依赖、运行代码）都会在这个目录下进行，保证环境整洁。
WORKDIR /app

# 3. RUN：安装 Linux 系统级的工具
# apt-get 是 Linux 的包管理工具。我们安装 build-essential（包含 gcc 编译器）和 curl。
# 意义：有些 Python 库（如某些版本的 torch 或 numpy）在安装时需要编译 C++ 代码，所以必须装这些。
# 最后一行 rm -rf 是为了删掉临时安装包，让镜像更瘦。
# ==============================================================================
# 1. 替换新版配置源：把系统的国外下载地址替换成清华源（|| true 表示如果没这个文件就跳过，不报错）
# 2. 替换老版配置源：给老版本配置文件也做一遍替换，上个双重保险
# 3. 更新软件菜单：去清华服务器获取最新的软件列表（--fix-missing 表示遇到小错误尽力修复，别直接罢工）
# 4. 安装核心软件：下载 build-essential (C/C++编译大礼包) 和 curl (联网下载工具)。-y 表示全程自动点“确定”
# 5. 打扫战场：安装完后，把第3步下载的临时“菜单”全删掉，让最终的 Docker 镜像更瘦、不占空间
# ==============================================================================
# RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources || true \
#     && sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list || true \
#     && apt-get update --fix-missing \
#     && apt-get install -y --fix-missing \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update --fix-missing \
    && apt-get install -y --fix-missing \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. COPY：先单独复制依赖清单
# 这一步非常关键！我们只把 requirements.txt 复制到容器的当前目录（.）。
# 意义（层缓存机制）：Docker 构建是分层的。只要你的 requirements.txt 没变，
# 下次构建时，Docker 就会直接跳过下面耗时的 pip install 步骤，实现秒速构建。
COPY requirements.txt .

# 5. RUN：安装 Python 依赖库
# --no-cache-dir 告诉 pip 不要保存下载的缓存文件，这样能减小最终镜像的体积。
# debugpy：VS Code 的 Python 调试器核心库。必须安装，否则无法在容器内设置断点和调试。
# ipdb：增强版的 Python 调试器（基于 pdb），支持彩色输出和更好的交互体验。如果你喜欢用命令行调试，可以用 import ipdb; ipdb.set_trace() 设置断点。
# ipython：交互式 Python shell，比默认的 python shell 功能更强（自动补全、语法高亮等）。方便你在容器内快速测试代码片段。
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install --no-cache-dir debugpy ipdb ipython

# 6. COPY：最后复制剩下的所有代码
# 把宿主机当前文件夹（.）下的所有代码和资源都搬进容器的当前目录（.）。
# 意义：由于我们在第一步配置了 .dockerignore，所以这里会自动跳过 .git、venv 等垃圾文件夹。
COPY . .

# 7. CMD：默认运行的命令（由于你在 docker-compose.yml里也写了command，那个会优先生效）
# 意义：如果有人直接 docker run 这个镜像而不带参数，容器就会自动运行这个测试脚本。
CMD ["python", "part_1/demo_mha_shapes.py"]
