# iFVC 复现记录

> 本文档随复现实验同步更新。只有经过实际执行并确认成功的操作，才会进入正式步骤；尚未验证的内容统一标为“待验证”。

## 1. 复现目标

本次工作需要从干净的服务器环境出发，完成 iFVC 的环境配置、数据准备、I 帧训练、P 帧训练、编码、解码、渲染和指标汇总。最终文档应使读者无需询问作者即可复现实验，并在另一台服务器上按照同一套步骤完成复验。

本项目没有需要下载的 iFVC 预训练 checkpoint。第 0 帧模型由数据现场训练产生，后续 P 帧依赖上一帧的解码模型。LPIPS 评测所需的网络参数属于指标依赖，将在评测部分单独说明，不作为 iFVC 模型权重介绍。

## 2. 当前复现基线

### 2.1 这一步要回答什么问题

服务器是否具备复现官方环境、编译 CUDA 扩展和运行完整训练链路的基础条件？

### 2.2 已验证的服务器配置

检查日期：2026-10-14。

| 项目 | 实际结果 |
| --- | --- |
| 操作系统 | Ubuntu 20.04.4 LTS |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| NVIDIA 驱动 | 550.100 |
| `nvidia-smi` 显示的最高 CUDA 兼容版本 | 12.4 |
| CUDA 编译器 | CUDA 11.6，`nvcc 11.6.124` |
| GCC | 9.4.0 |
| Conda | 4.10.3 |
| 系统盘 | 30 GB，检查时剩余约 29 GB |
| 数据盘 | 50 GB，挂载于 `/root/autodl-tmp` |

需要区分 `nvidia-smi` 与 `nvcc` 的输出：前者显示驱动能够兼容的最高 CUDA 版本，后者才是编译项目 CUDA 扩展时实际使用的工具链。本服务器的实际编译工具为 CUDA 11.6，与 iFVC 官方环境一致。

### 2.3 检查命令

```bash
nvidia-smi
nvcc --version
cat /etc/os-release
python3 --version
conda --version
conda info --envs
gcc --version
df -h / /root/autodl-tmp
free -h
```

### 2.4 如何判断成功

以下条件需要同时成立：

- `nvcc --version` 显示 CUDA 11.6；
- GCC 主版本为 9；
- `nvidia-smi` 能识别 GPU，且没有异常占用；
- `/root/autodl-tmp` 已挂载并有足够可用空间；
- Conda 可以正常执行。

本服务器已经通过上述检查。

## 3. Conda 与 Pip 镜像

### 3.1 为什么单独配置镜像

iFVC 依赖较旧的 Python、PyTorch 和 CUDA 生态。使用默认海外源不仅下载慢，而且容易因连接中断造成不完整安装。安装时应显式指定国内镜像，并避免让已有的未知备用源参与解析。

### 3.2 已确认的初始状态

服务器原有 Pip 主源为阿里云，Conda 原有清华镜像。已从 Conda 频道列表中移除 `defaults`，加入清华 `conda-forge` 镜像，并关闭 base 环境自动激活。

检查命令：

```bash
python -m pip config list -v
conda config --show-sources
conda config --show channels
```

### 3.3 已发现的问题与处理结论

直接让旧版 Conda 同时解析全部频道，会退回体积很大的完整索引，等待时间过长。后续命令应使用 `--override-channels`，只给当前步骤真正需要的镜像频道。这既缩短求解时间，也减少依赖来源的不确定性。

清华 Conda 镜像当前没有提供本次需要的 `cudatoolkit=11.6`，因此 PyTorch 不再通过该 Conda 组合安装。下一步计划从阿里云 PyTorch wheel 镜像安装带 CUDA 11.6 运行时的 PyTorch 1.12.1。该方案尚未验证，不能作为最终命令。

## 4. 创建独立 Python 环境

### 4.1 这一步在做什么

创建名为 `iFVC` 的独立环境，并锁定官方使用的 Python 3.7.13 和 Pip 22.3.1，避免改动服务器的 base 环境。

### 4.2 已验证命令

```bash
conda create -n iFVC python=3.7.13 pip=22.3.1 -y \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

### 4.3 实际结果与成功标准

环境已成功创建在：

```text
/root/miniconda3/envs/iFVC
```

后续还需要执行以下检查，确认解释器和 Pip 版本准确无误：

```bash
conda run -n iFVC python --version
conda run -n iFVC python -m pip --version
```

当前状态：环境创建成功，版本复查待执行。

## 5. RTX 4090 与 CUDA 11.6 的兼容处理

RTX 4090 的计算能力为 8.9，而 CUDA 11.6 的 `nvcc` 不能原生生成 `sm_89`。NVIDIA 保证为同一主版本中较低计算能力生成的 cubin 可以在更高的次版本上运行，因此为 Ampere `sm_86` 编译的扩展可以在 Ada `sm_89` 上运行。

编译 iFVC 的三个 CUDA 扩展时，需要显式设置：

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

这样可以避免 PyTorch 自动检测到 8.9 后，要求 CUDA 11.6 的 `nvcc` 编译它不认识的 `sm_89`。这一结论还需要通过三个扩展的实际编译和最小 CUDA 运算共同验证。

## 6. Fork、Clone 与复现分支

### 6.1 为什么同时需要 Fork 和 Clone

Fork 在 GitHub 上建立用户拥有写权限的远程仓库，Clone 则把该仓库下载到云服务器。只 Clone 官方仓库无法把复现修改稳定保存到用户自己的 GitHub，因此本次流程先 Fork，再从 Fork Clone。

已验证的仓库关系：

```text
origin    git@github.com:usukforever/iFVC.git
upstream  https://github.com/Pomelomm/iFVC.git
```

`origin` 指向用户 Fork，用于拉取和推送；`upstream` 指向官方仓库，只用于读取官方更新。为了避免误推，已将 `upstream` 的 Push URL 设置为 `DISABLED`，并把当前仓库的默认推送目标固定为 `origin`。

### 6.2 服务器代码目录

代码位于数据盘：

```text
/root/autodl-tmp/iFVC
```

从 `/root` 进入并克隆仓库时使用：

```bash
cd /root/autodl-tmp
git clone https://github.com/usukforever/iFVC.git
cd /root/autodl-tmp/iFVC
```

### 6.3 配置官方仓库与安全推送目标

```bash
git remote add upstream https://github.com/Pomelomm/iFVC.git
git remote set-url --push upstream DISABLED
git config remote.pushDefault origin
git config push.default current
```

检查命令：

```bash
git remote -v
```

成功时，`origin` 的 Fetch 和 Push 均指向用户 Fork，`upstream` 的 Fetch 指向官方仓库，而 Push 显示为 `DISABLED`。

### 6.4 创建并推送复现分支

```bash
git switch -c reproduction
git push -u origin reproduction
git status -sb
```

已经确认远程分支创建成功，最终状态为：

```text
## reproduction...origin/reproduction
```

服务器使用仓库专用的 GitHub Deploy Key 完成 SSH 推送。该密钥只授权当前 Fork，不授权用户的其他仓库。私钥保留在服务器，不进入仓库和复现文档。

## 7. 后续待完成步骤

- 已在用户的 GitHub 账号下创建 Fork：`https://github.com/usukforever/iFVC`；
- 已确认 Fork 的默认分支为 `main`，基线提交为 `7fa3de2281c271b3f1e305aef8fb48e9b3ba2b10`；
- 已将 Fork 克隆到服务器，把官方仓库配置为只读 `upstream`，并创建远程 `reproduction` 分支；
- 安装并验证 PyTorch 1.12.1、TorchVision 0.13.1 和 TorchAudio 0.12.1；
- 安装普通 Python 依赖；
- 编译并逐个导入三个 CUDA 扩展；
- 准备数据集并验证每帧的 COLMAP 文件和图像目录；
- 先完成第 0 帧和第 1 帧的最小闭环；
- 扩展到短序列，再运行完整序列；
- 汇总质量、大小、编解码速度和渲染速度；
- 在另一台干净服务器上复验全文档。
