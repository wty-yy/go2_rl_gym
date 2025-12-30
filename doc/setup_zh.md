# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 18.04 或更高版本  
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 525 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n unitree-rl python=3.8
```

### 1.3 激活虚拟环境

```bash
conda activate unitree-rl
```

---

## 2. 安装依赖

### 2.1 安装 PyTorch

PyTorch 是一个神经网络计算框架，用于模型训练和推理。使用以下命令安装：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.2 安装 Isaac Gym

Isaac Gym 是 Nvidia 提供的刚体仿真和训练框架。

#### 2.2.1 下载

从 Nvidia 官网下载 [Isaac Gym](https://developer.nvidia.com/isaac-gym)。

#### 2.2.2 安装

解压后进入 `isaacgym/python` 文件夹，执行以下命令安装：

```bash
cd isaacgym/python
pip install -e .
```

#### 2.2.3 验证安装

运行以下命令，若弹出窗口并显示 1080 个球下落，则安装成功：

```bash
cd examples
python 1080_balls_of_solitude.py
```

如有问题，可参考 `isaacgym/docs/index.html` 中的官方文档。

### 2.3 安装 rsl_rl

`rsl_rl` 是一个强化学习算法库。

我们仓库中是带有新算法的 `rsl_rl`，克隆 Git 仓库：

```bash
git clone https://github.com/wty-yy/go2_rl_gym.git
```

#### 2.3.1 安装

```bash
cd rsl_rl
pip install -e .
```

### 2.4 安装 go2_rl_gym

进入目录并安装：

```bash
cd go2_rl_gym
pip install -e .
```

### 2.5 真机部署（可选）

#### 2.5.1 unitree_sdk2

C++ sdk, 编译参考[官方教程](https://github.com/unitreerobotics/unitree_sdk2?tab=readme-ov-file#environment-setup)

#### 2.5.2 unitree_sdk2_python（选择用Python部署）

```bash
conda create -n kaiwu python=3.8
conda activate kaiwu
pip3 install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

### 2.5.3 安装 unitree_cpp_deploy（选择用C++部署）

我们基于unitree_rl_lab修改的C++部署仓库，专门用于部署本仓库训练的模型 [unitree_cpp_deploy](https://github.com/wty-yy-mini/unitree_cpp_deploy)
