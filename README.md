# synfuse5

五级全色锐化网络。上采样后的低分辨率多光谱图像做残差底图。彩色和全色各经过两层 3×3 卷积，得到 32 通道特征。每一级先在不重叠的 8×8 窗口里用官方 RWKV 双向扫描，再沿整行、整列用官方 Mamba 双向扫描。每个像素先由两路通道拼成一个 token，扫描后再写回两路。输出头只作用在彩色特征上，加回上采样图像。

宽度 32。五级合计 10 个 RWKV 块和 20 个 Mamba 块。

## 环境

训练和测评在 CUDA 上跑。当前实验用的是 PyTorch 2.1.2、`mamba-ssm==1.2.2`、`transformers==4.40.2`。

```bash
pip install -r requirements.txt
```

`mamba-ssm` 需要与本机 CUDA、PyTorch 匹配的编译环境。

## 数据

降分辨率训练和验证目录下面要有 `ms/` 和 `pan/`，图像为四波段。`option.yml` 里默认路径是：

- `data/yaogan/WV3_data/train128`
- `data/yaogan/WV3_data/test128`

WV2、GF2 的路径写在同文件的注释里。全分辨率无参考测评使用 `fullGF2_data` 或 `fullWV2_data_small`，目录可以是 `ms/`、`pan/`，也可以是 `test128/ms`、`test128/pan`。

## 训练

学习率是余弦退火，500 轮从 `5e-4` 降到 `5e-8`，损失是 L1，梯度裁剪为 4。

```bash
python main.py --option_path option.yml
```

每一轮验证在四波段上计算 PSNR、SSIM、SAM、ERGAS。日志里的 SAM 是角度。权重写到 `checkpoint/<algorithm>_4_<timestamp>/`，其中 `bestPSNR.pth` 是 PSNR 最高的一轮。

SwanLab 需要环境变量 `SWANLAB_API_KEY`。不用云日志时，把 `option.yml` 里的 `swanlab.enable` 改成 `False`。

## 降分辨率出图

把 `option.yml` 的 `test.model` 指到某次训练的 `bestPSNR.pth`，再运行：

```bash
python test.py
```

## 全分辨率无参考测评

输出光谱失真 \(D_\lambda\)、空间失真 \(D_s\) 和 \(\mathrm{QNR}=(1-D_\lambda)(1-D_s)\)。

```bash
python eval_fullres.py --algorithm synfuse --checkpoint checkpoint/<run>/bestPSNR.pth --data_dir data/yaogan/fullGF2_data --satellite GaoFen2
```

WorldView-2 全分辨率把 `--satellite` 换成 `WorldView2`。
