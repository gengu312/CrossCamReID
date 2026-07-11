# RF-DETR 显卡服务器实验说明

## 1. 当前结论

RF-DETR 已经作为可选检测后端接入项目，但目前还不能作为正式识别后端使用。

截至 2026-07-11，本机检查及单周期试验结果如下：

- YOLO 已就绪，当前项目权重为 `runs_yolo\pipe_yolov8n_hybrid_0710_v2\weights\best.pt`。
- RF-DETR 代码入口、训练、评估和对比脚本已经具备。
- 默认 YOLO 环境没有安装 `rfdetr`，RF-DETR 试验使用独立环境 `D:\SoftWare\PythonEnvs\CrossCamReID-rfdetr-cpu`。
- 独立环境使用 Python 3.11、RF-DETR 1.8.3 和 CPU 版 PyTorch，`torch.cuda.is_available()` 为 `False`。
- `datasets\pipe_rfdetr\` 已生成并通过训练前检查。
- 已生成单周期试验权重 `runs_rfdetr\pipe_rfdetr_nano_cpu_smoke\checkpoint_best_regular.pth`，但综合效果仍低于当前 YOLO。

所以现阶段继续以 YOLO 为默认后端。RF-DETR 只作为下一阶段的对比实验，不直接替换 YOLO。

## 2. 已有素材能否复用

可以复用，不需要重新拍摄和重新画检测框。

当前建议用于 RF-DETR 实验的数据集是：

```text
datasets\pipe_yolo_candidate_hybrid_0710_v2\
```

当前包含：

- 训练集：220 张图片和 220 个标注文件。
- 验证集：53 张图片和 53 个标注文件。
- 类别：`pipe`，共 1 类。

需要注意：

- 图片和 YOLO 方框标注可以转换后用于 RF-DETR 检测训练。
- YOLO 训练得到的 `best.pt` 不能转换成 RF-DETR 权重，RF-DETR 必须重新训练。
- 已录制的视频主要用于跟踪和接力测试，不会自动参与检测训练；只有抽帧并标注后才能加入训练集。
- 如果以后改成实例分割，现有方框标注不够，还需要补充每根目标的轮廓或掩码标注。
- `datasets/`、`dataset_raw/` 和训练输出已被 Git 忽略，克隆 GitHub 仓库后不会自动得到这些本地数据，需要单独传到服务器。

## 3. 显卡服务器的用途边界

显卡服务器适合完成：

1. RF-DETR 训练。
2. 验证集离线评估。
3. 与现有 YOLO 结果做同数据对比。

远程服务器通常不能直接读取本地电脑上的 USB 摄像头。因此，仅找到远程显卡服务器并不能直接完成本地双摄实时演示。

训练完成后有三种使用方式：

1. 把 RF-DETR 权重复制回本机，用本机运行。当前本机只有 CPU，能够尝试，但实时速度可能较慢。
2. 使用带 NVIDIA 显卡的本地电脑，并把两个摄像头接到这台电脑，最适合正式演示。
3. 后续增加网络视频传输和远程推理服务。当前项目还没有实现这部分。

## 4. 服务器准备

建议先使用 RF-DETR Nano 做第一轮实验。服务器应满足：

- Python 3.10 或更高版本。
- NVIDIA 显卡及正常可用的驱动。
- 安装与服务器 CUDA 环境匹配的 GPU 版 PyTorch。
- 能够访问项目代码、训练数据和预训练权重下载地址。

在项目根目录检查显卡：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

只有输出 `True` 并显示 NVIDIA 显卡名称后，才开始正式训练。

建议为 RF-DETR 创建单独的 Python 环境，避免影响当前 YOLO 演示环境。先根据服务器 CUDA 版本安装 GPU 版 PyTorch，再安装项目依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-rfdetr.txt
python -c "import torch, rfdetr; print('cuda=', torch.cuda.is_available())"
```

## 5. 把数据传到服务器

先克隆项目代码，再把本机下面整个目录复制到服务器同名位置：

```text
datasets\pipe_yolo_candidate_hybrid_0710_v2\
```

复制后确认以下文件存在：

```text
datasets\pipe_yolo_candidate_hybrid_0710_v2\data.yaml
datasets\pipe_yolo_candidate_hybrid_0710_v2\images\train\
datasets\pipe_yolo_candidate_hybrid_0710_v2\images\val\
datasets\pipe_yolo_candidate_hybrid_0710_v2\labels\train\
datasets\pipe_yolo_candidate_hybrid_0710_v2\labels\val\
```

## 6. 导出并检查 RF-DETR 数据集

在 Windows 服务器执行：

```powershell
.\export_rfdetr_dataset.bat -YoloRoot datasets\pipe_yolo_candidate_hybrid_0710_v2 -OutputRoot datasets\pipe_rfdetr -Clean -TestFromVal
.\train_rfdetr.bat -DatasetDir datasets\pipe_rfdetr -CheckOnly -SkipInstall
```

在 Linux 服务器执行等价命令：

```bash
python src/export_yolo_to_coco.py --yolo-root datasets/pipe_yolo_candidate_hybrid_0710_v2 --output-root datasets/pipe_rfdetr --class-names pipe --category-id-offset 1 --clean --test-from-val
python src/train_rfdetr.py --dataset-dir datasets/pipe_rfdetr --check-only
```

检查通过后，`datasets\pipe_rfdetr\` 应包含 `train`、`valid`、`test` 和对应的 `_annotations.coco.json`。

`-TestFromVal` 只是为了第一轮流程验证，测试集和验证集内容相同，不能作为最终独立测试结果。后续应补一份未参与训练和调参的独立测试集。

## 7. 第一轮训练

先训练 Nano 模型 10 个周期，确认流程和显存都正常：

```powershell
.\train_rfdetr.bat -DatasetDir datasets\pipe_rfdetr -OutputDir runs_rfdetr\pipe_rfdetr_nano -ModelSize nano -Epochs 10 -BatchSize 4 -GradAccumSteps 4 -NumClasses 1
```

Linux 等价命令：

```bash
python src/train_rfdetr.py --dataset-dir datasets/pipe_rfdetr --output-dir runs_rfdetr/pipe_rfdetr_nano --model-size nano --epochs 10 --batch-size 4 --grad-accum-steps 4 --num-classes 1
```

如果显存不足，先把 `BatchSize` 改为 `2` 或 `1`。第一轮正常完成后，再根据验证结果决定是否增加训练周期，不直接训练更大的模型。

训练完成后保留 `runs_rfdetr\pipe_rfdetr_nano\` 下的最佳 `.pth` 权重文件。

## 8. 评估和对比

把下面的权重路径替换成训练实际生成的最佳权重文件：

```powershell
.\evaluate_rfdetr.bat -DatasetRoot datasets\pipe_yolo_candidate_hybrid_0710_v2 -ModelSize nano -Weights runs_rfdetr\pipe_rfdetr_nano\checkpoint_best_ema.pth -NumClasses 1 -Classes 0 -CollectIssues -IssueClean
```

然后使用同一验证集和当前 YOLO 权重运行统一对比：

```powershell
.\evaluate_detectors.bat -DatasetRoot datasets\pipe_yolo_candidate_hybrid_0710_v2 -YoloModel runs_yolo\pipe_yolov8n_hybrid_0710_v2\weights\best.pt -YoloData datasets\pipe_yolo_candidate_hybrid_0710_v2\data.yaml -YoloDevice 0 -RfDetrModelSize nano -RfDetrWeights runs_rfdetr\pipe_rfdetr_nano\checkpoint_best_ema.pth -RfDetrNumClasses 1 -RfDetrClasses 0
```

不能只看一张图片。至少比较：

- 密集堆叠场景的漏检数量。
- 空桌面、手和杂物场景的误检数量。
- 只露头、只露侧面和被手遮挡时的识别情况。
- 六类测试视频中的目标锁定稳定性和跳框次数。
- 推理速度是否满足实时演示要求。

只有 RF-DETR 在同一测试集上明显优于 YOLO，并且实时速度可以接受，才考虑把它改成默认后端。

## 9. 把训练结果接回项目

把服务器训练生成的最佳权重复制到本机：

```text
runs_rfdetr\pipe_rfdetr_nano\
```

本机安装 RF-DETR 依赖并执行：

```powershell
.\check_detector_backends.bat
```

只有状态同时显示“依赖已安装”和“项目权重已找到”，RF-DETR 才算真正就绪。然后可以运行：

```powershell
.\run_pipe_rfdetr.bat
```

当前 YOLO 后端和权重应继续保留，RF-DETR 实验失败或速度不够时可以立即切回 YOLO。

## 10. 本地 CPU 单周期试验结果

试验配置：

- 模型：RF-DETR Nano。
- 数据：220 张训练图、53 张验证图，共 273 张有效图片。
- 训练：1 个周期，`BatchSize=1`，`GradAccumSteps=4`。
- 硬件：Intel Core i5-12500H，CPU 训练。
- 总耗时：约 6 分 20 秒，其中包含首次下载约 349 MB 预训练权重；实际训练和验证约 5 分钟。

同一组 53 张验证图、置信度阈值 0.25 的统一结果：

| 后端 | 匹配 | 误检 | 漏检 | Precision | Recall | F1 | 平均 IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO 当前权重 | 164 | 76 | 115 | 0.683 | 0.588 | 0.632 | 0.707 |
| RF-DETR 单周期普通权重 | 173 | 218 | 106 | 0.442 | 0.620 | 0.516 | 0.690 |

结论：

- CPU 能完成 RF-DETR Nano 训练，代码、数据、训练、评估和项目加载链路均已跑通。
- RF-DETR 单周期模型召回率略高，多匹配 9 个目标，但多产生 142 个误检，综合表现不如 YOLO。
- 置信度 0.35 对单周期 RF-DETR 过高，阈值扫描后 0.25 的 F1 最好，但误检仍然偏多。
- 斜放铅笔的水平检测框面积大、相互重叠，标准检测框难以精确分开每一根目标。后续仍应比较更多训练周期，并评估 OBB 或实例分割标注。
- 当前单周期权重只用于流程验证，不替换默认 YOLO，也不用于正式摄像头演示。

试验输出位于：

```text
runs_rfdetr\pipe_rfdetr_nano_cpu_smoke\
runs_rfdetr_eval\pipe_rfdetr_nano_cpu_smoke_regular_conf005\
runs_yolo_eval\pipe_yolov8n_hybrid_0710_v2_eval_conf025_predict\
```

这些目录包含本地数据和较大的权重文件，已被 Git 忽略，不应提交到 GitHub。

当前本机如需再次验证单周期权重，应先在 PowerShell 切换到独立环境，再启动项目：

```powershell
$env:RF_HOME = "D:\SoftWare\ModelCache\Roboflow"
$env:Path = "D:\SoftWare\PythonEnvs\CrossCamReID-rfdetr-cpu\Scripts;" + $env:Path
.\run_pipe_rfdetr.bat -RfDetrWeights runs_rfdetr\pipe_rfdetr_nano_cpu_smoke\checkpoint_best_regular.pth -RfDetrConf 0.25
```

这只是本机试验入口。日常演示仍运行默认 YOLO；不要在未切换独立环境时让 Python 3.14 自动安装 RF-DETR 训练依赖。

## 11. 官方参考

- RF-DETR 仓库：<https://github.com/roboflow/rf-detr>
- RF-DETR 数据集格式：<https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/>
- PyTorch 安装选择：<https://pytorch.org/get-started/locally/>
