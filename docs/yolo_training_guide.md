# YOLO 训练操作说明

本项目把铅笔和真实铁管统一标成 `pipe` 类别。第一阶段先用铅笔模拟铁管，跑通“拍照、标注、训练、接入双摄系统”的完整流程。

## 1. 照片放哪里

原始照片先放到：

```text
D:\WorkSpace\CrossCamReID\dataset_raw\
```

建议按场景分目录：

```text
dataset_raw/
  cam1_single/
  cam1_stack/
  cam1_hand_move/
  cam2_single/
  cam2_stack/
  cam2_hand_move/
  negative/
```

`dataset_raw` 只保存原始照片，方便以后重新标注或扩充数据。

项目提供了双摄采集脚本：

```powershell
.\capture_dataset.bat -Scenario stack
```

窗口打开后，按 `Space` 或 `B` 同时保存两个摄像头当前画面。也可以按 `1` 只保存左侧，按 `2` 只保存右侧，按 `Q` 或 `Esc` 退出。

场景参数对应保存目录：

```text
-Scenario single    -> dataset_raw/cam1_single/ 和 dataset_raw/cam2_single/
-Scenario stack     -> dataset_raw/cam1_stack/ 和 dataset_raw/cam2_stack/
-Scenario hand_move -> dataset_raw/cam1_hand_move/ 和 dataset_raw/cam2_hand_move/
-Scenario negative  -> dataset_raw/negative/
```

最终目标是堆叠铁管，所以第一批照片里 `stack` 应该占最多；`single` 只用于让模型学习基本外形。

真正用于 YOLO 训练的数据放到：

```text
D:\WorkSpace\CrossCamReID\datasets\pipe_yolo\
```

训练目录结构已经固定为：

```text
datasets/pipe_yolo/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

## 2. 标注方式

推荐先用 Label Studio、Roboflow、CVAT 或 makesense.ai 标矩形框。第一版只做 YOLO detect，不做分割。

类别名固定写：

```text
pipe
```

即使用铅笔模拟，也不要写 `pencil`。这样后面换真实铁管时，代码和模型文件名都不用改。

标注导出格式选择：

```text
YOLO
```

导出后每张图片会对应一个同名 `.txt` 标注文件，例如：

```text
images/train/img_001.jpg
labels/train/img_001.txt
```

YOLO 标注文件内容类似：

```text
0 0.512 0.438 0.220 0.060
```

其中 `0` 表示 `pipe` 类别。

## 3. 数据怎么分

第一批建议 100 到 200 张。

放入训练集：

```text
datasets/pipe_yolo/images/train/
datasets/pipe_yolo/labels/train/
```

放入验证集：

```text
datasets/pipe_yolo/images/val/
datasets/pipe_yolo/labels/val/
```

比例建议：

```text
80% train
20% val
```

例子：拍 150 张，则大约 120 张放 `train`，30 张放 `val`。

如果标注工具导出的是一个图片目录和一个标签目录，可以用脚本自动分配：

```powershell
.\prepare_yolo_dataset.bat -SourceImages exported\images -SourceLabels exported\labels -Clean
```

参数说明：

```text
-SourceImages：标注工具导出的图片目录
-SourceLabels：标注工具导出的 YOLO txt 标签目录
-Clean：整理前清空旧的 train/val 图片和标签，保留 .gitkeep
-ValRatio：验证集比例，默认 0.2
-AllowNegative：允许没有 txt 标签的图片作为负样本，并生成空标签文件
```

整理完成后再运行：

```powershell
.\train_pipe_yolo.bat -CheckOnly
```

## 4. 开始训练

进入项目目录：

```powershell
cd D:\WorkSpace\CrossCamReID
```

确认 YOLO 依赖已经安装：

```powershell
D:\SoftWare\python\python.exe -m pip install -r requirements-yolo.txt
```

训练前先检查图片和标签是否成对、标签格式是否正确：

```powershell
.\train_pipe_yolo.bat -CheckOnly
```

检查通过后开始训练：

```powershell
.\train_pipe_yolo.bat
```

它内部等价于：

```powershell
D:\SoftWare\python\Scripts\yolo.exe detect train model=yolov8n.pt data=datasets/pipe_yolo/data.yaml epochs=80 imgsz=640 batch=8 device=cpu project=runs_yolo name=pipe_yolov8n
```

如果以后有 NVIDIA 显卡，并且 PyTorch CUDA 环境已经装好，可以把 `device=cpu` 改成：

```powershell
.\train_pipe_yolo.bat -Device 0
```

训练完成后，模型通常在：

```text
runs_yolo/pipe_yolov8n/weights/best.pt
```

## 5. 评估训练后的模型

先不要直接上摄像头实时跑，先离线评估验证集：

```powershell
.\evaluate_pipe_yolo.bat
```

它会做两件事：

```text
1. 运行 YOLO val，输出 precision、recall、mAP 等指标。
2. 对 datasets/pipe_yolo/images/val/ 保存预测预览图。
```

输出目录：

```text
runs_yolo_eval/
```

重点看预测预览图：堆叠管子是否尽量一根一框，是否把手、桌面、阴影误识别成 `pipe`。

如果只想保存预测预览，不跑指标：

```powershell
.\evaluate_pipe_yolo.bat -PredictOnly
```

如果只想跑指标，不保存预测图：

```powershell
.\evaluate_pipe_yolo.bat -ValOnly
```

## 6. 用训练后的模型运行双摄

训练完成后，用这个模型切到 YOLO 检测：

```powershell
.\run_crosscam.bat -PipeMode -YoloConf 0.25
```

`-PipeMode` 会优先自动加载 `runs_yolo\pipe_yolov8n\weights\best.pt`，并使用 YOLO、多目标检测和较低的目标匹配阈值，适合堆叠管子场景。窗口中出现多个检测框后，直接点击其中一根管子的框即可注册为要追踪的目标。注册后其他管子仍会显示；系统会先在注册摄像头内跟踪 `G001`，只有当 `G001` 从原摄像头丢失后，另一个摄像头才允许接力为同一个 `G001`。

如果漏检多，可以降低置信度：

```powershell
.\run_crosscam.bat -PipeMode -YoloConf 0.15
```

如果误检多，可以提高置信度：

```powershell
.\run_crosscam.bat -PipeMode -YoloConf 0.35
```

实时运行结束后，分析最新事件日志：

```powershell
.\analyze_run.bat
```

如果要确认这次是否真的完成“注册目标离开原摄像头后，在另一个摄像头接力为同一个 G001”：

```powershell
.\analyze_run.bat -RequireHandoff
```

## 7. 第一轮训练后看什么

训练完成后先看三件事：

- 静止铅笔能不能框出来。
- 堆叠时是一根一根框，还是被框成一大块。
- 换到另一个摄像头、另一个光照后是否仍能识别。

如果效果不好，不要马上改代码，优先补照片：

- 漏检静止物体：多拍静止堆叠图。
- 堆叠识别成一坨：多拍密集堆叠并逐根标注。
- 换摄像头失败：两个摄像头都要拍训练照片。
- 手或桌面被误识别：增加负样本和误检场景。
