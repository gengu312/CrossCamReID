# CrossCamReID 双摄像头跨区域目标重识别 MVP

这是一个用于演示“两个物理摄像头之间识别同一个物体”的最小可行系统。

系统目标：

```text
摄像头 A 检测到一个物体
-> 给它分配全局 ID
-> 物体离开摄像头 A
-> 物体进入摄像头 B
-> 系统判断它和刚才离开的物体是否相似
-> 匹配成功后，在摄像头 B 继续使用同一个全局 ID
```

当前版本支持两种检测入口：

```text
motion：OpenCV 运动检测，默认稳定演示模式
yolo：Ultralytics YOLO 检测入口，用于后续接入自训练 pipe 模型
```

推荐演示方式是先让目标在画面中产生检测框，再把当前最佳检测注册成具体目标，系统保存它的轻量视觉模板，之后在两个摄像头中持续寻找与它相似的候选。最终铁管场景应逐步从 `motion` 切到 `yolo`，再用自采集数据训练 `pipe` 类别模型。

## 技术栈与版本

当前仓库包含 OpenCV MVP 稳定版，并已预留 YOLO 检测入口。当前还没有训练好的铅笔/铁管模型。

当前技术栈：

- Python 3.14.4
- OpenCV 4.13.0.92
- NumPy 2.5.0
- Pillow：用于在 OpenCV 窗口中绘制中文 UI 文本
- OpenCV MOG2 背景建模：用于运动目标检测
- Ultralytics YOLO：用于后续静止目标检测和自训练模型接入
- 简单质心/距离匹配：用于单摄像头内短时跟踪
- HSV 颜色直方图 + 颜色布局 + 边缘方向 + 形状比例：用于跨摄像头轻量 Re-ID 匹配

YOLO 版本会把“运动目标检测”替换为“训练后的铅笔/管状物目标检测模型”，但跨摄像头 ID 管理、相似度匹配和可视化框架继续复用。

## 当前 MVP 已实现

- 支持两个物理摄像头同时读取。
- 支持内置模拟双摄像头 demo，没有摄像头也能验证流程。
- 使用 OpenCV 做运动目标检测。
- 支持 `--detector yolo` 检测入口，后续可直接加载自训练模型。
- 对每个摄像头内的目标做简单跟踪。
- 为物体生成颜色、形状等轻量视觉指纹。
- 当目标从一个摄像头消失、另一个摄像头出现时，计算相似度并继承全局 ID。
- 支持主窗口 UI 按钮注册具体目标，注册后只跟踪与目标模板相似的候选。
- 支持手动框选备用模式，适合自动检测框不准时使用。
- 支持保存注册目标截图到 `runs/targets/`，方便后续整理训练素材。
- 实时窗口显示两个摄像头画面、检测框、全局 ID、局部 ID、相似度和事件日志。
- 支持无窗口 headless 验证模式，方便确认程序是否跑通。
- 支持摄像头后端自动尝试，默认依次尝试 DirectShow、MSMF 和 OpenCV 默认后端。
- 支持 ROI 检测区域，只检测画面中的指定桌面/通道区域，减少背景误检。
- 支持物理摄像头不可用时回退到内置 demo，便于演示时兜底。

## 环境安装

进入项目目录：

```powershell
cd D:\WorkSpace\CrossCamReID
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果要运行 YOLO 检测入口，再安装：

```powershell
python -m pip install -r requirements-yolo.txt
```

训练自定义铅笔/铁管模型前，先看：

```text
docs/yolo_training_guide.md
```

采集训练照片可以双击或运行：

```powershell
.\capture_dataset.bat -Scenario stack
```

默认会使用当前机器常用的 `0/2` 双摄，把照片保存到 `dataset_raw/` 下对应场景目录。

标注并整理到 `datasets/pipe_yolo/` 后，可以先检查数据集：

```powershell
.\train_pipe_yolo.bat -CheckOnly
```

如果标注工具导出的是单独的图片目录和标签目录，可以先自动整理为 train/val：

```powershell
.\prepare_yolo_dataset.bat -SourceImages exported\images -SourceLabels exported\labels -Clean
```

检查通过后再开始训练：

```powershell
.\train_pipe_yolo.bat
```

训练完成后，先离线评估模型和保存预测预览：

```powershell
.\evaluate_pipe_yolo.bat
```

评估输出会保存到 `runs_yolo_eval/`，用于检查验证集图片上每根管子的框是否合理。

本机已验证版本：

```text
Python 3.14.4
opencv-python 4.13.0.92
numpy 2.5.0
```

## 一键自检

开发或提交前可以运行：

```powershell
.\smoke_test.bat
```

它会验证 Python 编译、YOLO 数据整理/校验、模拟双摄跨镜头接力、日志分析、训练脚本检查分支和评估命令生成。这个自检不需要真实摄像头，也不会启动实际训练。

## 运行方式

### 0. 一键运行

当前电脑已验证的摄像头索引：

```text
0 = 笔记本自带摄像头
2 = RealSense RGB 摄像头
```

双击项目根目录下的：

```text
run_crosscam.bat
```

它会自动使用当前推荐参数运行：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 2 --backend dshow --roi-a 80,80,480,220 --roi-b 80,80,480,220 --warmup-frames 30 --min-area 900 --target-mode pencil --single-object --max-area-ratio 0.45 --max-shape-ratio 0.75 --min-long-side 45 --max-short-side 180 --cross-threshold 0.65 --target-threshold 0.58 --target-update-alpha 0.04 --log-dir runs
```

也可以在 PowerShell 里使用脚本参数：

```powershell
.\run_crosscam.bat -Probe
.\run_crosscam.bat -Demo
.\run_crosscam.bat -Headless -Frames 120
.\run_crosscam.bat -CamA 0 -CamB 2
.\run_crosscam.bat -Demo -AutoRegisterFirst -Headless -Frames 260
```

推荐的真实物体演示流程：

1. 启动 `.\run_crosscam.bat`。
2. 先让目标在左侧摄像头画面里轻微移动，看到检测框后点击底部的“注册左侧目标”。
3. 如果目标先出现在右侧摄像头，看到检测框后点击“注册右侧目标”。
4. 把同一个物体移动到另一个摄像头区域，观察是否继续显示 `G001`。
5. 底部事件日志区域可以用鼠标滚轮翻看较早的记录。

备用手动框选：

- 点击“手动框选左侧”从左侧画面手动框选。
- 点击“手动框选右侧”从右侧画面手动框选。
- 手动框选会弹出一个 OpenCV 小窗口，框选时主画面会暂停；按 `Enter` 或 `Space` 确认，按 `Esc` 或 `c` 取消。不要只点右上角关闭按钮。
- 键盘快捷键仍保留备用：`r/t` 自动注册，`m/n` 手动框选，`q` 退出。

注册后日志会额外记录 `target_similarity`。它表示当前候选与已注册目标模板的相似度，越接近 1 越像。

每次运行都会在 `runs/` 目录下生成一个 CSV 事件日志，例如：

```text
runs/20260624-093000-events.csv
```

跑完后可以直接分析最新日志：

```powershell
.\analyze_run.bat
```

如果要把“必须观察到注册目标跨摄像头接力”作为验收条件：

```powershell
.\analyze_run.bat -RequireHandoff
```

判断是否真的完成跨摄像头识别，重点看日志里有没有同一个全局 ID 的链路：

```text
摄像头1：已注册目标为 G001
摄像头1：匹配到目标 G001，目标相似度=1.00
摄像头2：匹配到目标 G001，目标相似度=0.xx
```

如果没有注册目标，系统会退回原来的运动目标跨摄像头匹配逻辑。此时如果只有大量 `new object`，没有 `matched`，说明只是检测到运动目标，还没有完成跨摄像头同一物体匹配。

当前一键脚本默认使用“铅笔演示模式”：

- 降低 `min-area`，避免细长的笔被过滤掉。
- 使用 `--target-mode pencil`，优先选择细长运动区域。
- 使用 `--single-object`，每个摄像头每帧只保留一个最佳目标，减少手和背景产生多个 ID。
- 使用 `--cross-threshold 0.65`，让两个摄像头光照差异较大时也更容易匹配。
- 使用 `--target-threshold 0.58`，注册目标后过滤掉不像目标的运动候选。

如果已经训练好 `pipe` 模型，使用管子多目标模式：

```powershell
.\run_crosscam.bat -PipeMode
```

`-PipeMode` 会优先自动加载 `runs_yolo\pipe_yolov8n\weights\best.pt`，切换到 YOLO、多目标检测，并保留每个摄像头最多 30 个候选框。窗口里可以直接点击某一根管子的检测框，把它注册为要追踪的目标；注册后其他管子仍会显示检测框。系统会先在注册摄像头内跟踪 `G001`，只有当 `G001` 从原摄像头丢失后，另一个摄像头才允许接力成同一个 `G001`，避免另一堆相似管子提前抢占目标 ID。

如果误匹配太多，可以提高阈值：

```powershell
.\run_crosscam.bat -CrossThreshold 0.75
```

如果仍然检测不到笔，可以继续降低面积阈值：

```powershell
.\run_crosscam.bat -MinArea 500
```

### 1. 无摄像头模拟测试

这个命令会打开一个窗口，显示两个模拟摄像头。一个“铅笔状物体”会先出现在左侧摄像头，再出现在右侧摄像头，系统会把它匹配成同一个全局 ID。

```powershell
python src\crosscam_mvp.py --demo
```

自动跑完并退出：

```powershell
python src\crosscam_mvp.py --demo --auto-register-first --frames 260 --require-match
```

无窗口验证：

```powershell
python src\crosscam_mvp.py --demo --auto-register-first --headless --frames 260 --require-match
```

如果成功，会看到类似输出：

```text
Processed frames: 260
Cross-camera match observed: yes
Cam2: target matched G001, target_sim=0.xx
```

### 2. 探测本机摄像头

```powershell
python src\crosscam_mvp.py --probe --probe-max 5
```

如果两个摄像头都可用，通常会看到类似输出：

```text
index 0: OK
index 1: OK
```

如果只看到一个 `OK`，说明当前电脑只开放了一个摄像头，或者另一个摄像头被微信、腾讯会议、浏览器、系统相机等程序占用。此时真实双摄命令会启动失败，可以先用 `--demo` 验证算法流程。

默认会自动尝试多个 OpenCV 摄像头后端。也可以手动指定：

```powershell
python src\crosscam_mvp.py --probe --backend dshow
python src\crosscam_mvp.py --probe --backend msmf
```

### 3. 使用两个真实摄像头运行

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1
```

关闭窗口：

```text
按 q 或 Esc
```

也可以无窗口跑一段，确认双摄像头采集没有问题：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --headless --frames 90
```

如果现场只识别到一个摄像头，但仍然想先展示系统流程，可以使用回退模式：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --fallback-demo
```

### 4. 限定检测区域提升稳定性

真实场景中建议只检测桌面或目标通过区域，减少人手、屏幕、背景运动造成的误检。ROI 参数格式是 `x,y,w,h`，坐标基于程序内部缩放后的 `640x360` 画面。

示例：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --roi-a 60,80,520,220 --roi-b 60,80,520,220
```

如果启动前背景还在变化，可以增加背景预热帧数：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --warmup-frames 45
```

### 5. YOLO 检测入口

当前 YOLO 入口已经接入，但还没有训练好的 `pipe` 模型。预训练 COCO 模型通常不认识铅笔/铁管，所以它主要用于验证代码链路，正式效果需要后续训练。

安装 YOLO 依赖：

```powershell
python -m pip install -r requirements-yolo.txt
```

用预训练模型测试入口：

```powershell
python src\crosscam_mvp.py --demo --detector yolo --yolo-model yolov8n.pt --headless --frames 5
```

后续有自训练模型后：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 2 --backend dshow --detector yolo --yolo-model runs_yolo\pipe_yolov8n\weights\best.pt --max-detections 30
```

一键脚本也支持：

```powershell
.\run_crosscam.bat -PipeMode
```

## 真实物体测试方法

建议先用容易区分的物体，例如黄色铅笔、彩色笔、带贴纸的小物体。

测试步骤：

1. 固定两个摄像头，分别对准两个不同区域。
2. 保持背景尽量静止，避免人手、屏幕、窗帘等大面积运动干扰。
3. 启动程序：

   ```powershell
   python src\crosscam_mvp.py --cam-a 0 --cam-b 1
   ```

4. 移动物体，让系统检测到它并显示检测框。
5. 如果是多目标管子模式，直接点击要追踪的那根管子的检测框；普通演示模式可以点击底部“注册左侧目标”。
6. 把物体移出摄像头 A，再放入摄像头 B 区域。
7. 观察右侧摄像头是否显示同一个全局 ID，例如 `G001`。
8. 查看底部事件日志，正常情况下会出现：

   ```text
   摄像头1：已注册目标为 G001
   摄像头1：匹配到目标 G001，目标相似度=1.00
   摄像头2：匹配到目标 G001，目标相似度=0.xx
   ```

如果没有匹配成功，可以先尝试：

- 换颜色更明显的物体。
- 给物体贴一个彩色贴纸。
- 减少背景运动。
- 注册前先轻微移动目标，让检测框尽量贴住物体。
- 如果自动检测框不准，再点击手动框选按钮，尽量框紧一点，少包含背景和手。
- 调低目标模板阈值，例如：

  ```powershell
  python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --target-threshold 0.50
  ```

## 当前方案的限制

当前 MVP 是“传统视觉 + 相似度匹配”，适合快速演示系统流程，但不是最终高鲁棒方案。

主要限制：

- 背景运动太多时容易误检。
- 两个物体外观非常相似时，可能匹配错误。
- 光照差异大时，颜色特征会变得不稳定。
- 铅笔这种细长小物体在画面中太小时，检测效果会下降。
- 注册目标时如果框里包含太多背景或手，后续匹配会变乱。
- 当前系统只能做概率判断，不能保证完全一样的两个物体 100% 区分。

## 四阶段完善路线

详细路线见 [四阶段完善路线](docs/four_stage_plan.md)。当前项目按下面顺序推进：

### 阶段 1：先把每根管子检测稳定

目标是让系统能在堆叠铅笔/管子中尽量框出每一个可见目标。

已完成：

- 已接入 YOLO 检测入口，支持静止目标检测。
- `-PipeMode` 会优先加载训练好的 `runs_yolo\pipe_yolov8n\weights\best.pt`。
- 支持点击检测框注册要追踪的目标。
- YOLO 检测已复用管状物几何过滤参数，过滤过大、过短、过粗的误检框。

继续要做：

- 补充堆叠、遮挡、手拿取、只露头、只露侧面的照片。
- 标注时按“每一根可见管子/铅笔”单独框，不要只框整堆。
- 每轮训练后用验证集检查漏检、框偏大、把多根框成一根的问题。

### 阶段 2：增强单摄像头内多目标追踪

目标是减少同一个摄像头内的 ID 跳变。

处理方法：

- 已在当前轻量 tracker 上加入短时运动预测，短暂遮挡或临时丢框时继续沿原方向寻找。
- 后续可接入 ByteTrack、BoT-SORT 或 DeepSORT。
- 通过日志统计 ID 切换次数，而不是只靠肉眼判断。

### 阶段 3：增强跨摄像头 ReID

目标是同一根管子从一个摄像头移到另一个摄像头时，尽量保持同一个全局 ID。

处理方法：

- 当前先使用颜色、边缘、形状等轻量特征做相似度，并保留注册目标的少量可靠模板。
- 后续可以接入 ResNet、CLIP、DINOv2 这类图像特征。
- 对注册目标保存多张模板，覆盖正面、侧面、手拿取、光照变化。

### 阶段 4：利用 RealSense 深度辅助

目标是在堆叠、遮挡、管子外观相似时，增加空间信息，降低纯 RGB 判断的不稳定。

处理方法：

- 使用深度估计管子的前后层次和距离变化。
- 对拿取动作记录目标从堆中离开的空间轨迹。
- 用 RGB 检测结果加深度位置共同判断目标是否连续。

## 项目定位

本项目不是简单的“OpenCV 识别铅笔”，而是：

```text
基于双摄像头视觉的跨区域目标检测与重识别跟踪系统
```

当前 MVP 已经完成系统闭环：

```text
双摄像头采集
-> 目标检测
-> 单摄像头跟踪
-> 视觉特征提取
-> 跨摄像头相似度匹配
-> 全局 ID 继承
-> 实时可视化展示
```

后续主要工作是把检测和重识别算法从 OpenCV 传统方法升级为 YOLO + 深度 Re-ID，从而提高复杂场景下的稳定性。
