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

当前版本先使用 OpenCV 的传统视觉方法完成闭环，不依赖训练模型，方便快速演示和调试。后续可以把检测模块替换成 YOLO，把重识别模块替换成深度学习 Re-ID 特征。

## 技术栈与版本

当前仓库是 OpenCV MVP 版本，不是 YOLO 训练版。

当前技术栈：

- Python 3.14.4
- OpenCV 4.13.0.92
- NumPy 2.5.0
- OpenCV MOG2 背景建模：用于运动目标检测
- 简单质心/距离匹配：用于单摄像头内短时跟踪
- HSV 颜色直方图 + 形状比例：用于跨摄像头轻量 Re-ID 匹配

后续 YOLO 版本会把“运动目标检测”替换为“训练后的铅笔/文具目标检测模型”，但跨摄像头 ID 管理、相似度匹配和可视化框架可以继续复用。

## 当前 MVP 已实现

- 支持两个物理摄像头同时读取。
- 支持内置模拟双摄像头 demo，没有摄像头也能验证流程。
- 使用 OpenCV 做运动目标检测。
- 对每个摄像头内的目标做简单跟踪。
- 为物体生成颜色、形状等轻量视觉指纹。
- 当目标从一个摄像头消失、另一个摄像头出现时，计算相似度并继承全局 ID。
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

本机已验证版本：

```text
Python 3.14.4
opencv-python 4.13.0.92
numpy 2.5.0
```

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
python src\crosscam_mvp.py --cam-a 0 --cam-b 2 --backend dshow --roi-a 80,80,480,220 --roi-b 80,80,480,220 --warmup-frames 45 --min-area 5000 --cross-threshold 0.72
```

也可以在 PowerShell 里使用脚本参数：

```powershell
.\run_crosscam.bat -Probe
.\run_crosscam.bat -Demo
.\run_crosscam.bat -Headless -Frames 120
.\run_crosscam.bat -CamA 0 -CamB 2
```

### 1. 无摄像头模拟测试

这个命令会打开一个窗口，显示两个模拟摄像头。一个“铅笔状物体”会先出现在左侧摄像头，再出现在右侧摄像头，系统会把它匹配成同一个全局 ID。

```powershell
python src\crosscam_mvp.py --demo
```

自动跑完并退出：

```powershell
python src\crosscam_mvp.py --demo --frames 260 --require-match
```

无窗口验证：

```powershell
python src\crosscam_mvp.py --demo --headless --frames 180 --require-match
```

如果成功，会看到类似输出：

```text
Processed frames: 180
Cross-camera match observed: yes
Cam2: matched G001, sim=0.97
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

## 真实物体测试方法

建议先用容易区分的物体，例如黄色铅笔、彩色笔、带贴纸的小物体。

测试步骤：

1. 固定两个摄像头，分别对准两个不同区域。
2. 保持背景尽量静止，避免人手、屏幕、窗帘等大面积运动干扰。
3. 启动程序：

   ```powershell
   python src\crosscam_mvp.py --cam-a 0 --cam-b 1
   ```

4. 把物体放到摄像头 A 区域内并移动，让系统框选到它。
5. 把物体移出摄像头 A。
6. 在 8 秒内把同一个物体放入摄像头 B 区域。
7. 观察右侧摄像头是否显示同一个全局 ID，例如 `G001`。
8. 查看底部事件日志，正常情况下会出现：

   ```text
   Cam1: new object G001
   Cam1: G001 left view
   Cam2: matched G001, sim=0.xx
   ```

如果没有匹配成功，可以先尝试：

- 换颜色更明显的物体。
- 给物体贴一个彩色贴纸。
- 减少背景运动。
- 缩短从摄像头 A 到摄像头 B 的转移时间。
- 调低匹配阈值，例如：

  ```powershell
  python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --cross-threshold 0.65
  ```

## 当前方案的限制

当前 MVP 是“传统视觉 + 相似度匹配”，适合快速演示系统流程，但不是最终高鲁棒方案。

主要限制：

- 背景运动太多时容易误检。
- 两个物体外观非常相似时，可能匹配错误。
- 光照差异大时，颜色特征会变得不稳定。
- 铅笔这种细长小物体在画面中太小时，检测效果会下降。
- 当前系统只能做概率判断，不能保证完全一样的两个物体 100% 区分。

## 接下来最应该实现的方案

### 阶段 1：增强 OpenCV MVP

优先级最高，适合短期继续完善演示：

- 已完成：增加摄像头后端自动尝试，提升不同 Windows 摄像头驱动下的可用性。
- 已完成：增加手动 ROI 区域，只检测桌面或指定区域，减少误检。
- 已完成：增加物理摄像头不可用时的 demo 回退模式，保证演示入口可用。
- 增加“进入区 / 离开区”判断，让跨摄像头匹配更稳定。
- 增加目标截图保存，方便后续训练 YOLO。
- 增加 CSV 日志，记录时间、摄像头、全局 ID、相似度。
- 增加参数配置文件，不用每次在命令行里写参数。

### 阶段 2：YOLO 目标检测

把当前的运动检测替换成 YOLO：

```text
摄像头画面
-> YOLO 检测铅笔/文具/指定物体
-> 输出检测框
-> 后续继续使用当前跟踪和跨摄像头匹配逻辑
```

需要采集并标注数据：

- 每个摄像头都拍一些样本。
- 包含不同角度、光照、背景、手持移动状态。
- 先采集 300 到 800 张图即可做一个小模型。

### 阶段 3：更强的 Re-ID 特征

当前使用颜色和形状特征，后续可以升级为：

- ResNet 图像 embedding
- CLIP 图像 embedding
- DINOv2 图像 embedding

这样能更好地区分不同物体，尤其是颜色接近但纹理、形状细节不同的情况。

### 阶段 4：最稳演示兜底

如果演示要求非常稳定，可以给物体贴小标签：

- ArUco
- AprilTag
- 彩色编号贴纸

这样系统可以直接读取唯一 ID，跨摄像头识别会非常稳定。可以作为工程演示兜底方案，同时保留无标记 Re-ID 作为研究方向。

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
