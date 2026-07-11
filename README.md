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

当前版本支持三种检测入口：

```text
motion：OpenCV 运动检测，默认稳定演示模式
yolo：Ultralytics YOLO 检测入口，用于后续接入自训练 pipe 模型
rfdetr：Roboflow RF-DETR 检测入口，用作后续可选检测后端对比
```

推荐演示方式是先让目标在画面中产生检测框，再把当前最佳检测注册成具体目标，系统保存它的轻量视觉模板，之后在两个摄像头中持续寻找与它相似的候选。最终铁管场景应逐步从 `motion` 切到 `yolo`，再用自采集数据训练 `pipe` 类别模型；`rfdetr` 作为后续可选检测后端，用来和 YOLO 做效果对比。

## 技术栈与版本

当前仓库包含 OpenCV MVP 稳定版、YOLO 检测入口和第一版自训练 `pipe` 模型；RF-DETR 已作为可选检测后端接入，但还没有本地训练权重。

当前技术栈：

- Python 3.14.4
- OpenCV 4.13.0.92
- NumPy 2.5.0
- Pillow：用于在 OpenCV 窗口中绘制中文 UI 文本
- OpenCV MOG2 背景建模：用于运动目标检测
- Ultralytics YOLO：用于后续静止目标检测和自训练模型接入
- Roboflow RF-DETR：作为可选检测后端，用于后续和 YOLO 做对比
- 简单质心/距离匹配：用于单摄像头内短时跟踪
- HSV 颜色直方图 + 颜色布局 + 边缘方向 + 局部纹理 + 形状比例：用于跨摄像头轻量 Re-ID 匹配

YOLO 版本会把“运动目标检测”替换为“训练后的铅笔/管状物目标检测模型”，但跨摄像头 ID 管理、相似度匹配和可视化框架继续复用。

## 当前 MVP 已实现

- 支持 1 到 3 个物理摄像头同时读取。
- 支持内置模拟 1-3 路摄像头 demo，没有摄像头也能验证流程。
- 使用 OpenCV 做运动目标检测。
- 支持 `--detector yolo` 检测入口，后续可直接加载自训练模型。
- 支持 `--detector rfdetr` 可选检测后端，便于后续对比 RF-DETR 和 YOLO。
- 对每个摄像头内的目标做简单跟踪。
- 为物体生成颜色、形状等轻量视觉指纹。
- 当目标从一个摄像头消失、另一个摄像头出现时，计算相似度并继承全局 ID。
- 支持主窗口 UI 按钮注册具体目标，注册后只跟踪与目标模板相似的候选。
- 支持手动框选备用模式，适合自动检测框不准时使用。
- 支持保存注册目标截图到 `runs/targets/`，方便后续整理训练素材。
- 实时窗口显示两个摄像头画面、检测框、全局 ID、局部 ID、相似度和事件日志。
- 支持无窗口 headless 验证模式，方便确认程序是否跑通。
- 支持摄像头后端自动尝试，默认依次尝试 DirectShow、MSMF 和 OpenCV 默认后端。
- 支持 A/B/C 三路 ROI 检测区域，只检测画面中的指定桌面/通道区域，减少背景误检。
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

项目当前问题、改进优先级和阶段验收标准见：

```text
docs/project_improvement_plan.md
```

本轮简要工作总结：

```text
docs/work_summary_20260710.md
```

增量训练的详细处理记录和新旧模型对比见：

```text
docs/yolo_incremental_20260710.md
```

### 本地数据、训练集和模型位置

所有原始照片、原始标签和回放视频统一归档在 `dataset_raw/` 这个总目录中；整理后的训练集放在 `datasets/`，训练权重和评估结果放在 `runs_yolo/`、`runs_yolo_eval/` 等输出目录，避免原始素材和生成结果相互覆盖。

| 内容 | 本地位置 | 当前数量或用途 |
| --- | --- | --- |
| 上一次分类原图归档 | `dataset_raw/import_20260626/` | 151 张，按 5 类保留原始目录结构 |
| 上一次原始照片和标签 | `dataset_raw/to_label_20260626/images/`、`labels/` | 151 张，全部保留并用于本轮训练 |
| 本次 0710 原始照片和标签 | `dataset_raw/to_label_next/images/`、`labels/` | 262 张，自动预标注后保留完整原图和标签 |
| 本次重复照片隔离区 | `dataset_raw/to_label_next/duplicates/` | 4 张，与正式图片重复，不进入训练 |
| 本次标注质量报告 | `dataset_raw/to_label_next/hybrid_label_report.csv` | 标记 `trusted` / `review`，`review` 不进入本轮训练 |
| 六类回放视频 | `dataset_raw/replay_videos/` | 只用于检测、跟踪和阈值测试，不直接训练 |
| 本轮合并训练源 | `dataset_raw/training_sources_0710_hybrid_v2/` | 旧图 151 张 + 筛选新图 122 张 |
| 原基础训练集 | `datasets/pipe_yolo/` | 151 张，train=121、val=30 |
| 本轮候选训练集 | `datasets/pipe_yolo_candidate_hybrid_0710_v2/` | 273 张，train=220、val=53 |
| 原基础模型 | `runs_yolo/pipe_yolov8n/weights/best.pt` | 保留为回退模型 |
| 本轮增量模型 | `runs_yolo/pipe_yolov8n_hybrid_0710_v2/weights/best.pt` | 当前 PipeMode 首选模型 |
| 照片评估结果 | `runs_yolo_eval/` | 新旧模型验证结果 |
| 视频对比结果 | `runs_detector_compare/0710_video_comparison/` | 六段视频和阈值对比 |

`dataset_raw/`、训练图片/标签、`runs*/` 和 `*.pt` 均在 `.gitignore` 中，只保存在本机，不会作为纯素材或大模型文件提交到 Git。README、脚本和数据处理说明可以正常提交。

采集训练照片可以双击或运行：

```powershell
.\capture_dataset.bat -Scenario stack
```

默认会按 `1,3,2,0,4,5` 的顺序自动寻找两个摄像头，优先使用 `1/3` 外接双摄，并把照片保存到 `dataset_raw/` 下对应场景目录。如果摄像头索引变化，可以加 `-CamA` / `-CamB` 手动指定。

采集前如果只想确认会使用哪两个摄像头和保存目录，可以先运行：

```powershell
.\capture_dataset.bat -PrintOnly
```

如果是绿色铅笔/管子照片，可以先用自动初标减少手工框选量：

```powershell
.\auto_label_pipes.bat -Images dataset_raw\to_label_20260626\images -Labels dataset_raw\to_label_20260626\labels -Preview -Overwrite
```

自动初标适合做第一版标签，仍需要人工打开预览图检查，尤其是堆叠、遮挡和只露头的照片。

本次 0710 素材可以使用混合几何预标注入口。它会按负样本、单根和约 7 根堆叠场景生成标签，并输出逐图质量报告：

```powershell
.\auto_label_pipes_hybrid.bat -Images dataset_raw\to_label_next\images -Labels dataset_raw\to_label_next\labels -Report dataset_raw\to_label_next\hybrid_label_report.csv -Overwrite
```

质量报告中标记为 `review` 的图片不应直接进入训练集。普通水平框在斜放、交叉和紧密平行场景会高度重叠，后续可增加 YOLO OBB 或实例分割路线改善这一问题。

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
.\evaluate_pipe_yolo.bat -CheckOnly
```

```powershell
.\evaluate_pipe_yolo.bat
```

评估输出会保存到 `runs_yolo_eval/`，包括验证指标、预测预览图、预测标签和 `analysis.csv`。其中 `analysis.csv` 会统计每张验证图的匹配数、漏检数、误检数和框偏大数量，方便判断模型主要差在哪里。

可以把漏检、误检、框偏大的图片单独收集出来复盘：

```powershell
.\evaluate_pipe_yolo.bat -CollectIssues -IssuePreview -IssueClean
```

默认输出到 `runs_yolo_eval\pipe_yolov8n_eval_predict\issue_samples\`，里面会有原图、真实标签、预测标签和带框预览图。

如果已经有 `analysis.csv`，也可以只收集问题样本：

```powershell
.\collect_yolo_issues.bat -Preview -Clean
```

如果已经有预测标签，只想重新分析检测质量：

```powershell
.\analyze_yolo_eval.bat
```

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

双击项目根目录下的：

```text
run_crosscam.bat
```

它会自动探测并选择当前可用的两个摄像头，再使用推荐参数运行：

```powershell
python src\crosscam_mvp.py --cam-a auto --cam-b auto --camera-scan-order 1,3,2,0,4,5 --backend dshow --roi-a 80,80,480,220 --roi-b 80,80,480,220 --warmup-frames 30 --min-area 900 --target-mode pencil --single-object --max-area-ratio 0.45 --max-shape-ratio 0.75 --min-long-side 45 --max-short-side 180 --cross-threshold 0.65 --target-threshold 0.58 --target-update-alpha 0.04 --log-dir runs
```

也可以在 PowerShell 里使用脚本参数：

```powershell
.\run_crosscam.bat -Probe
.\run_crosscam.bat -Probe -ProbeMax 10
.\run_crosscam.bat
.\select_cameras.bat
.\run_crosscam.bat -SelectCameras
.\run_crosscam.bat -Demo
.\run_crosscam.bat -Demo -CameraIndexes "1,2,3"
.\run_crosscam.bat -Headless -Frames 120
.\run_crosscam.bat -CameraIndexes "1,3"
.\run_crosscam.bat -CameraIndexes "1,2,3"
.\run_crosscam.bat -CameraIndexes "1,3" -FallbackDemo
.\run_crosscam.bat -CamA 1 -CamB 2
.\run_crosscam.bat -ViewOrder BA
.\run_crosscam.bat -FlipBoth
.\run_crosscam.bat -ShowTrails
.\run_crosscam.bat -CameraScanOrder "2,1,0,3,4,5"
.\run_crosscam.bat -Demo -AutoRegisterFirst -Headless -Frames 260
```

如果希望程序退出后自动分析本次日志，可以加 `-AnalyzeAfterRun`：

```powershell
.\run_crosscam.bat -Demo -AutoRegisterFirst -Headless -Frames 260 -RequireMatch -AnalyzeAfterRun -AnalyzeRequireHandoff -AnalyzeMinTargetMatches 2 -AnalyzeMaxNewAfterRegister 0 -AnalyzeMinCrossCameraIds 1 -AnalyzeMaxTargetSwitches 0 -AnalyzeMaxTargetDistance 120
```

自动分析会默认保存结构化结果到 `runs\latest-summary.json`；如果要改保存位置，可以加 `-AnalyzeSummaryJson 路径`。

如果两个摄像头摆放和窗口左右相反，用 `-ViewOrder BA` 交换显示左右；默认 `AB` 表示摄像头 A 在左、摄像头 B 在右。三摄模式下 `BA` 会交换前两路，第三路保持在后面。交换后，点击左侧画面和“注册左侧目标”按钮也会对应新的左侧画面。

双击或直接运行 `.\run_crosscam.bat` 会默认打开摄像头选择窗口，并优先勾选 `1,3` 两个外接摄像头。也可以运行 `.\select_cameras.bat` 或 `.\run_crosscam.bat -SelectCameras` 打开选择窗口。窗口会显示当前可用摄像头的预览图，勾选 1 到 3 个后启动，也可以直接选择检测后端、PipeMode、水平翻转、交换左右显示、轨迹线、演示兜底、退出后自动分析和接力成功验收。若只想采集训练照片，选择采集场景和保存目录，勾选刚好 2 个摄像头后点击“采集训练照片”。通过命令行传入的 `-Backend`、检测后端、模型路径和阈值参数会继续带入启动命令。若要跳过选择窗口，可以直接用 `-CameraIndexes "1,3"` 或 `-CameraIndexes "1,2,3"` 指定要打开的摄像头。
如果没有探测到摄像头，也可以勾选“摄像头打开失败时切到演示兜底”启动内置 demo。
如果勾选“接力成功验收”，至少需要选择 2 个摄像头。

如果单个摄像头画面像照镜子一样左右反了，用 `-FlipA`、`-FlipB`、`-FlipC` 或 `-FlipBoth` 做水平翻转。运行窗口底部也有“交换左右”“翻转左侧”“翻转右侧”，三摄模式下还会有“翻转第三路”按钮，可以现场调整。三摄模式下也可以直接用“注册第三路目标”和“手动框选第三路”。

默认只显示当前锁定框和编号，不显示历史轨迹线。若需要观察移动轨迹，再加 `-ShowTrails`。

自动选择摄像头时默认优先尝试 `1,3`，再尝试 `2,0,4,5`，找到两个可用摄像头就停止。若你的摄像头索引变了，可以用 `-CameraScanOrder` 调整优先顺序；如果索引超过默认扫描范围，可以用 `-ProbeMax` 调大上限。

推荐的真实物体演示流程：

1. 启动 `.\run_crosscam.bat`。
2. 先让目标在左侧摄像头画面里轻微移动，看到检测框后点击底部的“注册左侧目标”。
3. 如果目标先出现在右侧摄像头，看到检测框后点击“注册右侧目标”。
4. 把同一个物体移动到另一个摄像头区域，观察是否继续显示 `G001`。
5. 底部事件日志区域可以用鼠标滚轮翻看较早的记录。

备用手动框选：

- 点击“手动框选左侧”从左侧画面手动框选。
- 点击“手动框选右侧”从右侧画面手动框选。
- 三摄模式下可以点击“手动框选第三路”从第三路画面手动框选。
- 手动框选会弹出一个 OpenCV 小窗口，框选时主画面会暂停；按 `Enter` 或 `Space` 确认，按 `Esc` 或 `c` 取消。不要只点右上角关闭按钮。
- 键盘快捷键仍保留备用：`1/2/3` 注册左侧/右侧/第三路，`4/5/6` 手动框选左侧/右侧/第三路，`7` 翻转第三路，`q` 退出。

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

如果要同时检查是否频繁新建 ID，可以加质量阈值：

```powershell
.\analyze_run.bat -RequireHandoff -MinTargetMatches 2 -MinCrossCameraIds 1 -MaxUniqueIds 3
```

输出里的 `新建目标事件数`、`注册后新建目标事件数`、`唯一全局 ID 数` 可以用来判断追踪是否稳定。`注册目标距上次位置` 可以用来发现同一摄像头内的突然跳框，`-MaxTargetDistance` 就是给这个指标设置上限。如果同一根目标移动时不断出现新的 `Gxxx`，通常说明检测框不稳定或 ReID 接力没有接上。

输出里的 `注册目标相似度`、`注册目标离开次数` 和 `注册目标选择` 可以进一步判断锁定是否稳定。相似度最低值过低，通常说明当前框已经不像注册目标；离开次数过多，通常说明检测框在画面内断断续续；`switch` 次数多则说明目标经常跳到更远的相似候选。

如果要更严格检查“锁定框是否稳定”，可以加：

```powershell
.\analyze_run.bat -RequireHandoff -MinTargetMatches 2 -MinTargetSimilarity 0.70 -MaxNewAfterRegister 0 -MaxTargetSwitches 2 -MaxTargetDistance 120
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

`-PipeMode` 会优先自动加载 `runs_yolo\pipe_yolov8n_hybrid_0710_v2\weights\best.pt`；如果本机没有该权重，则回退到 `runs_yolo\pipe_yolov8n\weights\best.pt`。它会切换到 YOLO、多目标检测，并保留每个摄像头最多 30 个候选框。窗口里可以直接点击某一根管子的检测框，把它注册为要追踪的目标；注册后其他管子仍会显示检测框。系统会先在注册摄像头内跟踪 `G001`，只有当 `G001` 从原摄像头丢失后，另一个摄像头才允许接力成同一个 `G001`，避免另一堆相似管子提前抢占目标 ID。

注册后会优先保持上一帧附近的目标，减少在相似管子之间跳框。PipeMode 默认使用 `-TargetStickDistance 120` 和 `-TargetSwitchMargin 0.15`；如果目标移动很快可以适当调大距离，如果框总是粘住错误目标，可以调小距离或降低切换门槛。

如果目标短暂被手挡住或检测框临时消失，可以通过 `-MaxMissed` 和 `-LostTtl` 调整保留旧 ID 的时间；遮挡时间更长时适当调大，误接回旧目标时调小。

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
python src\crosscam_mvp.py --cam-a auto --cam-b auto
```

关闭窗口：

```text
按 q 或 Esc
```

也可以无窗口跑一段，确认双摄像头采集没有问题：

```powershell
python src\crosscam_mvp.py --cam-a auto --cam-b auto --headless --frames 90
```

如果现场只识别到一个摄像头，但仍然想先展示系统流程，可以使用回退模式：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --fallback-demo
.\run_crosscam.bat -CameraIndexes "1,3" -FallbackDemo
```

### 4. 限定检测区域提升稳定性

真实场景中建议只检测桌面或目标通过区域，减少人手、屏幕、背景运动造成的误检。ROI 参数格式是 `x,y,w,h`，坐标基于程序内部缩放后的 `640x360` 画面。三摄模式下可以继续加 `--roi-c` 或启动器参数 `-RoiC` 单独设置第三路。

示例：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --roi-a 60,80,520,220 --roi-b 60,80,520,220
```

如果启动前背景还在变化，可以增加背景预热帧数：

```powershell
python src\crosscam_mvp.py --cam-a 0 --cam-b 1 --warmup-frames 45
```

### 5. 使用离线视频重复测试

将同一测试动作保存为 1 到 3 路视频后，可以反复运行检测、目标注册、跟踪和日志分析，不需要每次重新占用摄像头。本次 0710 素材实际整理为：

```text
dataset_raw/replay_videos/01_stack_static/camera_a.mp4
dataset_raw/replay_videos/02_take_one/camera_a.mp4
dataset_raw/replay_videos/03_hand_occlusion/camera_a.mp4
dataset_raw/replay_videos/04_move_rotate/camera_a.mp4
dataset_raw/replay_videos/05_handoff/camera_a.mp4
dataset_raw/replay_videos/06_negative/camera_a.mp4
```

前四类和负样本可以只录单路。真实跨摄像头交接测试需要在 `05_handoff/` 中再放入同步录制的 `camera_b.mp4`；当前尚未补录该文件。

运行双视频锁定检查：

```powershell
.\run_video_replay.bat -VideoA "D:\videos\camera_a.mp4" -VideoB "D:\videos\camera_b.mp4"
```

程序默认使用 `PipeMode` 和第一版自训练 YOLO 模型。视频播放时点击其中一根目标的检测框，播放结束后会自动分析锁定质量。任一路视频先结束时，两路回放会一起停止，建议使用相同帧率、相近长度并同时开始录制的视频。

未指定 `-LogDir` 时，每次回放会自动使用独立目录：

```text
runs/video_replay/<视频所在场景目录>/<运行时间>/
```

目录中会保存事件 CSV、`latest-summary.json`、`latest-summary.md`、目标样本和 `run_manifest.json`。运行清单记录视频绝对路径、帧率、总帧数、检测模型参数、实际处理帧数和跨摄像头匹配结果，后续可以确认某份结果对应哪组素材和参数。需要指定输出位置时可显式传入 `-LogDir`，程序不会改写该路径。

修改锁定逻辑后，可以用两次运行的 `latest-summary.json` 做回归门槛检查。默认要求匹配次数不少于基线的 98%，切换次数、注册后新 ID 和注册目标离开次数不能增加，平均/最大位移最多增加 10%：

```powershell
.\compare_target_lock_runs.bat -Baseline runs\baseline\latest-summary.json -Candidate runs\candidate\latest-summary.json
```

检查返回非零退出码时不应提交该锁定算法改动。阈值可以通过 `-MinMatchRatio`、`-MaxSwitchIncrease`、`-MaxAverageDistanceRatio` 和 `-MaxMaximumDistanceRatio` 调整。

慢速播放便于点击和观察：

```powershell
.\run_video_replay.bat -VideoA "D:\videos\camera_a.mp4" -VideoB "D:\videos\camera_b.mp4" -VideoPlaybackRate 0.5
```

循环演示并要求检查跨摄像头接力：

```powershell
.\run_video_replay.bat -VideoA "D:\videos\camera_a.mp4" -VideoB "D:\videos\camera_b.mp4" -LoopVideos -AnalyzeRequireHandoff
```

也可以直接调用 Python，并使用 `--video-c` 增加第三路视频：

```powershell
python src\crosscam_mvp.py --video-a "D:\videos\camera_a.mp4" --video-b "D:\videos\camera_b.mp4" --video-c "D:\videos\camera_c.mp4"
```

### 6. YOLO 检测入口

当前 YOLO 入口已经接入，`-PipeMode` 会优先加载本轮增量模型 `runs_yolo\pipe_yolov8n_hybrid_0710_v2\weights\best.pt`，缺失时回退到原模型。增量模型已经覆盖堆叠、拿取、局部可见、角度/距离变化和负样本，但真实铁管现场仍需要单独采集和训练；通用 COCO 预训练模型通常不能直接识别本项目的铅笔/铁管目标。

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
python src\crosscam_mvp.py --cam-a 0 --cam-b 2 --backend dshow --detector yolo --yolo-model runs_yolo\pipe_yolov8n_hybrid_0710_v2\weights\best.pt --max-detections 30
```

一键脚本也支持：

```powershell
.\run_crosscam.bat -PipeMode
```

### 7. RF-DETR 可选检测入口

RF-DETR 现在作为可选检测后端接入，用来和 YOLO 做后续效果对比。默认仍建议优先使用 `-PipeMode` 的 YOLO 流程；RF-DETR 需要单独安装依赖。

`-ModelSize` / `--rfdetr-size` 支持 `nano`、`small`、`base`、`medium`、`large`、`xlarge`、`2xlarge`。当前默认用 `nano`，方便先验证流程。

安装 RF-DETR 依赖：

```powershell
python -m pip install -r requirements-rfdetr.txt
```

把当前 YOLO 数据集导出为 RF-DETR 需要的 COCO JSON 结构：

```powershell
.\export_rfdetr_dataset.bat -Clean -TestFromVal
```

输出目录默认为 `datasets\pipe_rfdetr\`，其中包含 `train`、`valid`、`test` 三个子目录和各自的 `_annotations.coco.json`。

训练前先检查 RF-DETR 数据集结构：

```powershell
.\train_rfdetr.bat -CheckOnly
```

确认数据没问题后再启动训练：

```powershell
.\train_rfdetr.bat -ModelSize nano -Epochs 10 -BatchSize 4 -GradAccumSteps 4
```

评估入口也可以先检查一遍：

```powershell
.\evaluate_rfdetr.bat -CheckOnly
```

训练完成后，用 RF-DETR 在验证集上导出预测结果，并复用 YOLO 的分析指标做对比：

```powershell
.\evaluate_rfdetr.bat -Weights runs_rfdetr\pipe_rfdetr_nano\checkpoint_best_ema.pth
```

如果 YOLO 和 RF-DETR 都已经生成 `analysis.csv`，可以直接汇总对比：

```powershell
.\compare_detector_eval.bat
```

也可以一键依次运行 YOLO 评估、RF-DETR 评估和检测器对比：

```powershell
.\evaluate_detectors.bat
```

正式运行前可以先做轻量检查，不启动模型推理：

```powershell
.\evaluate_detectors.bat -CheckOnly
```

用 RF-DETR Nano 跑入口测试：

```powershell
python src\crosscam_mvp.py --demo --detector rfdetr --rfdetr-size nano --headless --frames 5
```

如果后续训练出 RF-DETR 权重，可以这样指定：

```powershell
python src\crosscam_mvp.py --camera-indexes 1,3 --backend dshow --detector rfdetr --rfdetr-size nano --rfdetr-weights output\checkpoint_best_ema.pth --rfdetr-num-classes 1 --max-detections 30
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
- 标准水平检测框在斜放或交叉堆叠时会高度重叠，难以稳定区分每一根细长目标；可继续评估 OBB 或实例分割。
- 当前系统只能做概率判断，不能保证完全一样的两个物体 100% 区分。

## 四阶段完善路线

详细路线见 [四阶段完善路线](docs/four_stage_plan.md)。当前项目按下面顺序推进：

### 阶段 1：先把每根管子检测稳定

目标是让系统能在堆叠铅笔/管子中尽量框出每一个可见目标。

已完成：

- 已接入 YOLO 检测入口，支持静止目标检测。
- `-PipeMode` 会优先加载本轮 `runs_yolo\pipe_yolov8n_hybrid_0710_v2\weights\best.pt`，缺失时回退到原模型。
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

- 当前先使用颜色、边缘、纹理、形状等轻量特征做相似度，并保留注册目标的少量可靠模板。
- 注册目标和可靠匹配目标会保存裁剪样本到 `runs/targets/`，索引文件是 `runs/targets/target_samples.csv`。
- 后续可以接入 ResNet、CLIP、DINOv2 这类图像特征。
- 对注册目标保存多张模板，覆盖正面、侧面、手拿取、光照变化。

### 阶段 4：利用 RealSense 深度辅助

目标是在堆叠、遮挡、管子外观相似时，增加空间信息，降低纯 RGB 判断的不稳定。

处理方法：

- 已增加 RealSense 深度探测入口，用于确认设备、深度帧、中心距离和有效深度比例。
- 使用深度估计管子的前后层次和距离变化。
- 对拿取动作记录目标从堆中离开的空间轨迹。
- 用 RGB 检测结果加深度位置共同判断目标是否连续。

深度探测命令：

```powershell
.\probe_realsense_depth.bat -ListOnly
.\probe_realsense_depth.bat -Frames 30
```

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
