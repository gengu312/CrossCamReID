param(
    [int]$DemoFrames = 260,
    [switch]$SkipYolo
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Find-Python {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    throw "Python was not found. Install Python and try again."
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "== $Name =="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name"
        exit $LASTEXITCODE
    }
    Write-Host "OK: $Name"
}

$PythonExe = Find-Python
$SmokeRoot = Join-Path $env:TEMP "crosscam_reid_smoke"
$ExportRoot = Join-Path $SmokeRoot "exported"
$DatasetRoot = Join-Path $SmokeRoot "dataset"
$RfDetrDatasetRoot = Join-Path $SmokeRoot "dataset_rfdetr"
$AutoLabelRoot = Join-Path $SmokeRoot "auto_label"
$RunLogDir = Join-Path $SmokeRoot "runs"

Remove-Item -Recurse -Force $SmokeRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $ExportRoot "images"), (Join-Path $ExportRoot "labels") | Out-Null

Invoke-Step "Python compile" {
    & $PythonExe -m py_compile `
        src\crosscam_mvp.py `
        src\camera_selector.py `
        src\capture_dataset.py `
        src\summarize_capture_manifest.py `
        src\prepare_yolo_dataset.py `
        src\validate_yolo_dataset.py `
        src\auto_label_pipes.py `
        src\collect_yolo_issues.py `
        src\collect_target_samples.py `
        src\export_yolo_to_coco.py `
        src\train_rfdetr.py `
        src\evaluate_rfdetr.py `
        src\compare_detector_eval.py `
        src\analyze_run_log.py `
        src\analyze_yolo_eval.py `
        src\realsense_depth_probe.py
}

Invoke-Step "PowerShell syntax" {
    $ParseFailures = @()
    Get-ChildItem -LiteralPath (Join-Path $RepoRoot "scripts") -Filter *.ps1 | Sort-Object Name | ForEach-Object {
        $Tokens = $null
        $Errors = $null
        [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$Tokens, [ref]$Errors) | Out-Null
        if ($Errors -and $Errors.Count -gt 0) {
            foreach ($ParseError in $Errors) {
                $ParseFailures += "$($_.Name): $($ParseError.Message)"
            }
        }
    }
    if ($ParseFailures.Count -gt 0) {
        Write-Host "PowerShell parse failures:"
        $ParseFailures | ForEach-Object { Write-Host "  $_" }
        exit 2
    }
    Write-Host "All PowerShell scripts parse."
}

Invoke-Step "Batch wrapper targets" {
    $MissingTargets = @()
    $Pattern = 'scripts\\[^"%\s]+\.ps1'
    Get-ChildItem -LiteralPath $RepoRoot -Filter *.bat | ForEach-Object {
        $BatPath = $_.FullName
        $Content = Get-Content -LiteralPath $BatPath -Raw
        [regex]::Matches($Content, $Pattern) | ForEach-Object {
            $Target = Join-Path $RepoRoot $_.Value
            if (-not (Test-Path -LiteralPath $Target)) {
                $MissingTargets += "$(Split-Path -Leaf $BatPath) -> $($_.Value)"
            }
        }
    }
    if ($MissingTargets.Count -gt 0) {
        Write-Host "Batch wrappers reference missing scripts:"
        $MissingTargets | ForEach-Object { Write-Host "  $_" }
        exit 2
    }
    Write-Host "All batch wrapper script targets exist."
}

Invoke-Step "Target-lock batch wrapper" {
    $WrapperPath = Join-Path $RepoRoot "analyze_target_lock.bat"
    if (-not (Test-Path -LiteralPath $WrapperPath)) {
        Write-Host "Missing target-lock wrapper: $WrapperPath"
        exit 2
    }
    $WrapperContent = Get-Content -LiteralPath $WrapperPath -Raw
    if ($WrapperContent -notmatch "scripts\\analyze_run\.ps1" -or $WrapperContent -notmatch "-TargetLockGate") {
        Write-Host "Expected analyze_target_lock.bat to call analyze_run.ps1 with -TargetLockGate."
        exit 2
    }
    Write-Host "Target-lock batch wrapper ok."
}

Invoke-Step "Target-lock check launcher" {
    $WrapperPath = Join-Path $RepoRoot "run_target_lock_check.bat"
    if (-not (Test-Path -LiteralPath $WrapperPath)) {
        Write-Host "Missing target-lock check launcher: $WrapperPath"
        exit 2
    }
    $WrapperContent = Get-Content -LiteralPath $WrapperPath -Raw
    foreach ($RequiredText in @(
        "scripts\run_crosscam.ps1",
        "-SelectCameras",
        "-PipeMode",
        "-AnalyzeAfterRun",
        "-AnalyzeTargetLockGate",
        "-AnalyzeRequireHandoff"
    )) {
        if ($WrapperContent -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected run_target_lock_check.bat to include: $RequiredText"
            exit 2
        }
    }
    Write-Host "Target-lock check launcher ok."
}

Invoke-Step "RF-DETR pipe launcher" {
    $WrapperPath = Join-Path $RepoRoot "run_pipe_rfdetr.bat"
    if (-not (Test-Path -LiteralPath $WrapperPath)) {
        Write-Host "Missing RF-DETR pipe launcher: $WrapperPath"
        exit 2
    }
    $WrapperContent = Get-Content -LiteralPath $WrapperPath -Raw
    foreach ($RequiredText in @(
        "scripts\run_crosscam.ps1",
        "-SelectCameras",
        "-PipeMode",
        "-Detector rfdetr",
        "-RfDetrNumClasses 1",
        "-RfDetrClasses 0",
        "-AnalyzeAfterRun",
        "-AnalyzeTargetLockGate",
        "-AnalyzeRequireHandoff"
    )) {
        if ($WrapperContent -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected run_pipe_rfdetr.bat to include: $RequiredText"
            exit 2
        }
    }
    Write-Host "RF-DETR pipe launcher ok."
}

Invoke-Step "Capture dataset defaults" {
    @'
import sys
import tempfile
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "src"))
import capture_dataset
from capture_dataset import (
    apply_capture_flips,
    append_manifest_rows,
    capture_manifest_rows,
    capture_order,
    existing_images_message,
    image_count,
    image_counts,
    new_images_message,
    parse_args,
    render_canvas,
    resolve_camera_pair,
)

sys.argv = ["capture_dataset.py"]
args = parse_args()
if args.cam_a is not None or args.cam_b is not None:
    raise SystemExit(f"Expected capture defaults to use auto cameras, got {args.cam_a}/{args.cam_b}.")
if args.camera_scan_order[:2] != [1, 3]:
    raise SystemExit(f"Expected capture auto scan to prefer 1/3, got {args.camera_scan_order}.")
sys.argv = ["capture_dataset.py", "--flip-a", "--flip-b", "--flip-both"]
flip_args = parse_args()
if not (flip_args.flip_a and flip_args.flip_b and flip_args.flip_both):
    raise SystemExit("Expected capture flip flags to be parsed.")
sys.argv = ["capture_dataset.py", "--cam-a", "3", "--cam-b", "1", "--view-order", "BA"]
ordered_args = parse_args()
if capture_order(resolve_camera_pair(ordered_args), ordered_args.view_order) != [1, 3]:
    raise SystemExit("Expected capture view-order BA to swap camera A/B.")
sample_a = np.array([[[1, 0, 0], [2, 0, 0]]], dtype=np.uint8)
sample_b = np.array([[[3, 0, 0], [4, 0, 0]]], dtype=np.uint8)
flipped_a, kept_b = apply_capture_flips(sample_a, sample_b, True, False, False)
if flipped_a[0, 0, 0] != 2 or kept_b[0, 0, 0] != 3:
    raise SystemExit("Expected capture flip-a to affect only camera A.")
kept_a, flipped_b = apply_capture_flips(sample_a, sample_b, False, True, False)
if kept_a[0, 0, 0] != 1 or flipped_b[0, 0, 0] != 4:
    raise SystemExit("Expected capture flip-b to affect only camera B.")
both_a, both_b = apply_capture_flips(sample_a, sample_b, False, False, True)
if both_a[0, 0, 0] != 2 or both_b[0, 0, 0] != 4:
    raise SystemExit("Expected capture flip-both to affect both cameras.")

original_find_available = capture_dataset.find_available_camera_indexes
try:
    capture_dataset.find_available_camera_indexes = lambda *call_args, **kwargs: [1, 3]
    if resolve_camera_pair(args) != [1, 3]:
        raise SystemExit("Expected capture auto selection to resolve to 1/3.")

    sys.argv = ["capture_dataset.py", "--cam-a", "3", "--cam-b", "auto"]
    mixed_args = parse_args()
    calls = []
    def fake_find_available(*call_args, **kwargs):
        calls.append({"call_args": call_args, "kwargs": kwargs})
        return [1]
    capture_dataset.find_available_camera_indexes = fake_find_available
    if resolve_camera_pair(mixed_args) != [3, 1]:
        raise SystemExit("Expected capture mixed fixed/auto selection to keep fixed camera first.")
    if calls[-1]["kwargs"].get("needed") != 2:
        raise SystemExit("Expected capture mixed fixed/auto selection to probe past duplicate fixed indexes.")
    sys.argv = ["capture_dataset.py", "--cam-a", "auto", "--cam-b", "3"]
    right_fixed_args = parse_args()
    capture_dataset.find_available_camera_indexes = lambda *call_args, **kwargs: [1, 3]
    if resolve_camera_pair(right_fixed_args) != [1, 3]:
        raise SystemExit("Expected capture auto/fixed selection to preserve camera B slot.")

    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["capture_dataset.py", "--print-only", "--output-root", tmp]
        print_args = parse_args()
        if not print_args.print_only:
            raise SystemExit("Expected capture --print-only to be parsed.")
        capture_dataset.find_available_camera_indexes = lambda *call_args, **kwargs: [4, 5]
        exit_code = capture_dataset.run(print_args)
        if exit_code != 0:
            raise SystemExit(f"Expected capture print-only run to exit 0, got {exit_code}.")
        if (Path(tmp) / "cam1_stack").exists() or (Path(tmp) / "cam2_stack").exists():
            raise SystemExit("Expected capture print-only not to create output directories.")
        if (Path(tmp) / "capture_manifest.csv").exists():
            raise SystemExit("Expected capture print-only not to create a manifest.")

        left_dir = Path(tmp) / "cam1_stack"
        right_dir = Path(tmp) / "cam2_stack"
        left_dir.mkdir()
        right_dir.mkdir()
        (left_dir / "a.jpg").write_bytes(b"fake")
        (left_dir / "ignored.txt").write_text("not an image", encoding="utf-8")
        (right_dir / "b.png").write_bytes(b"fake")
        (right_dir / "c.jpeg").write_bytes(b"fake")
        if image_count(left_dir) != 1 or image_count(right_dir) != 2:
            raise SystemExit("Expected capture image counting to include common image suffixes only.")
        if image_counts(left_dir, right_dir) != (1, 2):
            raise SystemExit("Expected capture image_counts to return both directory counts.")
        both_message = existing_images_message(left_dir, right_dir)
        if "1" not in both_message or "2" not in both_message:
            raise SystemExit("Expected capture existing image message to report both camera directories.")
        same_message = existing_images_message(left_dir, left_dir)
        if str(left_dir) not in same_message or not same_message.endswith("=1"):
            raise SystemExit("Expected capture existing image message to collapse identical directories.")
        delta_message = new_images_message(left_dir, right_dir, (0, 1))
        if "1" not in delta_message:
            raise SystemExit("Expected capture new image message to include positive deltas.")
        same_delta_message = new_images_message(left_dir, left_dir, (0, 0))
        if str(left_dir) not in same_delta_message or not same_delta_message.endswith("=1"):
            raise SystemExit("Expected capture new image message to collapse identical directories.")
        manifest = Path(tmp) / "capture_manifest.csv"
        rows = capture_manifest_rows(
            "cap-001",
            "stack",
            "manual_pair",
            [("cam1", 1, left_dir / "a.jpg"), ("cam2", 3, right_dir / "b.png"), ("cam2", 3, None)],
            "BA",
            True,
            False,
            True,
        )
        append_manifest_rows(manifest, rows)
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        if len(manifest_rows) != 2:
            raise SystemExit(f"Expected two manifest rows, got {len(manifest_rows)}.")
        first_row = manifest_rows[0]
        if first_row["capture_id"] != "cap-001" or manifest_rows[1]["capture_id"] != "cap-001":
            raise SystemExit(f"Expected paired manifest rows to share capture_id: {manifest_rows}")
        if first_row["scenario"] != "stack" or first_row["action"] != "manual_pair":
            raise SystemExit(f"Unexpected manifest scenario/action: {first_row}")
        if first_row["camera_label"] != "cam1" or first_row["camera_index"] != "1":
            raise SystemExit(f"Unexpected manifest camera fields: {first_row}")
        if first_row["view_order"] != "BA" or first_row["flip_a"] != "True" or first_row["flip_both"] != "True":
            raise SystemExit(f"Unexpected manifest settings fields: {first_row}")

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        canvas = render_canvas(frame, frame, "stack", 3, both_message, "saved")
        if canvas.shape != (496, 1280, 3):
            raise SystemExit(f"Unexpected capture canvas shape: {canvas.shape}")

    sys.argv = ["capture_dataset.py", "--cam-a", "1", "--cam-b", "1"]
    duplicate_args = parse_args()
    try:
        resolve_camera_pair(duplicate_args)
    except RuntimeError:
        pass
    else:
        raise SystemExit("Expected capture to reject duplicate fixed camera indexes.")
finally:
    capture_dataset.find_available_camera_indexes = original_find_available

print("capture dataset defaults ok")
'@ | & $PythonExe -
}

Invoke-Step "Capture command generation" {
    $PrintRoot = Join-Path $SmokeRoot "capture_print"
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\capture_dataset.ps1 `
        -CamA 1 `
        -CamB 3 `
        -OutputRoot $PrintRoot `
        -PrintOnly
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if ((Test-Path -LiteralPath (Join-Path $PrintRoot "cam1_stack")) -or (Test-Path -LiteralPath (Join-Path $PrintRoot "cam2_stack"))) {
        Write-Host "Expected capture -PrintOnly not to create output directories."
        exit 2
    }
    $SwapOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\capture_dataset.ps1 `
        -CamA 3 `
        -CamB 1 `
        -ViewOrder BA `
        -OutputRoot $PrintRoot `
        -PrintOnly
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $SwapText = $SwapOutput -join "`n"
    if ($SwapText -notmatch "--view-order BA") {
        Write-Host "Expected capture wrapper to pass --view-order BA."
        exit 2
    }
    if ($SwapText -notmatch "A=1 B=3") {
        Write-Host "Expected capture -ViewOrder BA to swap printed camera order."
        exit 2
    }
}

Invoke-Step "Summarize capture manifest" {
    $ManifestRoot = Join-Path $SmokeRoot "manifest_summary"
    $ImageDir = Join-Path $ManifestRoot "cam1_stack"
    New-Item -ItemType Directory -Force -Path $ImageDir | Out-Null
    $ImagePath = Join-Path $ImageDir "sample.jpg"
    [System.IO.File]::WriteAllBytes($ImagePath, [byte[]](1, 2, 3))
    $ManifestPath = Join-Path $ManifestRoot "capture_manifest.csv"
    @(
        "capture_id,saved_at,scenario,action,camera_label,camera_index,path,view_order,flip_a,flip_b,flip_both",
        "cap-001,2026-07-02 19:00:00,stack,manual_left,cam1,1,$ImagePath,AB,False,False,False"
    ) | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\summarize_capture_manifest.ps1 `
        -Manifest $ManifestPath `
        -Strict
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $PlanOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\summarize_capture_manifest.ps1 `
        -Manifest $ManifestPath `
        -FirstBatchPlan
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $PlanText = $PlanOutput -join "`n"
    if ($PlanText -notmatch "single >= 20" -or $PlanText -notmatch "stack") {
        Write-Host "Expected summarize_capture_manifest to print first-batch capture shortages."
        exit 2
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\summarize_capture_manifest.ps1 `
        -Manifest $ManifestPath `
        -RequireFirstBatch
    $BatchCode = $LASTEXITCODE
    if ($BatchCode -eq 0) {
        Write-Host "Expected summarize_capture_manifest -RequireFirstBatch to fail on insufficient sample counts."
        exit 2
    }
    if ($BatchCode -ne 2) {
        Write-Host "Unexpected summarize_capture_manifest first-batch exit code: $BatchCode"
        exit $BatchCode
    }
    Write-Host "First-batch capture shortages rejected."
    & $PythonExe -c "pass"
    Remove-Item -LiteralPath $ImagePath
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\summarize_capture_manifest.ps1 `
        -Manifest $ManifestPath `
        -Strict
    $StrictCode = $LASTEXITCODE
    if ($StrictCode -eq 0) {
        Write-Host "Expected summarize_capture_manifest -Strict to fail on missing image path."
        exit 2
    }
    if ($StrictCode -ne 2) {
        Write-Host "Unexpected summarize_capture_manifest strict exit code: $StrictCode"
        exit $StrictCode
    }
    Write-Host "Missing capture manifest image rejected."
    & $PythonExe -c "pass"
}

Invoke-Step "Target sticky selection" {
    @'
import src.crosscam_mvp as mvp
import numpy as np
from types import SimpleNamespace

from src.crosscam_mvp import (
    CrossCameraTracker,
    Detection,
    TargetProfile,
    Track,
    action_from_key,
    apply_target_profile,
    build_initial_view_order,
    draw_event_panel,
    open_sources,
    roi_for_camera,
    resolve_camera_indexes,
    select_stable_target_match,
    synthetic_camera_count,
    synthetic_sources,
    target_status_for_ui,
)

profile = TargetProfile()
profile.last_camera_id = 0
profile.last_bbox = (100, 100, 40, 20)

near = Detection(
    camera_id=0,
    bbox=(110, 102, 40, 20),
    center=(130, 112),
    area=800,
    score=0.70,
    feature=np.ones(4),
    crop=np.zeros((4, 4, 3), dtype=np.uint8),
    target_similarity=0.87,
    is_target_match=True,
)
far = Detection(
    camera_id=0,
    bbox=(360, 100, 40, 20),
    center=(380, 110),
    area=800,
    score=0.90,
    feature=np.ones(4),
    crop=np.zeros((4, 4, 3), dtype=np.uint8),
    target_similarity=0.92,
    is_target_match=True,
)

chosen = select_stable_target_match([far, near], profile, stick_distance=80, switch_margin=0.08)
if chosen is not near or chosen.target_choice != "sticky":
    raise SystemExit("Expected nearby target to be kept when similarity gap is within margin.")

chosen = select_stable_target_match([far, near], profile, stick_distance=80, switch_margin=0.01)
if chosen is not far or chosen.target_choice != "switch":
    raise SystemExit("Expected far target to win when it is clearly more similar.")

adjacent = Detection(
    camera_id=0,
    bbox=(150, 104, 40, 20),
    center=(170, 114),
    area=800,
    score=0.95,
    feature=np.ones(4),
    crop=np.zeros((4, 4, 3), dtype=np.uint8),
    target_similarity=0.91,
    is_target_match=True,
)

chosen = select_stable_target_match([adjacent, near], profile, stick_distance=80, switch_margin=0.08)
if chosen is not near or chosen.target_choice != "sticky":
    raise SystemExit("Expected closest nearby target to be kept when adjacent candidates are similarly matched.")

chosen = select_stable_target_match([adjacent, near], profile, stick_distance=80, switch_margin=0.01)
if chosen is not adjacent or chosen.target_choice != "switch":
    raise SystemExit("Expected adjacent target to win only when similarity gain is large enough.")

claim_tracker = CrossCameraTracker(camera_count=2)
claim_tracker.activate_registered_target(1)
claim_profile = TargetProfile()
registered = Detection(
    camera_id=0,
    bbox=(100, 100, 40, 20),
    center=(120, 110),
    area=800,
    score=0.80,
    feature=np.ones(4),
    crop=np.zeros((4, 4, 3), dtype=np.uint8),
)
claim_profile.register_from_detection(registered)
claim_tracker.active[0].append(
    Track(
        camera_id=0,
        local_id=1,
        global_id=1,
        bbox=registered.bbox,
        center=registered.center,
        feature=registered.feature,
        last_seen=10.0,
    )
)
remote_candidate = Detection(
    camera_id=1,
    bbox=(105, 100, 40, 20),
    center=(125, 110),
    area=800,
    score=0.95,
    feature=np.ones(4),
    crop=np.zeros((4, 4, 3), dtype=np.uint8),
)
filtered = apply_target_profile(
    [remote_candidate],
    claim_profile,
    threshold=0.5,
    update_alpha=0.0,
    keep_all=True,
    can_claim_target=lambda detection: claim_tracker._can_claim_registered_target(detection, 10.0),
)
if filtered != [remote_candidate]:
    raise SystemExit("Expected blocked target candidate to remain visible when keep_all is enabled.")
if remote_candidate.is_target_match:
    raise SystemExit("Expected another camera not to claim G001 while the registered target is still active.")
if remote_candidate.target_choice != "blocked":
    raise SystemExit(f"Expected blocked target candidate to be marked for diagnostics, got {remote_candidate.target_choice}.")
if claim_profile.last_camera_id != 0:
    raise SystemExit("Expected blocked candidate not to update target profile camera.")
remote_tracks = claim_tracker.update(1, filtered, 10.0)
if not remote_tracks or remote_tracks[0].global_id == 1:
    raise SystemExit("Expected blocked target candidate to remain a non-G001 candidate.")
if remote_tracks[0].last_target_choice != "blocked":
    raise SystemExit("Expected blocked target diagnostic to be preserved on the created track.")

if build_initial_view_order(1, (1, 0)) != [0]:
    raise SystemExit("Expected one-camera view order to fall back to A.")
if build_initial_view_order(2, (1, 0)) != [1, 0]:
    raise SystemExit("Expected BA to swap two-camera view order.")
if build_initial_view_order(3, (1, 0)) != [1, 0, 2]:
    raise SystemExit("Expected BA to swap the first two slots and keep C for three cameras.")
roi_args = SimpleNamespace(
    roi_a=(1, 2, 3, 4),
    roi_b=(5, 6, 7, 8),
    roi_c=(9, 10, 11, 12),
)
if roi_for_camera(roi_args, 0) != roi_args.roi_a:
    raise SystemExit("Expected camera A to use roi_a.")
if roi_for_camera(roi_args, 1) != roi_args.roi_b:
    raise SystemExit("Expected camera B to use roi_b.")
if roi_for_camera(roi_args, 2) != roi_args.roi_c:
    raise SystemExit("Expected camera C to use roi_c.")
mixed_camera_args = SimpleNamespace(
    camera_indexes=None,
    cam_a=3,
    cam_b=None,
    probe_max=10,
    backend="any",
    camera_scan_order=[1, 3, 2, 0],
)
original_find_available = mvp.find_available_camera_indexes
mixed_calls = []
try:
    def fake_find_available(*call_args, **kwargs):
        mixed_calls.append({"call_args": call_args, "kwargs": kwargs})
        return [1]
    mvp.find_available_camera_indexes = fake_find_available
    if resolve_camera_indexes(mixed_camera_args) != [3, 1]:
        raise SystemExit("Expected crosscam mixed fixed/auto selection to keep fixed camera first.")
    if mixed_calls[-1]["kwargs"].get("needed") != 2:
        raise SystemExit("Expected crosscam mixed fixed/auto selection to probe past duplicate fixed indexes.")
    right_fixed_args = SimpleNamespace(
        camera_indexes=None,
        cam_a=None,
        cam_b=3,
        probe_max=10,
        backend="any",
        camera_scan_order=[1, 3, 2, 0],
    )
    mvp.find_available_camera_indexes = lambda *call_args, **kwargs: [1, 3]
    if resolve_camera_indexes(right_fixed_args) != [1, 3]:
        raise SystemExit("Expected crosscam auto/fixed selection to preserve camera B slot.")
finally:
    mvp.find_available_camera_indexes = original_find_available
default_demo_args = SimpleNamespace(camera_indexes=None)
if synthetic_camera_count(default_demo_args) != 2:
    raise SystemExit("Expected synthetic demo to default to two cameras.")
three_demo_args = SimpleNamespace(demo=True, camera_indexes=[1, 2, 3])
if synthetic_camera_count(three_demo_args) != 3:
    raise SystemExit("Expected explicit camera indexes to request three synthetic cameras.")
sources = open_sources(three_demo_args)
if len(sources) != 3:
    raise SystemExit(f"Expected three synthetic sources, got {len(sources)}.")
for source in sources:
    source.release()
if len(synthetic_sources(1)) != 1:
    raise SystemExit("Expected one-camera synthetic demo to be supported.")

fallback_args = SimpleNamespace(
    demo=False,
    fallback_demo=True,
    camera_indexes=None,
    cam_a=None,
    cam_b=None,
    probe_max=1,
    backend="any",
    camera_scan_order=[],
)
original_find_available = mvp.find_available_camera_indexes
try:
    mvp.find_available_camera_indexes = lambda *args, **kwargs: []
    fallback_sources = mvp.open_sources(fallback_args)
    if len(fallback_sources) != 2:
        raise SystemExit(f"Expected fallback demo to create two synthetic sources, got {len(fallback_sources)}.")
    for source in fallback_sources:
        source.release()
finally:
    mvp.find_available_camera_indexes = original_find_available

tracker = CrossCameraTracker(camera_count=2)
status, _color = target_status_for_ui(TargetProfile(), tracker, [[], []])
if "\u672a\u9501\u5b9a" not in status:
    raise SystemExit("Expected UI status to explain that no target is locked.")

tracker.activate_registered_target(1)
active_profile = TargetProfile()
active_profile.register_from_detection(near)
matched_track = Track(
    camera_id=0,
    local_id=1,
    global_id=1,
    bbox=(100, 100, 40, 20),
    center=(120, 110),
    feature=np.ones(4),
    last_seen=0.0,
    last_target_similarity=0.91,
)
status, _color = target_status_for_ui(active_profile, tracker, [[matched_track], []])
if "G001" not in status or "\u6444\u50cf\u59341" not in status:
    raise SystemExit("Expected UI status to show the locked target camera.")

panel_frame = np.zeros((360, 1920, 3), dtype=np.uint8)
_panel, buttons, _scroll, _visible = draw_event_panel(
    panel_frame,
    ["event"],
    view_order=[0, 1, 2],
    flip_horizontal=[False, True, False],
)
actions = {button.action for button in buttons}
if "register_third" not in actions:
    raise SystemExit("Expected three-camera UI to expose register_third action.")
if "manual_third" not in actions:
    raise SystemExit("Expected three-camera UI to expose manual_third action.")
if "flip_third" not in actions:
    raise SystemExit("Expected three-camera UI to expose flip_third action.")

expected_key_actions = {
    "1": "register_left",
    "2": "register_right",
    "3": "register_third",
    "4": "manual_left",
    "5": "manual_right",
    "6": "manual_third",
    "7": "flip_third",
    "q": "quit",
}
for key, expected_action in expected_key_actions.items():
    actual_action = action_from_key(ord(key))
    if actual_action != expected_action:
        raise SystemExit(f"Expected key {key} -> {expected_action}, got {actual_action}.")

print("target sticky selection ok")
'@ | & $PythonExe -
}

Invoke-Step "Camera selector launch command" {
    @'
from pathlib import Path
import sys

from src.camera_selector import (
    CAPTURE_SCENARIOS,
    CameraProbe,
    PIPE_DETECTORS,
    RFDETR_SIZES,
    build_capture_command,
    build_crosscam_command,
    capture_order,
    detector_for_selector,
    default_selected_indexes,
    extra_arg_flag,
    extra_arg_value,
    mousewheel_units,
    ordered_selected_indexes,
    parse_args,
    parse_preferred_indexes,
    selection_error,
    selector_extra_args,
    set_extra_arg_flag,
    set_extra_arg_pair,
    target_review_flags,
)
from src.crosscam_mvp import parse_args as parse_crosscam_args

sys.argv = ["camera_selector.py", "--extra-arg=-Detector", "--extra-arg=rfdetr"]
args = parse_args()
if args.extra_arg != ["-Detector", "rfdetr"]:
    raise SystemExit(f"Unexpected selector extra args: {args.extra_arg}")
if args.capture_script != "scripts/capture_dataset.ps1":
    raise SystemExit(f"Unexpected selector capture script default: {args.capture_script}")
if args.capture_scenario != "stack":
    raise SystemExit(f"Unexpected selector capture scenario default: {args.capture_scenario}")
if args.capture_output_root != "dataset_raw":
    raise SystemExit(f"Unexpected selector capture output root default: {args.capture_output_root}")
if set(CAPTURE_SCENARIOS) != {"stack", "single", "hand_move", "negative"}:
    raise SystemExit(f"Unexpected capture scenarios: {CAPTURE_SCENARIOS}")
if set(RFDETR_SIZES) != {"nano", "small", "base", "medium", "large", "xlarge", "2xlarge"}:
    raise SystemExit(f"Unexpected RF-DETR model sizes: {RFDETR_SIZES}")
if set(PIPE_DETECTORS) != {"yolo", "rfdetr"}:
    raise SystemExit(f"Unexpected PipeMode detector set: {PIPE_DETECTORS}")
if extra_arg_value(args.extra_arg, "-Detector", "motion") != "rfdetr":
    raise SystemExit("Expected selector to read detector from extra args.")
if parse_preferred_indexes("1,3,bad,3,-1") != [1, 3]:
    raise SystemExit("Expected selector preferred indexes to ignore duplicates and invalid entries.")
selected_defaults = default_selected_indexes(
    [
        CameraProbe(index=0, width=640, height=360, frame_bgr=None),
        CameraProbe(index=1, width=640, height=360, frame_bgr=None),
        CameraProbe(index=2, width=640, height=360, frame_bgr=None),
        CameraProbe(index=3, width=640, height=360, frame_bgr=None),
    ],
    [1, 3],
)
if selected_defaults != {1, 3}:
    raise SystemExit(f"Expected selector to prefer external indexes 1 and 3, got {selected_defaults}.")
ordered_defaults = ordered_selected_indexes(
    [1, 3],
    [1, 3],
    [
        CameraProbe(index=0, width=640, height=360, frame_bgr=None),
        CameraProbe(index=1, width=640, height=360, frame_bgr=None),
        CameraProbe(index=2, width=640, height=360, frame_bgr=None),
        CameraProbe(index=3, width=640, height=360, frame_bgr=None),
    ],
)
if ordered_defaults != [1, 3]:
    raise SystemExit(f"Expected selector to keep preferred selected order 1,3, got {ordered_defaults}.")
ordered_custom = ordered_selected_indexes(
    [1, 3],
    [3, 1],
    [
        CameraProbe(index=0, width=640, height=360, frame_bgr=None),
        CameraProbe(index=1, width=640, height=360, frame_bgr=None),
        CameraProbe(index=2, width=640, height=360, frame_bgr=None),
        CameraProbe(index=3, width=640, height=360, frame_bgr=None),
    ],
)
if ordered_custom != [3, 1]:
    raise SystemExit(f"Expected selector to honor custom preferred order 3,1, got {ordered_custom}.")
if capture_order([3, 1], "AB") != [3, 1]:
    raise SystemExit("Expected capture AB order to keep selected camera order.")
if capture_order([3, 1], "BA") != [1, 3]:
    raise SystemExit("Expected capture BA order to swap the first two selected cameras.")
sys.argv = ["crosscam_mvp.py"]
crosscam_args = parse_crosscam_args()
if crosscam_args.camera_scan_order[:2] != [1, 3]:
    raise SystemExit(f"Expected crosscam auto scan to prefer 1,3, got {crosscam_args.camera_scan_order}.")
extra_args = set_extra_arg_pair(
    ["-Detector", "motion", "-RfDetrSize", "small"],
    "-Detector",
    "yolo",
)
if extra_args != ["-RfDetrSize", "small", "-Detector", "yolo"]:
    raise SystemExit(f"Unexpected selector detector replacement: {extra_args}")
if detector_for_selector(["-Detector", "motion"], pipe_mode=True) != "yolo":
    raise SystemExit("Expected PipeMode selector default detector to be yolo.")
if detector_for_selector(["-Detector", "rfdetr"], pipe_mode=True) != "rfdetr":
    raise SystemExit("Expected PipeMode selector to preserve explicit RF-DETR detector.")
if detector_for_selector(["-Detector", "rfdetr"], pipe_mode=False) != "rfdetr":
    raise SystemExit("Expected RF-DETR selector default detector to be preserved.")
if detector_for_selector(["-Detector", "bad"], pipe_mode=False) != "motion":
    raise SystemExit("Expected invalid selector detector to fall back to motion.")
if mousewheel_units(120, None) != -1:
    raise SystemExit("Expected positive wheel delta to scroll up.")
if mousewheel_units(-120, None) != 1:
    raise SystemExit("Expected negative wheel delta to scroll down.")
if mousewheel_units(1, None) != -1:
    raise SystemExit("Expected small positive wheel delta to still scroll up.")
if mousewheel_units(-1, None) != 1:
    raise SystemExit("Expected small negative wheel delta to still scroll down.")
if mousewheel_units(0, 4) != -1 or mousewheel_units(0, 5) != 1:
    raise SystemExit("Expected Linux wheel button events to map to scroll units.")
if mousewheel_units(0, None) != 0:
    raise SystemExit("Expected zero wheel delta to do nothing.")
if selection_error(0, require_handoff=False) is None:
    raise SystemExit("Expected selector to reject zero cameras.")
if selection_error(0, require_handoff=False, fallback_demo=True) is not None:
    raise SystemExit("Expected selector to allow zero cameras when fallback demo is enabled.")
if selection_error(0, require_handoff=True, fallback_demo=True) is None:
    raise SystemExit("Expected selector to reject zero cameras when handoff check is required.")
if selection_error(4, require_handoff=False) is None:
    raise SystemExit("Expected selector to reject more than three cameras.")
if selection_error(1, require_handoff=False) is not None:
    raise SystemExit("Expected selector to allow one camera without handoff check.")
if selection_error(1, require_handoff=True) is None:
    raise SystemExit("Expected selector to reject one camera when handoff check is required.")
if selection_error(2, require_handoff=True) is not None:
    raise SystemExit("Expected selector to allow two cameras with handoff check.")
handoff_extra_args = selector_extra_args(
    ["-Detector", "motion"],
    detector="yolo",
    fallback_demo=False,
    analyze_after_run=False,
    require_handoff=True,
    target_lock_gate=True,
    collect_target_samples=True,
    target_sample_preview=True,
)
if "-AnalyzeAfterRun" not in handoff_extra_args or "-AnalyzeRequireHandoff" not in handoff_extra_args:
    raise SystemExit(f"Expected handoff check to imply analysis: {handoff_extra_args}")
if "-AnalyzeTargetLockGate" not in handoff_extra_args:
    raise SystemExit(f"Expected target-lock gate flag to be preserved: {handoff_extra_args}")
if "-CollectTargetSamplesAfterRun" not in handoff_extra_args or "-TargetSamplePreview" not in handoff_extra_args:
    raise SystemExit(f"Expected target sample review flags to be preserved: {handoff_extra_args}")
if extra_arg_value(handoff_extra_args, "-Detector", "motion") != "yolo":
    raise SystemExit(f"Expected selector extra args to update detector: {handoff_extra_args}")
if target_review_flags(True, False, False) != (True, True):
    raise SystemExit("Expected target-lock gate to enable target sample review flags.")
auto_review_extra_args = selector_extra_args(
    ["-Detector", "motion"],
    detector="motion",
    fallback_demo=False,
    analyze_after_run=False,
    require_handoff=False,
    target_lock_gate=True,
    collect_target_samples=False,
    target_sample_preview=False,
)
if "-CollectTargetSamplesAfterRun" not in auto_review_extra_args or "-TargetSamplePreview" not in auto_review_extra_args:
    raise SystemExit(f"Expected target-lock gate to auto-enable target sample review flags: {auto_review_extra_args}")
rfdetr_extra_args = selector_extra_args(
    ["-Detector", "motion", "-RfDetrWeights", "old.pth"],
    detector="rfdetr",
    fallback_demo=False,
    analyze_after_run=False,
    require_handoff=False,
    target_lock_gate=False,
    collect_target_samples=False,
    target_sample_preview=False,
    rfdetr_size="small",
    rfdetr_weights="runs_rfdetr/best.pth",
    rfdetr_num_classes="1",
    rfdetr_conf="0.42",
)
if extra_arg_value(rfdetr_extra_args, "-RfDetrSize", "") != "small":
    raise SystemExit(f"Expected RF-DETR size to be passed from selector: {rfdetr_extra_args}")
if extra_arg_value(rfdetr_extra_args, "-RfDetrWeights", "") != "runs_rfdetr/best.pth":
    raise SystemExit(f"Expected RF-DETR weights to be passed from selector: {rfdetr_extra_args}")
if extra_arg_value(rfdetr_extra_args, "-RfDetrNumClasses", "") != "1":
    raise SystemExit(f"Expected RF-DETR class count to be passed from selector: {rfdetr_extra_args}")
if extra_arg_value(rfdetr_extra_args, "-RfDetrConf", "") != "0.42":
    raise SystemExit(f"Expected RF-DETR confidence to be passed from selector: {rfdetr_extra_args}")
cleared_rfdetr_weights = selector_extra_args(
    rfdetr_extra_args,
    detector="rfdetr",
    fallback_demo=False,
    analyze_after_run=False,
    require_handoff=False,
    target_lock_gate=False,
    collect_target_samples=False,
    target_sample_preview=False,
    rfdetr_size="nano",
    rfdetr_weights="",
    rfdetr_num_classes="0",
    rfdetr_conf="0.35",
)
if "-RfDetrWeights" in cleared_rfdetr_weights:
    raise SystemExit(f"Expected empty RF-DETR weights to clear the old selector value: {cleared_rfdetr_weights}")
plain_extra_args = selector_extra_args(
    handoff_extra_args,
    detector="motion",
    fallback_demo=False,
    analyze_after_run=False,
    require_handoff=False,
    target_lock_gate=False,
    collect_target_samples=False,
    target_sample_preview=False,
)
if "-AnalyzeAfterRun" in plain_extra_args or "-AnalyzeRequireHandoff" in plain_extra_args:
    raise SystemExit(f"Expected disabled analysis flags to be removed: {plain_extra_args}")
if "-AnalyzeTargetLockGate" in plain_extra_args:
    raise SystemExit(f"Expected disabled target-lock gate flag to be removed: {plain_extra_args}")
if "-CollectTargetSamplesAfterRun" in plain_extra_args or "-TargetSamplePreview" in plain_extra_args:
    raise SystemExit(f"Expected disabled target sample review flags to be removed: {plain_extra_args}")
if not extra_arg_flag(["-Detector", "motion", "-FallbackDemo"], "-FallbackDemo"):
    raise SystemExit("Expected selector fallback flag to be detected.")
with_fallback = set_extra_arg_flag(["-Detector", "motion"], "-FallbackDemo", True)
if with_fallback != ["-Detector", "motion", "-FallbackDemo"]:
    raise SystemExit(f"Unexpected fallback flag insertion: {with_fallback}")
without_fallback = set_extra_arg_flag(with_fallback, "-FallbackDemo", False)
if without_fallback != ["-Detector", "motion"]:
    raise SystemExit(f"Unexpected fallback flag removal: {without_fallback}")
with_analyze = set_extra_arg_flag(["-Detector", "motion"], "-AnalyzeAfterRun", True)
if with_analyze != ["-Detector", "motion", "-AnalyzeAfterRun"]:
    raise SystemExit(f"Unexpected analyze flag insertion: {with_analyze}")
without_analyze = set_extra_arg_flag(with_analyze, "-AnalyzeAfterRun", False)
if without_analyze != ["-Detector", "motion"]:
    raise SystemExit(f"Unexpected analyze flag removal: {without_analyze}")
with_handoff_check = set_extra_arg_flag(["-Detector", "motion"], "-AnalyzeRequireHandoff", True)
if with_handoff_check != ["-Detector", "motion", "-AnalyzeRequireHandoff"]:
    raise SystemExit(f"Unexpected handoff check flag insertion: {with_handoff_check}")
without_handoff_check = set_extra_arg_flag(with_handoff_check, "-AnalyzeRequireHandoff", False)
if without_handoff_check != ["-Detector", "motion"]:
    raise SystemExit(f"Unexpected handoff check flag removal: {without_handoff_check}")

command = build_crosscam_command(
    Path("scripts/run_crosscam.ps1"),
    [3, 1],
    backend="msmf",
    pipe_mode=True,
    flip_both=True,
    show_trails=False,
    view_order="BA",
    extra_args=[
        "-Detector",
        "rfdetr",
        "-RfDetrSize",
        "small",
        "-RoiC",
        "10,20,300,180",
        "-MaxMissed",
        "18",
        "-LostTtl",
        "9.5",
        "-FallbackDemo",
        "-RequireMatch",
        "-AnalyzeAfterRun",
        "-AnalyzeRequireHandoff",
        "-AnalyzeTargetLockGate",
        "-CollectTargetSamplesAfterRun",
        "-TargetSamplePreview",
        "-LogDir",
        "runs_test",
    ],
)
text = " ".join(command)
if "-CameraIndexes" not in command or "3,1" not in command:
    raise SystemExit(f"CameraIndexes missing from selector command: {text}")
if "-Backend" not in command or "msmf" not in command:
    raise SystemExit(f"Backend missing from selector command: {text}")
if "-ViewOrder" not in command or "BA" not in command:
    raise SystemExit(f"ViewOrder missing from selector command: {text}")
if "-PipeMode" not in command or "-FlipBoth" not in command:
    raise SystemExit(f"Expected launch options missing from selector command: {text}")
if "-Detector" not in command or "rfdetr" not in command or "-RfDetrSize" not in command or "small" not in command:
    raise SystemExit(f"Detector passthrough missing from selector command: {text}")
if "-RoiC" not in command or "10,20,300,180" not in command:
    raise SystemExit(f"Camera C ROI passthrough missing from selector command: {text}")
if "-MaxMissed" not in command or "18" not in command or "-LostTtl" not in command or "9.5" not in command:
    raise SystemExit(f"Tracking lifetime passthrough missing from selector command: {text}")
if "-FallbackDemo" not in command:
    raise SystemExit(f"Fallback demo passthrough missing from selector command: {text}")
if "-RequireMatch" not in command:
    raise SystemExit(f"RequireMatch passthrough missing from selector command: {text}")
if "-AnalyzeAfterRun" not in command:
    raise SystemExit(f"AnalyzeAfterRun passthrough missing from selector command: {text}")
if "-AnalyzeRequireHandoff" not in command:
    raise SystemExit(f"AnalyzeRequireHandoff passthrough missing from selector command: {text}")
if "-AnalyzeTargetLockGate" not in command:
    raise SystemExit(f"AnalyzeTargetLockGate passthrough missing from selector command: {text}")
if "-CollectTargetSamplesAfterRun" not in command:
    raise SystemExit(f"CollectTargetSamplesAfterRun passthrough missing from selector command: {text}")
if "-TargetSamplePreview" not in command:
    raise SystemExit(f"TargetSamplePreview passthrough missing from selector command: {text}")

capture_command = build_capture_command(
    Path("scripts/capture_dataset.ps1"),
    [3, 1],
    backend="dshow",
    scenario="hand_move",
    output_root="dataset_raw/session_001",
    flip_both=True,
)
capture_text = " ".join(capture_command)
if "-CamA" not in capture_command or capture_command[capture_command.index("-CamA") + 1] != "3":
    raise SystemExit(f"CamA missing from selector capture command: {capture_text}")
if "-CamB" not in capture_command or capture_command[capture_command.index("-CamB") + 1] != "1":
    raise SystemExit(f"CamB missing from selector capture command: {capture_text}")
if "-Backend" not in capture_command or "dshow" not in capture_command:
    raise SystemExit(f"Backend missing from selector capture command: {capture_text}")
if "-Scenario" not in capture_command or "hand_move" not in capture_command:
    raise SystemExit(f"Scenario missing from selector capture command: {capture_text}")
if "-OutputRoot" not in capture_command or "dataset_raw/session_001" not in capture_command:
    raise SystemExit(f"OutputRoot missing from selector capture command: {capture_text}")
if "-FlipBoth" not in capture_command:
    raise SystemExit(f"FlipBoth missing from selector capture command: {capture_text}")
if "scripts/capture_dataset.ps1" not in capture_text.replace("\\", "/"):
    raise SystemExit(f"Capture script missing from selector capture command: {capture_text}")
swapped_capture_command = build_capture_command(
    Path("scripts/capture_dataset.ps1"),
    [3, 1],
    backend="dshow",
    scenario="stack",
    output_root="dataset_raw",
    view_order="BA",
)
if swapped_capture_command[swapped_capture_command.index("-CamA") + 1] != "1":
    raise SystemExit(f"Expected swapped capture CamA to be 1: {' '.join(swapped_capture_command)}")
if swapped_capture_command[swapped_capture_command.index("-CamB") + 1] != "3":
    raise SystemExit(f"Expected swapped capture CamB to be 3: {' '.join(swapped_capture_command)}")

fallback_command = build_crosscam_command(
    Path("scripts/run_crosscam.ps1"),
    [],
    backend="dshow",
    pipe_mode=False,
    flip_both=False,
    show_trails=False,
    view_order="AB",
    extra_args=["-FallbackDemo"],
)
if "-CameraIndexes" in fallback_command:
    raise SystemExit(f"Empty camera selection should not pass CameraIndexes: {' '.join(fallback_command)}")
if "-FallbackDemo" not in fallback_command:
    raise SystemExit(f"Fallback demo flag missing from empty camera command: {' '.join(fallback_command)}")

print("camera selector launch command ok")
'@ | & $PythonExe -
}

Invoke-Step "RF-DETR adapter fake module" {
    @'
import sys
import types

import numpy as np


class FakePredictions:
    xyxy = np.array([10, 20, 60, 45], dtype=np.float32)
    confidence = np.array([0.88], dtype=np.float32)
    class_id = np.array([0], dtype=np.int32)


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, image, threshold=0.35):
        if image.shape[:2] != (70, 80):
            raise RuntimeError(f"Unexpected ROI shape: {image.shape}")
        return FakePredictions()


fake_rfdetr = types.SimpleNamespace(
    RFDETRNano=FakeModel,
    RFDETRSmall=FakeModel,
    RFDETRBase=FakeModel,
    RFDETRMedium=FakeModel,
    RFDETRLarge=FakeModel,
    RFDETRXLarge=FakeModel,
    RFDETR2XLarge=FakeModel,
)
sys.modules["rfdetr"] = fake_rfdetr

from src.crosscam_mvp import RfDetrDetector
from src.evaluate_rfdetr import prediction_arrays

frame = np.zeros((120, 140, 3), dtype=np.uint8)
frame[30:55, 15:65] = (20, 180, 20)
detector = RfDetrDetector(
    camera_id=0,
    model_size="2xlarge",
    num_classes=1,
    confidence=0.42,
    classes=[0],
    roi=(5, 10, 80, 70),
    max_detections=3,
)
detections = detector.detect(frame)
if len(detections) != 1:
    raise SystemExit(f"Expected one fake RF-DETR detection, got {len(detections)}")
if detections[0].bbox != (15, 30, 50, 25):
    raise SystemExit(f"Unexpected fake RF-DETR bbox: {detections[0].bbox}")
if abs(detections[0].score - 0.88) > 1e-6:
    raise SystemExit(f"Unexpected fake RF-DETR score: {detections[0].score}")

filtered = RfDetrDetector(camera_id=0, classes=[1], roi=(5, 10, 80, 70)).detect(frame)
if filtered:
    raise SystemExit("Expected class filter to remove fake RF-DETR detection.")

FakePredictions.class_id = np.array([1], dtype=np.int32)
mapped = RfDetrDetector(
    camera_id=0,
    classes=[0],
    class_id_mode="auto",
    category_id_offset=1,
    roi=(5, 10, 80, 70),
).detect(frame)
if len(mapped) != 1:
    raise SystemExit("Expected RF-DETR auto class-id mapping to keep COCO category 1 as YOLO class 0.")

unmapped = RfDetrDetector(
    camera_id=0,
    classes=[0],
    class_id_mode="zero",
    category_id_offset=1,
    roi=(5, 10, 80, 70),
).detect(frame)
if unmapped:
    raise SystemExit("Expected RF-DETR zero class-id mode to filter out raw class 1 when keeping class 0.")
FakePredictions.class_id = np.array([0], dtype=np.int32)

xyxy, conf, class_ids = prediction_arrays(FakePredictions())
if xyxy.shape != (1, 4) or conf.tolist() != [0.8799999952316284] or class_ids.tolist() != [0]:
    raise SystemExit("Expected RF-DETR prediction arrays to handle a single 1D box.")

flat_predictions = types.SimpleNamespace(
    xyxy=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
    confidence=np.array([0.4, 0.8], dtype=np.float32),
    class_id=np.array([0, 0], dtype=np.int32),
)
xyxy, conf, class_ids = prediction_arrays(flat_predictions)
if xyxy.shape != (2, 4) or conf.tolist() != [0.4000000059604645, 0.800000011920929]:
    raise SystemExit("Expected RF-DETR prediction arrays to reshape flat multi-box xyxy output.")

short_predictions = types.SimpleNamespace(
    xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
    confidence=np.array([0.5], dtype=np.float32),
    class_id=np.array([2], dtype=np.int32),
)
xyxy, conf, class_ids = prediction_arrays(short_predictions)
if conf.tolist() != [0.5, 1.0] or class_ids.tolist() != [2, 0]:
    raise SystemExit("Expected RF-DETR prediction arrays to pad short confidence/class arrays.")

bad_predictions = types.SimpleNamespace(xyxy=np.array([1, 2, 3], dtype=np.float32))
for reader in (prediction_arrays, RfDetrDetector._prediction_arrays):
    try:
        reader(bad_predictions)
    except RuntimeError as exc:
        if "xyxy" not in str(exc):
            raise SystemExit(f"Expected invalid RF-DETR xyxy error to mention xyxy, got: {exc}")
    else:
        raise SystemExit("Expected invalid RF-DETR xyxy output to fail clearly.")

print("rf-detr adapter fake module ok")
'@ | & $PythonExe -
}

Invoke-Step "RF-DETR PipeMode launcher branch" {
    $FakeRfDetrRoot = Join-Path $SmokeRoot "fake_rfdetr_launcher"
    New-Item -ItemType Directory -Force -Path $FakeRfDetrRoot | Out-Null
    $FakeModulePath = Join-Path $FakeRfDetrRoot "rfdetr.py"
    [System.IO.File]::WriteAllText(
        $FakeModulePath,
        @'
import numpy as np
from types import SimpleNamespace


class _FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, image, threshold=0.35):
        height, width = image.shape[:2]
        return SimpleNamespace(
            xyxy=np.array([[width * 0.35, height * 0.35, width * 0.55, height * 0.44]], dtype=np.float32),
            confidence=np.array([0.91], dtype=np.float32),
            class_id=np.array([0], dtype=np.int32),
        )


RFDETRNano = _FakeModel
RFDETRSmall = _FakeModel
RFDETRBase = _FakeModel
RFDETRMedium = _FakeModel
RFDETRLarge = _FakeModel
RFDETRXLarge = _FakeModel
RFDETR2XLarge = _FakeModel
'@,
        $Utf8NoBom
    )

    $DefaultRfDetrWeightDir = Join-Path $RepoRoot "runs_rfdetr\pipe_rfdetr_small"
    $DefaultRfDetrWeightPath = Join-Path $DefaultRfDetrWeightDir "checkpoint_best_total.pth"
    $CreatedDefaultRfDetrDir = $false
    $CreatedDefaultRfDetrWeight = $false
    if (-not (Test-Path -LiteralPath $DefaultRfDetrWeightDir)) {
        New-Item -ItemType Directory -Force -Path $DefaultRfDetrWeightDir | Out-Null
        $CreatedDefaultRfDetrDir = $true
    }
    if (-not (Test-Path -LiteralPath $DefaultRfDetrWeightPath)) {
        Set-Content -LiteralPath $DefaultRfDetrWeightPath -Value "fake checkpoint for smoke test" -Encoding UTF8
        $CreatedDefaultRfDetrWeight = $true
    }

    $OldPythonPath = $env:PYTHONPATH
    if ([string]::IsNullOrWhiteSpace($OldPythonPath)) {
        $env:PYTHONPATH = $FakeRfDetrRoot
    } else {
        $env:PYTHONPATH = "$FakeRfDetrRoot;$OldPythonPath"
    }

    try {
        $LauncherOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_crosscam.ps1 `
            -Demo `
            -Headless `
            -Frames 3 `
            -PipeMode `
            -Detector rfdetr `
            -RfDetrSize small `
            -SkipInstall `
            -LogDir (Join-Path $SmokeRoot "runs_rfdetr_launcher") 2>&1
        $LauncherExitCode = $LASTEXITCODE
    } finally {
        $env:PYTHONPATH = $OldPythonPath
        if ($CreatedDefaultRfDetrWeight) {
            Remove-Item -LiteralPath $DefaultRfDetrWeightPath -Force -ErrorAction SilentlyContinue
        }
        if ($CreatedDefaultRfDetrDir) {
            Remove-Item -LiteralPath $DefaultRfDetrWeightDir -Force -ErrorAction SilentlyContinue
        }
    }

    $LauncherText = $LauncherOutput -join "`n"
    if ($LauncherExitCode -ne 0) {
        Write-Host $LauncherText
        exit $LauncherExitCode
    }
    if ($LauncherText -notmatch "--detector rfdetr") {
        Write-Host $LauncherText
        Write-Host "Expected PipeMode launcher to preserve RF-DETR detector."
        exit 2
    }
    if ($LauncherText -match "--detector yolo") {
        Write-Host $LauncherText
        Write-Host "Expected explicit RF-DETR detector not to be overwritten by PipeMode."
        exit 2
    }
    if ($LauncherText -notmatch [regex]::Escape("--rfdetr-weights runs_rfdetr\pipe_rfdetr_small\checkpoint_best_total.pth")) {
        Write-Host $LauncherText
        Write-Host "Expected PipeMode RF-DETR launcher to auto-use the default trained checkpoint."
        exit 2
    }
    if ($LauncherText -notmatch "--rfdetr-num-classes 1" -or $LauncherText -notmatch "--rfdetr-classes 0") {
        Write-Host $LauncherText
        Write-Host "Expected PipeMode RF-DETR launcher to default to one pipe class when a trained checkpoint exists."
        exit 2
    }
    Write-Host "RF-DETR PipeMode launcher preserved detector."
}

Invoke-Step "Auto-label green pipe sample" {
    $AutoLabelImages = Join-Path $AutoLabelRoot "images"
    $AutoLabelLabels = Join-Path $AutoLabelRoot "labels"
    $AutoLabelPreviews = Join-Path $AutoLabelRoot "previews"
    New-Item -ItemType Directory -Force -Path $AutoLabelImages | Out-Null
    $env:CROSSCAM_SMOKE_AUTOLABEL = $AutoLabelImages
    @'
import cv2
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["CROSSCAM_SMOKE_AUTOLABEL"])
image = np.full((180, 420, 3), (45, 55, 65), dtype=np.uint8)
cv2.rectangle(image, (70, 75), (350, 110), (70, 180, 70), -1)
cv2.imwrite(str(root / "single_green_pipe.jpg"), image)
'@ | & $PythonExe -

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\auto_label_pipes.ps1 `
        -Images $AutoLabelImages `
        -Labels $AutoLabelLabels `
        -Previews $AutoLabelPreviews `
        -Preview `
        -Overwrite

    $LabelPath = Join-Path $AutoLabelLabels "single_green_pipe.txt"
    if (-not (Test-Path $LabelPath)) {
        Write-Host "Missing auto-label output: $LabelPath"
        exit 2
    }
    if (-not (Get-Content $LabelPath | Where-Object { $_.Trim() -ne "" })) {
        Write-Host "Auto-label output is empty: $LabelPath"
        exit 2
    }
    $PreviewPath = Join-Path $AutoLabelPreviews "single_green_pipe.jpg"
    if (-not (Test-Path $PreviewPath)) {
        Write-Host "Missing auto-label preview: $PreviewPath"
        exit 2
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\auto_label_pipes.ps1 `
        -Images (Join-Path $AutoLabelRoot "missing_images") `
        -Labels (Join-Path $AutoLabelRoot "missing_labels")
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected auto_label_pipes.ps1 to fail on a missing image directory."
        exit 2
    }
    Write-Host "Missing auto-label input rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Prepare sample YOLO export" {
    $env:CROSSCAM_SMOKE_EXPORT = $ExportRoot
    @'
import cv2
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["CROSSCAM_SMOKE_EXPORT"])
for i in range(1, 6):
    image = np.full((64, 128, 3), 20 + i * 30, dtype=np.uint8)
    cv2.rectangle(image, (24, 22), (104, 42), (0, 180, 220), -1)
    image_path = root / "images" / f"pipe_{i:03d}.jpg"
    label_path = root / "labels" / f"pipe_{i:03d}.txt"
    cv2.imwrite(str(image_path), image)
    label_path.write_text("0 0.500000 0.500000 0.625000 0.312500\n", encoding="utf-8")
'@ | & $PythonExe -
}

Invoke-Step "Prepare train/val dataset" {
    & $PythonExe src\prepare_yolo_dataset.py `
        --source-images (Join-Path $ExportRoot "images") `
        --source-labels (Join-Path $ExportRoot "labels") `
        --dataset-root $DatasetRoot `
        --clean
    $PreparedDataYaml = Join-Path $DatasetRoot "data.yaml"
    if (-not (Test-Path $PreparedDataYaml)) {
        Write-Host "Expected prepare_yolo_dataset.py to write data.yaml: $PreparedDataYaml"
        exit 2
    }

    $DuplicateExportRoot = Join-Path $SmokeRoot "duplicate_export"
    $DuplicateImages = Join-Path $DuplicateExportRoot "images"
    $DuplicateLabels = Join-Path $DuplicateExportRoot "labels"
    New-Item -ItemType Directory -Force -Path $DuplicateImages, $DuplicateLabels | Out-Null
    Copy-Item -LiteralPath (Join-Path $ExportRoot "images\pipe_001.jpg") -Destination (Join-Path $DuplicateImages "pipe_dup.jpg") -Force
    Copy-Item -LiteralPath (Join-Path $ExportRoot "images\pipe_001.jpg") -Destination (Join-Path $DuplicateImages "pipe_dup.png") -Force
    Copy-Item -LiteralPath (Join-Path $ExportRoot "labels\pipe_001.txt") -Destination (Join-Path $DuplicateLabels "pipe_dup.txt") -Force
    & $PythonExe src\prepare_yolo_dataset.py `
        --source-images $DuplicateImages `
        --source-labels $DuplicateLabels `
        --dataset-root (Join-Path $SmokeRoot "duplicate_dataset") `
        --clean
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected prepare_yolo_dataset.py to reject duplicate image stems."
        exit 2
    }
    Write-Host "Duplicate source image stems rejected."
    $global:LASTEXITCODE = 0

    $SafeDatasetRoot = Join-Path $SmokeRoot "safe_prepare_dataset"
    & $PythonExe src\prepare_yolo_dataset.py `
        --source-images (Join-Path $ExportRoot "images") `
        --source-labels (Join-Path $ExportRoot "labels") `
        --dataset-root $SafeDatasetRoot `
        --clean
    $ExistingSafeImageCount = @(Get-ChildItem -LiteralPath (Join-Path $SafeDatasetRoot "images\train") -File).Count
    if ($ExistingSafeImageCount -lt 1) {
        Write-Host "Expected safe prepare fixture to contain train images before failed clean retry."
        exit 2
    }
    & $PythonExe src\prepare_yolo_dataset.py `
        --source-images $DuplicateImages `
        --source-labels $DuplicateLabels `
        --dataset-root $SafeDatasetRoot `
        --clean
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected failed clean retry with duplicate source images."
        exit 2
    }
    $RemainingSafeImageCount = @(Get-ChildItem -LiteralPath (Join-Path $SafeDatasetRoot "images\train") -File).Count
    if ($RemainingSafeImageCount -ne $ExistingSafeImageCount) {
        Write-Host "Failed prepare retry should not clean the existing dataset before source validation."
        exit 2
    }
    Write-Host "Failed prepare retry preserved existing dataset."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Validate prepared dataset" {
    & $PythonExe src\validate_yolo_dataset.py --dataset-root $DatasetRoot

    $DuplicateDatasetRoot = Join-Path $SmokeRoot "duplicate_validate_dataset"
    Copy-Item -Recurse -Force -Path $DatasetRoot -Destination $DuplicateDatasetRoot
    $FirstTrainImage = Get-ChildItem -LiteralPath (Join-Path $DuplicateDatasetRoot "images\train") -Filter *.jpg | Select-Object -First 1
    if (-not $FirstTrainImage) {
        Write-Host "Expected at least one train image for duplicate validation test."
        exit 2
    }
    Copy-Item -LiteralPath $FirstTrainImage.FullName -Destination (Join-Path $FirstTrainImage.DirectoryName "$($FirstTrainImage.BaseName).png") -Force
    & $PythonExe src\validate_yolo_dataset.py --dataset-root $DuplicateDatasetRoot
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected validate_yolo_dataset.py to reject duplicate image stems."
        exit 2
    }
    Write-Host "Duplicate dataset image stems rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Export RF-DETR COCO dataset" {
    & $PythonExe src\export_yolo_to_coco.py `
        --yolo-root $DatasetRoot `
        --output-root $RfDetrDatasetRoot `
        --class-names pipe `
        --clean `
        --test-from-val
    foreach ($RelativeJson in @("train\_annotations.coco.json", "valid\_annotations.coco.json", "test\_annotations.coco.json")) {
        $JsonPath = Join-Path $RfDetrDatasetRoot $RelativeJson
        if (-not (Test-Path $JsonPath)) {
            Write-Host "Missing RF-DETR annotation JSON: $JsonPath"
            exit 2
        }
    }

    & $PythonExe src\export_yolo_to_coco.py `
        --yolo-root $DatasetRoot `
        --output-root (Join-Path $SmokeRoot "bad_rfdetr_export") `
        --class-names pipe,pipe
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected RF-DETR export to reject duplicate class names."
        exit 2
    }
    Write-Host "Invalid RF-DETR export class names rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Analyze sample YOLO predictions" {
    $PredLabelRoot = Join-Path $SmokeRoot "pred_labels"
    $EvalSummaryJson = Join-Path $SmokeRoot "yolo_eval_summary.json"
    $EvalSummaryMd = Join-Path $SmokeRoot "yolo_eval_summary.md"
    New-Item -ItemType Directory -Force -Path $PredLabelRoot | Out-Null
    Copy-Item -Path (Join-Path $DatasetRoot "labels\val\*.txt") -Destination $PredLabelRoot -Force
    & $PythonExe src\analyze_yolo_eval.py `
        --dataset-root $DatasetRoot `
        --split val `
        --pred-labels $PredLabelRoot `
        --report-csv (Join-Path $SmokeRoot "yolo_eval_analysis.csv") `
        --summary-json $EvalSummaryJson `
        --summary-md $EvalSummaryMd `
        --require-predictions `
        --min-precision 1.0 `
        --min-recall 1.0 `
        --max-false-positives 0 `
        --max-false-negatives 0
    if (-not (Test-Path -LiteralPath $EvalSummaryJson)) {
        Write-Host "Expected YOLO eval JSON summary: $EvalSummaryJson"
        exit 2
    }
    if (-not (Test-Path -LiteralPath $EvalSummaryMd)) {
        Write-Host "Expected YOLO eval Markdown summary: $EvalSummaryMd"
        exit 2
    }
    $EvalSummary = Get-Content $EvalSummaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $EvalSummary.passed -or $EvalSummary.summary.precision -ne 1 -or $EvalSummary.summary.recall -ne 1) {
        Write-Host "Expected YOLO eval JSON summary to pass with precision/recall 1.0."
        exit 2
    }
    $EvalSummaryMdText = Get-Content $EvalSummaryMd -Raw -Encoding UTF8
    if ($EvalSummaryMdText -notmatch "Precision" -or $EvalSummaryMdText -notmatch "Recall") {
        Write-Host "Expected YOLO eval Markdown summary to include precision and recall."
        exit 2
    }

    $BadPredLabelRoot = Join-Path $SmokeRoot "bad_pred_labels"
    New-Item -ItemType Directory -Force -Path $BadPredLabelRoot | Out-Null
    $FirstValLabel = Get-ChildItem -LiteralPath (Join-Path $DatasetRoot "labels\val") -Filter *.txt | Select-Object -First 1
    if (-not $FirstValLabel) {
        Write-Host "Expected at least one validation label for bad analyze test."
        exit 2
    }
    "0 not-a-number 0.500000 0.250000 0.250000" | Set-Content -LiteralPath (Join-Path $BadPredLabelRoot $FirstValLabel.Name) -Encoding UTF8
    $BadAnalyzeOutput = & $PythonExe src\analyze_yolo_eval.py `
        --dataset-root $DatasetRoot `
        --split val `
        --pred-labels $BadPredLabelRoot `
        --require-predictions 2>&1
    $BadAnalyzeExit = $LASTEXITCODE
    if ($BadAnalyzeExit -eq 0) {
        $BadAnalyzeOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected analyze_yolo_eval.py to reject invalid numeric YOLO labels."
        exit 2
    }
    $BadAnalyzeText = $BadAnalyzeOutput | Out-String
    if ($BadAnalyzeText -notmatch "not-a-number" -or $BadAnalyzeText -match "Traceback") {
        $BadAnalyzeOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected invalid YOLO eval labels to fail with a friendly message and no traceback."
        exit 2
    }
    Write-Host "Invalid YOLO eval label rejected."
    $global:LASTEXITCODE = 0

    & $PythonExe src\analyze_yolo_eval.py `
        --dataset-root $DatasetRoot `
        --split val `
        --pred-labels $PredLabelRoot `
        --iou-threshold 1.5 `
        --require-predictions
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected analyze_yolo_eval.py to reject invalid IoU threshold."
        exit 2
    }
    Write-Host "Invalid YOLO eval threshold rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Collect YOLO issue samples" {
    $IssuePredRoot = Join-Path $SmokeRoot "issue_pred_labels"
    $IssueAnalysisCsv = Join-Path $SmokeRoot "issue_analysis.csv"
    $IssueOutputDir = Join-Path $SmokeRoot "issue_samples"
    New-Item -ItemType Directory -Force -Path $IssuePredRoot | Out-Null
    & $PythonExe src\analyze_yolo_eval.py `
        --dataset-root $DatasetRoot `
        --split val `
        --pred-labels $IssuePredRoot `
        --report-csv $IssueAnalysisCsv `
        --require-predictions
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\collect_yolo_issues.ps1 `
        -AnalysisCsv $IssueAnalysisCsv `
        -DatasetRoot $DatasetRoot `
        -PredLabels $IssuePredRoot `
        -OutputDir $IssueOutputDir `
        -Preview `
        -Clean
    foreach ($RelativePath in @("issues.csv", "images", "labels_gt", "labels_pred", "previews")) {
        $ExpectedPath = Join-Path $IssueOutputDir $RelativePath
        if (-not (Test-Path $ExpectedPath)) {
            Write-Host "Missing issue sample output: $ExpectedPath"
            exit 2
        }
    }
    $IssueRows = @(Import-Csv (Join-Path $IssueOutputDir "issues.csv"))
    if ($IssueRows.Count -lt 1) {
        Write-Host "Expected at least one collected issue sample."
        exit 2
    }

    $BadPredRoot = Join-Path $SmokeRoot "bad_issue_pred_labels"
    $BadOutputDir = Join-Path $SmokeRoot "bad_issue_samples"
    New-Item -ItemType Directory -Force -Path $BadPredRoot | Out-Null
    $FirstIssueImage = (Import-Csv $IssueAnalysisCsv | Select-Object -First 1).image
    if (-not $FirstIssueImage) {
        Write-Host "Expected issue analysis CSV to contain an image name."
        exit 2
    }
    "0 not-a-number 0.500000 0.250000 0.250000" | Set-Content -LiteralPath (Join-Path $BadPredRoot "$FirstIssueImage.txt") -Encoding UTF8
    $BadCollectOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\collect_yolo_issues.ps1 `
        -AnalysisCsv $IssueAnalysisCsv `
        -DatasetRoot $DatasetRoot `
        -PredLabels $BadPredRoot `
        -OutputDir $BadOutputDir `
        -Preview `
        -Clean 2>&1
    $BadCollectExit = $LASTEXITCODE
    if ($BadCollectExit -eq 0) {
        $BadCollectOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected collect_yolo_issues.ps1 to reject invalid numeric YOLO labels."
        exit 2
    }
    $BadCollectText = $BadCollectOutput | Out-String
    if ($BadCollectText -notmatch "not-a-number" -or $BadCollectText -match "Traceback") {
        $BadCollectOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected invalid YOLO labels to fail with a friendly message and no traceback."
        exit 2
    }
    Write-Host "Invalid issue label rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Compare detector eval reports" {
    $SampleReportCsv = Join-Path $SmokeRoot "yolo_eval_analysis.csv"
    $CompareCsv = Join-Path $SmokeRoot "detector_compare.csv"
    $CompareSummaryJson = Join-Path $SmokeRoot "detector_compare_summary.json"
    $CompareSummaryMd = Join-Path $SmokeRoot "detector_compare_summary.md"
    $CompareWrapperOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\compare_detector_eval.ps1 -PrintOnly
    $CompareWrapperText = $CompareWrapperOutput -join "`n"
    foreach ($RequiredText in @(
        "--summary-json runs_detector_compare\yolo_vs_rfdetr_summary.json",
        "--summary-md runs_detector_compare\yolo_vs_rfdetr_summary.md"
    )) {
        if ($CompareWrapperText -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected compare detector wrapper to include: $RequiredText"
            exit 2
        }
    }
    & $PythonExe src\compare_detector_eval.py `
        --left-csv $SampleReportCsv `
        --right-csv $SampleReportCsv `
        --left-name YOLO `
        --right-name RF-DETR `
        --output-csv $CompareCsv `
        --summary-json $CompareSummaryJson `
        --summary-md $CompareSummaryMd
    if (-not (Test-Path -LiteralPath $CompareCsv)) {
        Write-Host "Expected detector comparison CSV: $CompareCsv"
        exit 2
    }
    if (-not (Test-Path -LiteralPath $CompareSummaryJson)) {
        Write-Host "Expected detector comparison JSON summary: $CompareSummaryJson"
        exit 2
    }
    if (-not (Test-Path -LiteralPath $CompareSummaryMd)) {
        Write-Host "Expected detector comparison Markdown summary: $CompareSummaryMd"
        exit 2
    }
    $CompareSummary = Get-Content $CompareSummaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($CompareSummary.left_name -ne "YOLO" -or $CompareSummary.right_name -ne "RF-DETR") {
        Write-Host "Expected detector comparison summary to keep detector names."
        exit 2
    }
    if ($CompareSummary.diff.precision_delta_right_minus_left -ne 0 -or $CompareSummary.diff.recall_delta_right_minus_left -ne 0) {
        Write-Host "Expected identical detector reports to have zero precision/recall deltas."
        exit 2
    }
    if ($CompareSummary.verdict.winner -ne "tie") {
        Write-Host "Expected identical detector reports to have tie verdict."
        exit 2
    }
    $CompareSummaryMdText = Get-Content $CompareSummaryMd -Raw -Encoding UTF8
    $MissingCompareSummaryText = @("YOLO", "RF-DETR") |
        Where-Object { $CompareSummaryMdText -notmatch [regex]::Escape($_) }
    if ($MissingCompareSummaryText.Count -gt 0) {
        Write-Host "Expected detector comparison Markdown summary to include detector names."
        exit 2
    }
    $MissingCompareOutput = & $PythonExe src\compare_detector_eval.py `
        --left-csv (Join-Path $SmokeRoot "missing_yolo_analysis.csv") `
        --right-csv (Join-Path $SmokeRoot "missing_rfdetr_analysis.csv") `
        --left-name YOLO `
        --right-name RF-DETR 2>&1
    $MissingCompareExit = $LASTEXITCODE
    if ($MissingCompareExit -eq 0) {
        $MissingCompareOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected compare_detector_eval.py to reject missing analysis CSV files."
        exit 2
    }
    $MissingCompareText = $MissingCompareOutput | Out-String
    foreach ($RequiredText in @(".\evaluate_pipe_yolo.bat", ".\evaluate_rfdetr.bat")) {
        if ($MissingCompareText -notmatch [regex]::Escape($RequiredText)) {
            $MissingCompareOutput | ForEach-Object { Write-Host $_ }
            Write-Host "Expected missing detector comparison input to suggest: $RequiredText"
            exit 2
        }
    }
    if ($MissingCompareText -match "Traceback") {
        $MissingCompareOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected missing detector comparison inputs to fail without traceback."
        exit 2
    }
    Write-Host "Missing detector comparison inputs rejected with next-step commands."
    $global:LASTEXITCODE = 0

    $BadReportCsv = Join-Path $SmokeRoot "bad_eval_analysis.csv"
    @'
image,gt,pred,matched,false_positive,false_negative,oversized,avg_iou
bad_001,1,1,1,0,0,0,not-a-number
'@ | Set-Content -LiteralPath $BadReportCsv -Encoding UTF8
    & $PythonExe src\compare_detector_eval.py `
        --left-csv $BadReportCsv `
        --right-csv $SampleReportCsv `
        --left-name Bad `
        --right-name Good
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected compare_detector_eval.py to reject invalid avg_iou."
        exit 2
    }
    Write-Host "Invalid detector comparison CSV rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Detector evaluation pipeline command generation" {
    $DetectorPipelineOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_detectors.ps1 `
        -PrintOnly `
        -SkipInstall `
        -CollectIssues `
        -IssuePreview `
        -IssueClean
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $DetectorPipelineText = $DetectorPipelineOutput -join "`n"
    foreach ($RequiredText in @(
        "scripts\evaluate_pipe_yolo.ps1",
        "-PredictOnly",
        "scripts\evaluate_rfdetr.ps1",
        "scripts\compare_detector_eval.ps1",
        "-YoloCsv runs_yolo_eval\pipe_yolov8n_eval_predict\analysis.csv",
        "-RfDetrCsv runs_rfdetr_eval\pipe_rfdetr_nano_eval\analysis.csv",
        "-CollectIssues",
        "-IssuePreview",
        "-IssueClean"
    )) {
        if ($DetectorPipelineText -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected detector evaluation pipeline command to include: $RequiredText"
            exit 2
        }
    }

    $Detector2xlargeOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_detectors.ps1 `
        -PrintOnly `
        -SkipInstall `
        -RfDetrModelSize 2xlarge
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $Detector2xlargeText = $Detector2xlargeOutput -join "`n"
    if ($Detector2xlargeText -notmatch [regex]::Escape("runs_rfdetr_eval\pipe_rfdetr_2xlarge_eval\analysis.csv")) {
        Write-Host "Expected detector evaluation pipeline to compare the 2xlarge RF-DETR output path."
        exit 2
    }

    $SpacedYoloProject = Join-Path $SmokeRoot "runs yolo eval"
    $DetectorSpacedOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_detectors.ps1 `
        -PrintOnly `
        -SkipInstall `
        -YoloProject $SpacedYoloProject
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $DetectorSpacedText = $DetectorSpacedOutput -join "`n"
    if ($DetectorSpacedText -notmatch [regex]::Escape("'$SpacedYoloProject'")) {
        Write-Host "Expected detector evaluation pipeline PrintOnly output to quote paths with spaces."
        exit 2
    }

    $FakeYoloModel = Join-Path $SmokeRoot "fake_yolo_best.pt"
    Set-Content -LiteralPath $FakeYoloModel -Value "fake checkpoint for detector pipeline check" -Encoding UTF8
    $DetectorCheckOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_detectors.ps1 `
        -CheckOnly `
        -SkipInstall `
        -DatasetRoot $DatasetRoot `
        -YoloData (Join-Path $DatasetRoot "data.yaml") `
        -YoloModel $FakeYoloModel `
        -YoloProject (Join-Path $SmokeRoot "runs_yolo_eval") `
        -RfDetrProject (Join-Path $SmokeRoot "runs_rfdetr_eval")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $DetectorCheckText = $DetectorCheckOutput -join "`n"
    foreach ($RequiredText in @(
        "Detector evaluation check:",
        "YOLO evaluation check:",
        "CheckOnly: detector evaluation inputs are ready",
        "model_size=nano"
    )) {
        if ($DetectorCheckText -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected detector evaluation check output to include: $RequiredText"
            exit 2
        }
    }

    $BadDetectorCheckOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_detectors.ps1 `
        -CheckOnly `
        -SkipInstall `
        -DatasetRoot $DatasetRoot `
        -YoloData (Join-Path $DatasetRoot "data.yaml") `
        -YoloModel (Join-Path $SmokeRoot "missing_yolo_best.pt") 2>&1
    $BadDetectorCheckExit = $LASTEXITCODE
    if ($BadDetectorCheckExit -eq 0) {
        $BadDetectorCheckOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected detector evaluation check to reject a missing YOLO model."
        exit 2
    }
    $BadDetectorCheckText = $BadDetectorCheckOutput | Out-String
    if ($BadDetectorCheckText -notmatch "YOLO model" -or $BadDetectorCheckText -match "Traceback") {
        $BadDetectorCheckOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected missing YOLO model to fail with a friendly message and no traceback."
        exit 2
    }
    Write-Host "Invalid detector evaluation check inputs rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Synthetic cross-camera handoff demo" {
    & $PythonExe src\crosscam_mvp.py `
        --demo `
        --auto-register-first `
        --headless `
        --frames $DemoFrames `
        --require-match `
        --log-dir $RunLogDir
}

Invoke-Step "Verify run config event" {
    $LatestEventLog = Get-ChildItem -LiteralPath $RunLogDir -Filter "*-events.csv" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $LatestEventLog) {
        Write-Host "Missing event log under: $RunLogDir"
        exit 2
    }
    $RunConfigRows = @(Import-Csv $LatestEventLog.FullName | Where-Object { $_.event_type -eq "run_config" })
    if ($RunConfigRows.Count -ne 1) {
        Write-Host "Expected exactly one run_config event in $($LatestEventLog.FullName), got $($RunConfigRows.Count)."
        exit 2
    }
    $RunConfigMessage = $RunConfigRows[0].message
    foreach ($RequiredText in @("detector=motion", "camera_count=2", "backend=auto")) {
        if ($RunConfigMessage -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected run_config message to include: $RequiredText"
            Write-Host $RunConfigMessage
            exit 2
        }
    }
    Write-Host "Run config event recorded."
}

Invoke-Step "Verify target sample index" {
    $TargetSampleCsv = Join-Path $RunLogDir "targets\target_samples.csv"
    if (-not (Test-Path $TargetSampleCsv)) {
        Write-Host "Missing target sample index: $TargetSampleCsv"
        exit 2
    }
    $SampleLines = Get-Content $TargetSampleCsv
    if ($SampleLines.Count -lt 2) {
        Write-Host "Target sample index has no sample rows: $TargetSampleCsv"
        exit 2
    }
}

Invoke-Step "Collect target samples" {
    $TargetSampleCsv = Join-Path $RunLogDir "targets\target_samples.csv"
    $TargetReviewDir = Join-Path $SmokeRoot "target_sample_review"
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\collect_target_samples.ps1 `
        -SamplesCsv $TargetSampleCsv `
        -OutputDir $TargetReviewDir `
        -MinCount 2 `
        -Preview `
        -Clean `
        -Strict
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    foreach ($RelativePath in @("target_samples.csv", "target_samples_preview.jpg", "images")) {
        $ExpectedPath = Join-Path $TargetReviewDir $RelativePath
        if (-not (Test-Path $ExpectedPath)) {
            Write-Host "Missing target sample review output: $ExpectedPath"
            exit 2
        }
    }
    $CollectedSamples = @(Import-Csv (Join-Path $TargetReviewDir "target_samples.csv"))
    if ($CollectedSamples.Count -lt 2) {
        Write-Host "Expected at least two collected target samples."
        exit 2
    }

    $MatchReviewDir = Join-Path $SmokeRoot "target_match_review"
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\collect_target_samples.ps1 `
        -SamplesCsv $TargetSampleCsv `
        -OutputDir $MatchReviewDir `
        -Source match `
        -MinCount 1 `
        -Clean `
        -Strict
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $MatchSamples = @(Import-Csv (Join-Path $MatchReviewDir "target_samples.csv"))
    if ($MatchSamples.Count -lt 1 -or ($MatchSamples | Where-Object { $_.source -ne "match" }).Count -gt 0) {
        Write-Host "Expected source=match filter to collect only matched target samples."
        exit 2
    }

    $BadSampleDir = Join-Path $SmokeRoot "bad_target_sample_review"
    $BadSampleCsv = Join-Path $SmokeRoot "bad-target-samples.csv"
    @(
        "time,camera,source,target_similarity,image",
        "2026-07-02 19:00:00,1,register,1.0000,missing-target.jpg"
    ) | Set-Content -LiteralPath $BadSampleCsv -Encoding UTF8
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\collect_target_samples.ps1 `
        -SamplesCsv $BadSampleCsv `
        -OutputDir $BadSampleDir `
        -Strict
    $CollectCode = $LASTEXITCODE
    if ($CollectCode -eq 0) {
        Write-Host "Expected collect_target_samples.ps1 -Strict to reject missing images."
        exit 2
    }
    if ($CollectCode -ne 2) {
        Write-Host "Unexpected target sample collect exit code: $CollectCode"
        exit $CollectCode
    }
    Write-Host "Missing target sample images rejected."
    & $PythonExe -c "pass"
}

Invoke-Step "Analyze handoff log" {
    $SummaryJson = Join-Path $SmokeRoot "handoff-summary.json"
    $SummaryMd = Join-Path $SmokeRoot "handoff-summary.md"
    & $PythonExe src\analyze_run_log.py `
        --log-dir $RunLogDir `
        --require-handoff `
        --min-target-matches 2 `
        --min-target-similarity 0.95 `
        --max-registered-lefts 2 `
        --max-new-after-register 0 `
        --max-target-switches 0 `
        --max-target-distance 20 `
        --max-target-jumps 0 `
        --min-cross-camera-ids 1 `
        --min-target-samples 2 `
        --min-match-samples 1 `
        --min-sample-cameras 2 `
        --max-unique-ids 1 `
        --summary-json $SummaryJson `
        --summary-md $SummaryMd
    $Summary = Get-Content $SummaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $Summary.passed -or -not $Summary.handoff_success) {
        Write-Host "Expected handoff summary to pass: $SummaryJson"
        exit 2
    }
    if ($Summary.target_lock_status -ne "handoff_ok") {
        Write-Host "Expected handoff summary target_lock_status=handoff_ok, got $($Summary.target_lock_status)."
        exit 2
    }
    if ($Summary.run_config.detector -ne "motion" -or $Summary.run_config.camera_count -ne "2" -or $Summary.run_config.backend -ne "auto") {
        Write-Host "Expected handoff summary to include run_config detector=motion, camera_count=2, backend=auto."
        exit 2
    }
    if (-not $Summary.target_lock_status_label -or $Summary.target_lock_status_label -eq $Summary.target_lock_status) {
        Write-Host "Expected handoff summary target_lock_status_label to explain success."
        exit 2
    }
    if ($Summary.target_sample_count -lt 2 -or $Summary.match_sample_count -lt 1) {
        Write-Host "Expected handoff summary to include target samples."
        exit 2
    }
    if ($Summary.target_sample_missing_image_count -ne 0) {
        Write-Host "Expected target sample images to exist."
        exit 2
    }
    if ($Summary.target_large_jump_count -ne 0) {
        Write-Host "Expected handoff summary to report zero large target jumps."
        exit 2
    }
    if (-not $Summary.recommended_actions -or $Summary.recommended_actions.Count -lt 1) {
        Write-Host "Expected passing summary to include a usable recommended action."
        exit 2
    }
    if (-not (Test-Path -LiteralPath $SummaryMd)) {
        Write-Host "Expected markdown handoff summary: $SummaryMd"
        exit 2
    }
    $SummaryMdText = Get-Content $SummaryMd -Raw -Encoding UTF8
    if ($SummaryMdText -notmatch "handoff_ok" -or $SummaryMdText -notmatch "target_samples.csv" -or $SummaryMdText -notmatch "detector=motion") {
        Write-Host "Expected markdown handoff summary to include target lock status, target sample index, and run config."
        exit 2
    }
}

Invoke-Step "Analyze unregistered target diagnosis" {
    $UnregisteredLog = Join-Path $SmokeRoot "unregistered-events.csv"
    $UnregisteredSummaryJson = Join-Path $SmokeRoot "unregistered-summary.json"
    @(
        "time,event_type,camera,global_id,local_id,similarity,target_similarity,target_choice,target_distance,bbox,message",
        '1,new,1,G001,1,0.8,,,,,"new candidate"',
        '2,track_created,1,G001,1,0.8,,,,,"track created"'
    ) | Set-Content -LiteralPath $UnregisteredLog -Encoding UTF8
    & $PythonExe src\analyze_run_log.py `
        --log $UnregisteredLog `
        --summary-json $UnregisteredSummaryJson
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $UnregisteredSummary = Get-Content $UnregisteredSummaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $UnregisteredSummary.diagnosis -or $UnregisteredSummary.diagnosis.Count -lt 1) {
        Write-Host "Expected analyzer to diagnose detected-but-unregistered target."
        exit 2
    }
    if ($UnregisteredSummary.target_lock_status -ne "needs_registration") {
        Write-Host "Expected unregistered summary target_lock_status=needs_registration, got $($UnregisteredSummary.target_lock_status)."
        exit 2
    }
    if (-not $UnregisteredSummary.target_lock_status_label -or $UnregisteredSummary.target_lock_status_label -eq $UnregisteredSummary.target_lock_status) {
        Write-Host "Expected unregistered summary target_lock_status_label to mention registration."
        exit 2
    }
    $UnregisteredActionsText = $UnregisteredSummary.recommended_actions | Out-String
    if ($UnregisteredActionsText -notmatch "target_registered") {
        Write-Host "Expected unregistered summary to recommend registering the target."
        exit 2
    }
}

Invoke-Step "Analyze blocked target candidate diagnosis" {
    $BlockedLog = Join-Path $SmokeRoot "blocked-candidate-events.csv"
    $BlockedSummaryJson = Join-Path $SmokeRoot "blocked-candidate-summary.json"
    @(
        "time,event_type,camera,global_id,local_id,similarity,target_similarity,target_choice,target_distance,bbox,message",
        '1,target_registered,1,G001,1,1.0,1.0,register,0,"0,0,10,10",registered',
        '2,new,2,G002,1,,0.91,blocked,35,"10,10,20,20","blocked similar candidate"'
    ) | Set-Content -LiteralPath $BlockedLog -Encoding UTF8
    & $PythonExe src\analyze_run_log.py `
        --log $BlockedLog `
        --summary-json $BlockedSummaryJson
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $BlockedSummary = Get-Content $BlockedSummaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($BlockedSummary.blocked_target_candidate_count -ne 1) {
        Write-Host "Expected analyzer to count blocked target candidates."
        exit 2
    }
    if ($BlockedSummary.target_lock_status -ne "blocked_candidate") {
        Write-Host "Expected blocked summary target_lock_status=blocked_candidate, got $($BlockedSummary.target_lock_status)."
        exit 2
    }
    if (-not $BlockedSummary.target_lock_status_label -or $BlockedSummary.target_lock_status_label -eq $BlockedSummary.target_lock_status) {
        Write-Host "Expected blocked summary target_lock_status_label to mention blocked candidates."
        exit 2
    }
    if (-not $BlockedSummary.diagnosis -or $BlockedSummary.diagnosis.Count -lt 2) {
        Write-Host "Expected analyzer to explain blocked target candidates."
        exit 2
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\analyze_run.ps1 `
        -Log $BlockedLog `
        -MaxBlockedTargetCandidates 0
    $AnalyzeCode = $LASTEXITCODE
    if ($AnalyzeCode -eq 0) {
        Write-Host "Expected blocked target candidate gate to reject the log."
        exit 2
    }
    if ($AnalyzeCode -ne 2) {
        Write-Host "Unexpected blocked target candidate gate exit code: $AnalyzeCode"
        exit $AnalyzeCode
    }
    Write-Host "Invalid blocked target candidate count rejected."
    & $PythonExe -c "pass"
    $DefaultGateSummaryJson = Join-Path $SmokeRoot "latest-target-lock-summary.json"
    $DefaultGateSummaryMd = Join-Path $SmokeRoot "latest-target-lock-summary.md"
    Remove-Item -LiteralPath $DefaultGateSummaryJson -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $DefaultGateSummaryMd -ErrorAction SilentlyContinue
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\analyze_run.ps1 `
        -Log $BlockedLog `
        -TargetLockGate
    $GateCode = $LASTEXITCODE
    if ($GateCode -eq 0) {
        Write-Host "Expected target-lock gate preset to reject the blocked candidate log."
        exit 2
    }
    if ($GateCode -ne 2) {
        Write-Host "Unexpected target-lock gate preset exit code: $GateCode"
        exit $GateCode
    }
    if (-not (Test-Path -LiteralPath $DefaultGateSummaryJson)) {
        Write-Host "Expected target-lock gate to write default summary JSON: $DefaultGateSummaryJson"
        exit 2
    }
    if (-not (Test-Path -LiteralPath $DefaultGateSummaryMd)) {
        Write-Host "Expected target-lock gate to write default markdown summary: $DefaultGateSummaryMd"
        exit 2
    }
    $DefaultGateSummary = Get-Content $DefaultGateSummaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($DefaultGateSummary.passed -ne $false -or $DefaultGateSummary.blocked_target_candidate_count -ne 1) {
        Write-Host "Unexpected default target-lock summary contents."
        exit 2
    }
    if ($DefaultGateSummary.target_lock_status -ne "blocked_candidate") {
        Write-Host "Expected default target-lock summary status blocked_candidate, got $($DefaultGateSummary.target_lock_status)."
        exit 2
    }
    $DefaultGateActionsText = $DefaultGateSummary.recommended_actions | Out-String
    if ($DefaultGateActionsText -notmatch "G001") {
        Write-Host "Expected blocked target summary to recommend checking the blocked similar candidate."
        exit 2
    }
    Write-Host "Target-lock gate preset rejected unstable log."
    & $PythonExe -c "pass"
}

Invoke-Step "Analyze target jump gate" {
    $JumpLog = Join-Path $SmokeRoot "jump-events.csv"
    @(
        "time,event_type,camera,global_id,local_id,similarity,target_similarity,target_choice,target_distance,bbox,message",
        '1,target_registered,1,1,1,1.0,1.0,register,0,"0,0,10,10",registered',
        '2,target_refreshed,1,1,1,0.9,0.9,switch,450,"450,0,10,10",jumped'
    ) | Set-Content -LiteralPath $JumpLog -Encoding UTF8
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\analyze_run.ps1 -Log $JumpLog -MaxTargetDistance 100 -MaxTargetJumps 0
    $AnalyzeCode = $LASTEXITCODE
    if ($AnalyzeCode -eq 0) {
        Write-Host "Expected max-target-distance gate to reject jump log."
        exit 2
    }
    if ($AnalyzeCode -ne 2) {
        Write-Host "Unexpected analyzer exit code: $AnalyzeCode"
        exit $AnalyzeCode
    }
    Write-Host "Invalid target jump rejected."
    & $PythonExe -c "pass"
}

Invoke-Step "Analyze target sample gate" {
    $SampleGateLog = Join-Path $SmokeRoot "sample-gate-events.csv"
    $SampleGateDir = Join-Path $SmokeRoot "sample_gate_targets"
    New-Item -ItemType Directory -Force -Path $SampleGateDir | Out-Null
    $SampleCsv = Join-Path $SampleGateDir "target_samples.csv"
    @(
        "time,event_type,camera,global_id,local_id,similarity,target_similarity,target_choice,target_distance,bbox,message",
        '1,target_registered,1,1,1,1.0,1.0,register,0,"0,0,10,10",registered',
        '2,target_matched,1,1,1,0.9,0.9,near,0,"0,0,10,10",matched'
    ) | Set-Content -LiteralPath $SampleGateLog -Encoding UTF8
    @(
        "time,camera,source,target_similarity,image",
        "2026-07-02 19:00:00,1,register,1.0000,missing-register.jpg"
    ) | Set-Content -LiteralPath $SampleCsv -Encoding UTF8
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\analyze_run.ps1 `
        -Log $SampleGateLog `
        -TargetSamplesCsv $SampleCsv `
        -MinTargetSamples 2 `
        -MinMatchSamples 1 `
        -MinSampleCameras 2
    $AnalyzeCode = $LASTEXITCODE
    if ($AnalyzeCode -eq 0) {
        Write-Host "Expected target sample gate to reject missing or insufficient samples."
        exit 2
    }
    if ($AnalyzeCode -ne 2) {
        Write-Host "Unexpected target sample gate exit code: $AnalyzeCode"
        exit $AnalyzeCode
    }
    Write-Host "Invalid target samples rejected."
    & $PythonExe -c "pass"
}

Invoke-Step "Launcher auto analysis" {
    $LauncherLogDir = Join-Path $SmokeRoot "runs_launcher"
    $LauncherSummaryJson = Join-Path $LauncherLogDir "latest-summary.json"
    $LauncherSummaryMd = Join-Path $LauncherLogDir "latest-summary.md"
    $LauncherTargetReviewDir = Join-Path $LauncherLogDir "target_sample_review"
    $LauncherOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_crosscam.ps1 `
        -Demo `
        -AutoRegisterFirst `
        -Headless `
        -Frames $DemoFrames `
        -RequireMatch `
        -LogDir $LauncherLogDir `
        -AnalyzeAfterRun `
        -AnalyzeRequireHandoff `
        -AnalyzeTargetLockGate `
        -AnalyzeMinTargetMatches 2 `
        -AnalyzeMinTargetSimilarity 0.95 `
        -AnalyzeMaxNewAfterRegister 0 `
        -AnalyzeMaxRegisteredLefts 2 `
        -AnalyzeMaxTargetSwitches 0 `
        -AnalyzeMaxTargetDistance 20 `
        -AnalyzeMaxTargetJumps 0 `
        -AnalyzeMaxBlockedTargetCandidates 0 `
        -AnalyzeMinCrossCameraIds 1 `
        -AnalyzeMinTargetSamples 2 `
        -AnalyzeMinMatchSamples 1 `
        -AnalyzeMinSampleCameras 2 `
        -AnalyzeMaxUniqueIds 1 `
        -SkipInstall 2>&1
    $LauncherExitCode = $LASTEXITCODE
    $LauncherOutput | ForEach-Object { Write-Host $_ }
    if ($LauncherExitCode -ne 0) {
        exit $LauncherExitCode
    }
    $LauncherOutputText = $LauncherOutput | Out-String
    if ($LauncherOutputText -notmatch "Launch summary:" -or $LauncherOutputText -notmatch "detector: motion") {
        Write-Host "Expected launcher output to include launch summary."
        exit 2
    }
    if ($LauncherOutputText -notmatch "Run analysis quick summary" -or $LauncherOutputText -notmatch "handoff_ok") {
        Write-Host "Expected launcher output to include quick target-lock summary."
        exit 2
    }
    if ($LauncherOutputText -notmatch "report:") {
        Write-Host "Expected launcher output to include markdown report path."
        exit 2
    }
    if (-not (Test-Path -LiteralPath $LauncherSummaryMd)) {
        Write-Host "Expected launcher markdown summary: $LauncherSummaryMd"
        exit 2
    }
    $LauncherSummaryMdText = Get-Content $LauncherSummaryMd -Raw -Encoding UTF8
    if (
        $LauncherSummaryMdText -notmatch "handoff_ok" -or
        $LauncherSummaryMdText -notmatch "target_samples.csv" -or
        $LauncherSummaryMdText -notmatch "target_samples_preview.jpg"
    ) {
        Write-Host "Expected launcher markdown summary to include target lock status and target sample review paths."
        exit 2
    }
    $LauncherSummary = Get-Content $LauncherSummaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $LauncherSummary.passed -or -not $LauncherSummary.handoff_success) {
        Write-Host "Expected launcher analysis summary to pass: $LauncherSummaryJson"
        exit 2
    }
    if ($LauncherSummary.target_lock_status -ne "handoff_ok") {
        Write-Host "Expected launcher summary target_lock_status=handoff_ok, got $($LauncherSummary.target_lock_status)."
        exit 2
    }
    if ($LauncherSummary.target_sample_count -lt 2 -or $LauncherSummary.match_sample_count -lt 1) {
        Write-Host "Expected launcher analysis summary to include target samples."
        exit 2
    }
    foreach ($RelativePath in @("target_samples.csv", "target_samples_preview.jpg", "images")) {
        $ExpectedPath = Join-Path $LauncherTargetReviewDir $RelativePath
        if (-not (Test-Path $ExpectedPath)) {
            Write-Host "Missing launcher target sample review output: $ExpectedPath"
            exit 2
        }
    }
}

Invoke-Step "Launcher failed analysis still collects target samples" {
    $FailedGateLogDir = Join-Path $SmokeRoot "runs_launcher_failed_gate"
    $FailedGateSummaryMd = Join-Path $FailedGateLogDir "latest-summary.md"
    $FailedGateReviewDir = Join-Path $FailedGateLogDir "target_sample_review"
    $FailedGateOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_crosscam.ps1 `
        -Demo `
        -AutoRegisterFirst `
        -Headless `
        -Frames $DemoFrames `
        -RequireMatch `
        -LogDir $FailedGateLogDir `
        -AnalyzeAfterRun `
        -AnalyzeTargetLockGate `
        -AnalyzeMinTargetMatches 999 `
        -SkipInstall 2>&1
    $FailedGateExitCode = $LASTEXITCODE
    if ($FailedGateExitCode -eq 0) {
        $FailedGateOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected launcher failed gate run to return a non-zero analysis exit code."
        exit 2
    }
    if (-not (Test-Path -LiteralPath (Join-Path $FailedGateReviewDir "target_samples_preview.jpg"))) {
        $FailedGateOutput | ForEach-Object { Write-Host $_ }
        Write-Host "Expected target sample preview to be collected even when analysis fails."
        exit 2
    }
    if (-not (Test-Path -LiteralPath $FailedGateSummaryMd)) {
        Write-Host "Expected markdown summary after failed gate: $FailedGateSummaryMd"
        exit 2
    }
    $FailedGateSummaryMdText = Get-Content $FailedGateSummaryMd -Raw -Encoding UTF8
    if (
        $FailedGateSummaryMdText -notmatch "Target Sample Review" -or
        $FailedGateSummaryMdText -notmatch "target_samples_preview.jpg"
    ) {
        Write-Host "Expected failed-gate markdown summary to include target sample review paths."
        exit 2
    }
    Write-Host "Failed analysis still collected target samples."
    & $PythonExe -c "pass"
}

Invoke-Step "Launcher fallback demo" {
    $FallbackLogDir = Join-Path $SmokeRoot "runs_fallback"
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_crosscam.ps1 `
        -CameraIndexes "999,998" `
        -ProbeMax 7 `
        -FallbackDemo `
        -Headless `
        -Frames 3 `
        -MaxMissed 18 `
        -LostTtl 9.5 `
        -LogDir $FallbackLogDir `
        -SkipInstall
}

Invoke-Step "Launcher print-only summary" {
    $PrintOnlyLogDir = Join-Path $SmokeRoot "runs_print_only"
    $PrintOnlyOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_crosscam.ps1 `
        -Demo `
        -PipeMode `
        -Headless `
        -Frames 1 `
        -LogDir $PrintOnlyLogDir `
        -PrintOnly 2>&1
    $PrintOnlyExit = $LASTEXITCODE
    $PrintOnlyOutput | ForEach-Object { Write-Host $_ }
    if ($PrintOnlyExit -ne 0) {
        exit $PrintOnlyExit
    }
    $PrintOnlyText = $PrintOnlyOutput | Out-String
    if (
        $PrintOnlyText -notmatch "Preparing CrossCamReID launch" -or
        $PrintOnlyText -notmatch "Launch summary:" -or
        $PrintOnlyText -notmatch "detector: yolo" -or
        $PrintOnlyText -notmatch "PrintOnly: CrossCamReID was not started"
    ) {
        Write-Host "Expected launcher PrintOnly output to include summary and not-started message."
        exit 2
    }
    if (Test-Path -LiteralPath $PrintOnlyLogDir) {
        Write-Host "Expected launcher PrintOnly not to create log directory: $PrintOnlyLogDir"
        exit 2
    }
}

Invoke-Step "Training script check branch" {
    $PreparedDataYaml = Join-Path $DatasetRoot "data.yaml"
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_pipe_yolo.ps1 `
        -Data $PreparedDataYaml `
        -DatasetRoot $DatasetRoot `
        -CheckOnly `
        -SkipInstall

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_pipe_yolo.ps1 `
        -Data $PreparedDataYaml `
        -DatasetRoot $DatasetRoot `
        -Epochs 0 `
        -CheckOnly `
        -SkipInstall `
        -SkipValidate
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected train_pipe_yolo.ps1 to reject non-positive epochs."
        exit 2
    }
    Write-Host "Invalid YOLO training parameters rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "Evaluation command generation" {
    $YoloEvalOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_pipe_yolo.ps1 `
        -PrintOnly `
        -SkipInstall `
        -CollectIssues `
        -IssuePreview `
        -IssueClean
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $YoloEvalText = $YoloEvalOutput -join "`n"
    foreach ($RequiredText in @(
        "--summary-json runs_yolo_eval\pipe_yolov8n_eval_predict\analysis_summary.json",
        "--summary-md runs_yolo_eval\pipe_yolov8n_eval_predict\analysis_summary.md"
    )) {
        if ($YoloEvalText -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected YOLO evaluation command to include: $RequiredText"
            exit 2
        }
    }

    $YoloCheckModel = Join-Path $SmokeRoot "fake_yolo_eval_best.pt"
    Set-Content -LiteralPath $YoloCheckModel -Value "fake checkpoint for YOLO eval check" -Encoding UTF8
    $YoloCheckOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_pipe_yolo.ps1 `
        -CheckOnly `
        -SkipInstall `
        -PredictOnly `
        -Model $YoloCheckModel `
        -Data (Join-Path $DatasetRoot "data.yaml") `
        -DatasetRoot $DatasetRoot `
        -Source (Join-Path $DatasetRoot "images\val")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $YoloCheckText = $YoloCheckOutput -join "`n"
    foreach ($RequiredText in @(
        "YOLO evaluation check:",
        "CheckOnly: YOLO evaluation inputs are ready"
    )) {
        if ($YoloCheckText -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected YOLO evaluation check output to include: $RequiredText"
            exit 2
        }
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_pipe_yolo.ps1 `
        -PrintOnly `
        -SkipInstall `
        -ValOnly `
        -CollectIssues
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected -ValOnly -CollectIssues to fail because issue collection needs predictions and analysis."
        exit 2
    }
    Write-Host "Invalid evaluation option combination rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "RF-DETR training script check branch" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_rfdetr.ps1 `
        -DatasetDir $RfDetrDatasetRoot `
        -OutputDir (Join-Path $SmokeRoot "runs_rfdetr") `
        -CheckOnly `
        -SkipInstall

    $Train2xlargeOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_rfdetr.ps1 `
        -DatasetDir $RfDetrDatasetRoot `
        -ModelSize 2xlarge `
        -CheckOnly `
        -SkipInstall 2>&1
    $Train2xlargeExit = $LASTEXITCODE
    $Train2xlargeOutput | ForEach-Object { Write-Host $_ }
    if ($Train2xlargeExit -ne 0) {
        exit $Train2xlargeExit
    }
    $Train2xlargeText = $Train2xlargeOutput | Out-String
    if ($Train2xlargeText -notmatch "pipe_rfdetr_2xlarge") {
        Write-Host "Expected 2xlarge RF-DETR training default output dir to include pipe_rfdetr_2xlarge."
        exit 2
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_rfdetr.ps1 `
        -DatasetDir $RfDetrDatasetRoot `
        -Epochs 0 `
        -CheckOnly `
        -SkipInstall
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected RF-DETR training check to reject non-positive epochs."
        exit 2
    }
    Write-Host "Invalid RF-DETR training parameters rejected."
    $global:LASTEXITCODE = 0

    $BadRfDetrDatasetRoot = Join-Path $SmokeRoot "dataset_rfdetr_bad_annotation"
    Copy-Item -Recurse -Force -Path $RfDetrDatasetRoot -Destination $BadRfDetrDatasetRoot
    $BadAnnotationJson = Join-Path $BadRfDetrDatasetRoot "train\_annotations.coco.json"
    $env:CROSSCAM_BAD_RFDETR_JSON = $BadAnnotationJson
    @'
import json
import os
from pathlib import Path

path = Path(os.environ["CROSSCAM_BAD_RFDETR_JSON"])
data = json.loads(path.read_text(encoding="utf-8"))
data["annotations"][0]["image_id"] = 999999
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
'@ | & $PythonExe -
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_rfdetr.ps1 `
        -DatasetDir $BadRfDetrDatasetRoot `
        -CheckOnly `
        -SkipInstall
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected RF-DETR training check to reject invalid annotation image_id."
        exit 2
    }
    Write-Host "Invalid RF-DETR annotation references rejected."
    $global:LASTEXITCODE = 0

    & $PythonExe src\export_yolo_to_coco.py `
        --yolo-root $DatasetRoot `
        --output-root (Join-Path $SmokeRoot "dataset_rfdetr_two_classes") `
        --class-names pipe,extra `
        --clean `
        --test-from-val
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_rfdetr.ps1 `
        -DatasetDir (Join-Path $SmokeRoot "dataset_rfdetr_two_classes") `
        -NumClasses 1 `
        -CheckOnly `
        -SkipInstall
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected RF-DETR training check to reject num-classes smaller than dataset categories."
        exit 2
    }
    Write-Host "Too-small RF-DETR num-classes rejected."
    $global:LASTEXITCODE = 0
}

Invoke-Step "RF-DETR evaluation script check branch" {
    $RfDetrEvalPrintOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_rfdetr.ps1 `
        -DatasetRoot $DatasetRoot `
        -Project (Join-Path $SmokeRoot "runs_rfdetr_eval") `
        -Name "pipe_rfdetr_nano_eval" `
        -PrintOnly `
        -CollectIssues `
        -IssuePreview `
        -IssueClean `
        -SkipInstall
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $RfDetrEvalPrintText = $RfDetrEvalPrintOutput -join "`n"
    foreach ($RequiredText in @(
        "--summary-json $SmokeRoot\runs_rfdetr_eval\pipe_rfdetr_nano_eval\analysis_summary.json",
        "--summary-md $SmokeRoot\runs_rfdetr_eval\pipe_rfdetr_nano_eval\analysis_summary.md",
        "src\collect_yolo_issues.py",
        "--output-dir $SmokeRoot\runs_rfdetr_eval\pipe_rfdetr_nano_eval\issue_samples",
        "--preview",
        "--clean"
    )) {
        if ($RfDetrEvalPrintText -notmatch [regex]::Escape($RequiredText)) {
            Write-Host "Expected RF-DETR evaluation command to include: $RequiredText"
            exit 2
        }
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_rfdetr.ps1 `
        -DatasetRoot $DatasetRoot `
        -Project (Join-Path $SmokeRoot "runs_rfdetr_eval") `
        -Name "pipe_rfdetr_nano_eval" `
        -CheckOnly `
        -CollectIssues `
        -SkipInstall
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected -CheckOnly -CollectIssues to fail because issue collection needs predictions and analysis."
        exit 2
    }
    Write-Host "Invalid RF-DETR issue collection option combination rejected."
    $global:LASTEXITCODE = 0

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_rfdetr.ps1 `
        -DatasetRoot $DatasetRoot `
        -Project (Join-Path $SmokeRoot "runs_rfdetr_eval") `
        -Name "pipe_rfdetr_nano_eval" `
        -CheckOnly `
        -SkipInstall

    $Eval2xlargeOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_rfdetr.ps1 `
        -DatasetRoot $DatasetRoot `
        -Project (Join-Path $SmokeRoot "runs_rfdetr_eval") `
        -ModelSize 2xlarge `
        -CheckOnly `
        -SkipInstall 2>&1
    $Eval2xlargeExit = $LASTEXITCODE
    $Eval2xlargeOutput | ForEach-Object { Write-Host $_ }
    if ($Eval2xlargeExit -ne 0) {
        exit $Eval2xlargeExit
    }
    $Eval2xlargeText = $Eval2xlargeOutput | Out-String
    if ($Eval2xlargeText -notmatch "pipe_rfdetr_2xlarge_eval") {
        Write-Host "Expected 2xlarge RF-DETR evaluation default name to include pipe_rfdetr_2xlarge_eval."
        exit 2
    }

    $DefaultRfDetrWeightDir = Join-Path $RepoRoot "runs_rfdetr\pipe_rfdetr_small"
    $DefaultRfDetrWeightPath = Join-Path $DefaultRfDetrWeightDir "checkpoint_best_total.pth"
    $CreatedDefaultRfDetrDir = $false
    $CreatedDefaultRfDetrWeight = $false
    if (-not (Test-Path -LiteralPath $DefaultRfDetrWeightDir)) {
        New-Item -ItemType Directory -Force -Path $DefaultRfDetrWeightDir | Out-Null
        $CreatedDefaultRfDetrDir = $true
    }
    if (-not (Test-Path -LiteralPath $DefaultRfDetrWeightPath)) {
        Set-Content -LiteralPath $DefaultRfDetrWeightPath -Value "fake checkpoint for smoke test" -Encoding UTF8
        $CreatedDefaultRfDetrWeight = $true
    }
    try {
        $EvalDefaultWeightOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_rfdetr.ps1 `
            -DatasetRoot $DatasetRoot `
            -Project (Join-Path $SmokeRoot "runs_rfdetr_eval") `
            -ModelSize small `
            -CheckOnly `
            -SkipInstall 2>&1
        $EvalDefaultWeightExit = $LASTEXITCODE
    } finally {
        if ($CreatedDefaultRfDetrWeight) {
            Remove-Item -LiteralPath $DefaultRfDetrWeightPath -Force -ErrorAction SilentlyContinue
        }
        if ($CreatedDefaultRfDetrDir) {
            Remove-Item -LiteralPath $DefaultRfDetrWeightDir -Force -ErrorAction SilentlyContinue
        }
    }
    $EvalDefaultWeightOutput | ForEach-Object { Write-Host $_ }
    if ($EvalDefaultWeightExit -ne 0) {
        exit $EvalDefaultWeightExit
    }
    $EvalDefaultWeightText = $EvalDefaultWeightOutput | Out-String
    if ($EvalDefaultWeightText -notmatch [regex]::Escape("runs_rfdetr\pipe_rfdetr_small\checkpoint_best_total.pth")) {
        Write-Host "Expected RF-DETR evaluation to auto-use the default trained checkpoint."
        exit 2
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_rfdetr.ps1 `
        -DatasetRoot $DatasetRoot `
        -Conf 1.5 `
        -CheckOnly `
        -SkipInstall
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected RF-DETR evaluation check to reject invalid confidence."
        exit 2
    }
    Write-Host "Invalid RF-DETR evaluation parameters rejected."
    $global:LASTEXITCODE = 0

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_rfdetr.ps1 `
        -DatasetRoot $DatasetRoot `
        -Classes 3 `
        -CheckOnly `
        -SkipInstall
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Expected RF-DETR evaluation check to reject out-of-range class filters."
        exit 2
    }
    Write-Host "Invalid RF-DETR evaluation class filters rejected."
    $global:LASTEXITCODE = 0
}

if (-not $SkipYolo) {
    Invoke-Step "YOLO detector smoke path" {
        $YoloSmokeOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_crosscam.ps1 -Demo -PipeMode -Headless -Frames 1 -LogDir $RunLogDir -SkipInstall 2>&1
        $YoloSmokeExit = $LASTEXITCODE
        $YoloSmokeOutput | ForEach-Object { Write-Host $_ }
        if ($YoloSmokeExit -ne 0) {
            exit $YoloSmokeExit
        }
        $YoloSmokeText = $YoloSmokeOutput | Out-String
        if ($YoloSmokeText -notmatch "Launch summary:" -or $YoloSmokeText -notmatch "detector: yolo") {
            Write-Host "Expected YOLO smoke output to include launch summary."
            exit 2
        }
    }
}

Write-Host ""
Write-Host "Smoke test passed."
Write-Host "Temporary files: $SmokeRoot"
