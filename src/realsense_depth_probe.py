from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Intel RealSense depth availability for CrossCamReID.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only list RealSense devices.")
    parser.add_argument("--frames", type=int, default=30, help="Frames to sample when depth stream is started.")
    parser.add_argument("--width", type=int, default=640, help="Depth/color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Depth/color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS.")
    return parser.parse_args()


def load_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "未安装 pyrealsense2，无法读取 RealSense 深度流。"
            "请先确认 RealSense SDK Python 绑定可用。"
        ) from exc
    return rs


def device_info(rs, device) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "usb": rs.camera_info.usb_type_descriptor,
        "port": rs.camera_info.physical_port,
    }
    info: dict[str, str] = {}
    for key, field in fields.items():
        try:
            info[key] = device.get_info(field)
        except RuntimeError:
            info[key] = ""
    return info


def list_devices(rs) -> list:
    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        print("没有发现 RealSense 设备。")
        return []

    print(f"发现 RealSense 设备：{len(devices)} 个")
    for index, device in enumerate(devices):
        info = device_info(rs, device)
        print(
            f"  [{index}] {info['name']} serial={info['serial']} "
            f"firmware={info['firmware']} usb={info['usb']} port={info['port']}"
        )
    return devices


def sample_depth(rs, args: argparse.Namespace) -> int:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        print(f"深度比例 depth_scale={depth_scale:.6f} 米/单位")

        valid_ratios: list[float] = []
        center_depths: list[float] = []
        for _ in range(max(1, args.frames)):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth = np.asanyarray(depth_frame.get_data())
            valid = depth > 0
            valid_ratios.append(float(np.count_nonzero(valid)) / float(depth.size))
            center_depths.append(float(depth_frame.get_distance(args.width // 2, args.height // 2)))
            time.sleep(0.005)

        if not valid_ratios:
            print("没有采样到有效深度帧。")
            return 2

        print(
            "深度采样："
            f"frames={len(valid_ratios)}, "
            f"valid_ratio_avg={np.mean(valid_ratios):.3f}, "
            f"center_depth_m_avg={np.mean(center_depths):.3f}"
        )
        return 0
    finally:
        pipeline.stop()


def main() -> int:
    args = parse_args()
    try:
        rs = load_realsense()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    devices = list_devices(rs)
    if not devices:
        return 2
    if args.list_only:
        return 0
    return sample_depth(rs, args)


if __name__ == "__main__":
    raise SystemExit(main())
