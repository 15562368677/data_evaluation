"""数据解析工具：parquet 转 mp4、parquet/hdf5 关节数据解析。"""

import io
import json
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.s3_client import download_s3_file, s3_object_exists, generate_presigned_url

# 关节名称
JOINT_NAMES = [
    "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "head_yaw_joint", "head_roll_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

HAND_JOINT_NAMES = [
    "L_pinky_proximal_joint", "L_ring_proximal_joint", "L_middle_proximal_joint",
    "L_index_proximal_joint", "L_thumb_proximal_pitch_joint", "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint", "R_ring_proximal_joint", "R_middle_proximal_joint",
    "R_index_proximal_joint", "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint",
]


def _detect_path_format(file_path: str) -> int:
    """检测路径格式类型：1=长路径(episode_xxx), 2=短路径(ULID/parquet), 3=短路径(ULID/top/rgb)"""
    if "episode_" in file_path and ".parquet" in file_path:
        return 1
    elif ".parquet" in file_path and "/" in file_path:
        return 2
    elif file_path.rstrip("/").endswith("/rgb") or file_path.rstrip("/").endswith("/top/rgb"):
        return 3
    # 默认尝试格式2
    return 2


def _extract_base_path(file_path: str) -> str:
    """从 file_path 提取基础路径（去掉文件名部分）。"""
    fmt = _detect_path_format(file_path)
    if fmt == 1:
        # factory/2026-02/.../episode_000000053/observation.images.camera_top.parquet
        # 基础路径: factory/2026-02/.../episode_000000053
        parts = file_path.rsplit("/", 1)
        return parts[0] if len(parts) > 1 else file_path
    elif fmt == 2:
        # 01KBRNK0WF7G21QR9T/observation.images.camera_top.parquet
        # 基础路径: factory/01KBRNK0WF7G21QR9T
        parts = file_path.rsplit("/", 1)
        base = parts[0] if len(parts) > 1 else file_path
        if not base.startswith("factory/"):
            base = f"factory/{base}"
        return base
    else:
        # 01K3J4DQTYEB9ZXZ0A/top/rgb
        # 基础路径: factory/01K3J4DQTYEB9ZXZ0A
        # 去掉 /top/rgb 后缀
        path = file_path.rstrip("/")
        for suffix in ["/top/rgb", "/top/depth"]:
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        if not path.startswith("factory/"):
            path = f"factory/{path}"
        return path


def resolve_video_path(file_path: str) -> dict:
    """
    根据 file_path 解析视频路径。
    返回 {"type": "mp4"|"parquet", "key": "s3_key"} 或 None。
    """
    fmt = _detect_path_format(file_path)
    base = _extract_base_path(file_path)

    if fmt == 3:
        # 格式3: 直接就是 mp4
        mp4_key = f"{base}/top/rgb/video.mp4"
        if s3_object_exists(mp4_key):
            return {"type": "mp4", "key": mp4_key}
        return None

    # 格式1和2: 先尝试 mp4，再回退 parquet
    mp4_key = f"{base}/top/rgb/video.mp4"
    if s3_object_exists(mp4_key):
        return {"type": "mp4", "key": mp4_key}

    # 回退到 parquet
    if fmt == 1:
        parquet_key = file_path
    else:
        parts = file_path.rsplit("/", 1)
        filename = parts[-1] if len(parts) > 1 else file_path
        parquet_key = f"{base}/{filename}"

    if s3_object_exists(parquet_key):
        return {"type": "parquet", "key": parquet_key}

    return None


def resolve_joint_paths(file_path: str) -> dict:
    """
    根据 file_path 解析关节数据路径。
    返回 {"type": "parquet"|"hdf5", "action_key": ..., "state_key": ...} 或 {"type": "hdf5", "key": ...}
    """
    fmt = _detect_path_format(file_path)
    base = _extract_base_path(file_path)

    if fmt == 3:
        # 格式3: hdf5
        hdf5_key = f"{base}/data.hdf5"
        if s3_object_exists(hdf5_key):
            return {"type": "hdf5", "key": hdf5_key}
        return None

    # 格式1和2: parquet
    action_key = f"{base}/action.parquet"
    state_key = f"{base}/observation.state.parquet"

    action_exists = s3_object_exists(action_key)
    state_exists = s3_object_exists(state_key)

    if action_exists or state_exists:
        result = {"type": "parquet"}
        if action_exists:
            result["action_key"] = action_key
        if state_exists:
            result["state_key"] = state_key
        return result

    # 回退到 hdf5
    hdf5_key = f"{base}/data.hdf5"
    if s3_object_exists(hdf5_key):
        return {"type": "hdf5", "key": hdf5_key}

    return None


def parquet_to_mp4(parquet_path: Path) -> Path | None:
    """将 parquet 中的图像数据转换为 mp4 视频文件。"""
    try:
        df = pd.read_parquet(parquet_path)
        if df.empty:
            return None

        # 找到图像列
        img_col = None
        for col in df.columns:
            if "camera_top" in col and col != "timestamp_utc":
                img_col = col
                break
        if img_col is None:
            return None

        # 解码第一帧获取尺寸
        first_img = Image.open(io.BytesIO(df.iloc[0][img_col]))
        w, h = first_img.size

        # 创建临时 mp4 文件
        mp4_path = parquet_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30  # 默认帧率
        writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))

        for i in range(len(df)):
            img_data = df.iloc[i][img_col]
            img = Image.open(io.BytesIO(img_data))
            frame = np.array(img)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()
        return mp4_path
    except Exception as e:
        print(f"[Parser] parquet 转 mp4 失败: {e}")
        return None


def get_video_url(file_path: str) -> str | None:
    """获取视频的可播放 URL（预签名 URL 或本地转换后的路径）。"""
    video_info = resolve_video_path(file_path)
    if video_info is None:
        return None

    if video_info["type"] == "mp4":
        return generate_presigned_url(video_info["key"])
    else:
        # parquet 需要下载并转换
        local_parquet = download_s3_file(video_info["key"])
        if local_parquet is None:
            return None
        mp4_path = parquet_to_mp4(local_parquet)
        if mp4_path is None:
            return None
        # 返回本地文件路径（后续通过 Flask 静态路由提供）
        return str(mp4_path)


def _parse_joint_column(df: pd.DataFrame, col_name: str) -> dict[str, list[float]]:
    """解析 parquet 中的关节列（action 或 observation.state），支持 list 和 ndarray。"""
    joints: dict[str, list[float]] = {}
    if col_name not in df.columns:
        return joints

    for i in range(len(df)):
        data = df.iloc[i][col_name]
        # data 可能是 list 或 numpy.ndarray，都是可迭代的 dict 序列
        if data is None:
            continue
        try:
            for item in data:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    name = item["name"]
                    if name not in joints:
                        joints[name] = []
                    joints[name].append(float(item["value"]))
        except Exception:
            continue

    return joints


def parse_parquet_joints(action_path: Path | None, state_path: Path | None) -> dict:
    """解析 parquet 格式的关节数据，返回 plotly 可用的数据结构。"""
    result = {"action": {}, "state": {}, "timestamps_action": [], "timestamps_state": [], "absolute_timestamps_action": [], "absolute_timestamps_state": []}

    if action_path and action_path.exists():
        try:
            df = pd.read_parquet(action_path)
            if "timestamp_utc" in df.columns:
                # 保留原始的绝对时间戳
                result["absolute_timestamps_action"] = df["timestamp_utc"].tolist()
                timestamps = pd.to_datetime(df["timestamp_utc"], errors="coerce")
                t0 = timestamps.iloc[0]
                result["timestamps_action"] = [(t - t0).total_seconds() for t in timestamps]

            result["action"] = _parse_joint_column(df, "action")
        except Exception as e:
            print(f"[Parser] 解析 action parquet 失败: {e}")

    if state_path and state_path.exists():
        try:
            df = pd.read_parquet(state_path)
            if "timestamp_utc" in df.columns:
                # 保留原始的绝对时间戳
                result["absolute_timestamps_state"] = df["timestamp_utc"].tolist()
                timestamps = pd.to_datetime(df["timestamp_utc"], errors="coerce")
                t0 = timestamps.iloc[0]
                result["timestamps_state"] = [(t - t0).total_seconds() for t in timestamps]

            result["state"] = _parse_joint_column(df, "observation.state")
        except Exception as e:
            print(f"[Parser] 解析 state parquet 失败: {e}")

    return result


def parse_hdf5_joints(hdf5_path: Path) -> dict:
    """解析 HDF5 格式的关节数据。"""
    import h5py

    result = {"action": {}, "state": {}, "timestamps_action": [], "timestamps_state": [], "absolute_timestamps_action": [], "absolute_timestamps_state": []}

    try:
        with h5py.File(hdf5_path, "r") as f:
            timestamps = f["timestamp"][:]
            
            # 保存原始的绝对时间戳
            result["absolute_timestamps_action"] = timestamps.tolist()
            result["absolute_timestamps_state"] = timestamps.tolist()
            
            t0 = timestamps[0]
            ts_list = [(t - t0) for t in timestamps]
            result["timestamps_action"] = ts_list
            result["timestamps_state"] = ts_list

            n = len(timestamps)

            # action/robot 和 state/robot
            if "action/robot" in f and "state/robot" in f:
                action_robot = f["action/robot"][:]
                state_robot = f["state/robot"][:]

                for j, name in enumerate(JOINT_NAMES):
                    if j < action_robot.shape[1]:
                        result["action"][name] = action_robot[:, j].tolist()
                    if j < state_robot.shape[1]:
                        result["state"][name] = state_robot[:, j].tolist()

            # action/hand 和 state/hand
            if "action/hand" in f and "state/hand" in f:
                action_hand = f["action/hand"][:]
                state_hand = f["state/hand"][:]

                for j, name in enumerate(HAND_JOINT_NAMES):
                    if j < action_hand.shape[1]:
                        result["action"][name] = action_hand[:, j].tolist()
                    if j < state_hand.shape[1]:
                        result["state"][name] = state_hand[:, j].tolist()

    except Exception as e:
        print(f"[Parser] 解析 HDF5 失败: {e}")

    return result


def load_joint_data(file_path: str) -> dict | None:
    """加载关节数据（自动判断 parquet/hdf5）。"""
    joint_info = resolve_joint_paths(file_path)
    if joint_info is None:
        return None

    if joint_info["type"] == "parquet":
        action_local = None
        state_local = None
        if "action_key" in joint_info:
            action_local = download_s3_file(joint_info["action_key"])
        if "state_key" in joint_info:
            state_local = download_s3_file(joint_info["state_key"])
        return parse_parquet_joints(action_local, state_local)
    else:
        hdf5_local = download_s3_file(joint_info["key"])
        if hdf5_local is None:
            return None
        return parse_hdf5_joints(hdf5_local)
