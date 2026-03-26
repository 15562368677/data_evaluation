"""兼容旧路径的 PnP 检测导出。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.acceptance_service.validators.acceptance.ee_action import (
    EEActionValidator,
    calculate_closure_degree,
    calculate_closure_metrics_from_dataframe,
    calculate_closure_velocity,
)
from src.acceptance_service.validators.core.base import ValidatorConfig

_DEFAULT_CONFIG = ValidatorConfig()

HAND_CONFIG_BASE = {
    "right": {
        "right_hand_fingers": _DEFAULT_CONFIG.get_hand_fingers("right"),
        "joint_direction_coefficients": _DEFAULT_CONFIG.get_hand_direction_coefficients("right"),
    },
    "left": {
        "right_hand_fingers": _DEFAULT_CONFIG.get_hand_fingers("left"),
        "joint_direction_coefficients": _DEFAULT_CONFIG.get_hand_direction_coefficients("left"),
    },
}


def pick_identify(
    closure_degrees: np.ndarray,
    closure_velocities: np.ndarray,
    state_action_diffs: Dict[str, np.ndarray],
    config: Dict[str, Any],
    state_df: pd.DataFrame = None,
    action_df: pd.DataFrame = None,
) -> List[Tuple[int, int]]:
    """兼容旧入口，内部转发到 EEActionValidator。"""

    hand_config = dict(config)
    if "finger_joints" not in hand_config:
        hand_config["finger_joints"] = list(
            hand_config.get("right_hand_fingers") or hand_config.get("finger_joints") or []
        )
    if "joint_direction_coefficients" not in hand_config:
        hand_name = str(hand_config.get("hand") or "right")
        hand_config["joint_direction_coefficients"] = _DEFAULT_CONFIG.get_hand_direction_coefficients(
            hand_name
        )
    hand_config.setdefault("hand", str(hand_config.get("hand") or "right"))

    return EEActionValidator.detect_pick_segments(
        closure_degrees=closure_degrees,
        closure_velocities=closure_velocities,
        state_action_diffs=state_action_diffs,
        hand_config=hand_config,
        state_df=state_df,
        action_df=action_df,
    )


__all__ = [
    "HAND_CONFIG_BASE",
    "calculate_closure_degree",
    "calculate_closure_metrics_from_dataframe",
    "calculate_closure_velocity",
    "pick_identify",
]
