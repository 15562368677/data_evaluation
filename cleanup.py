import re

with open('src/engines/pnp_detector/data_detector.py', 'r') as f:
    content = f.read()

# Remove load_config_from_yaml
content = re.sub(r'def load_config_from_yaml.*?^def print_config', 'def print_config', content, flags=re.MULTILINE|re.DOTALL)

# Remove print_config
content = re.sub(r'def print_config.*?^def load_joint_data_from_parquet', 'def load_joint_data_from_parquet', content, flags=re.MULTILINE|re.DOTALL)

# Remove load_joint_data_from_parquet
content = re.sub(r'def load_joint_data_from_parquet.*?^def load_joint_data_from_hdf5', 'def load_joint_data_from_hdf5', content, flags=re.MULTILINE|re.DOTALL)

# Remove load_joint_data_from_hdf5
content = re.sub(r'def load_joint_data_from_hdf5.*?^from pathlib import Path', 'from pathlib import Path', content, flags=re.MULTILINE|re.DOTALL)

# Remove compute_state_action_diffs
content = re.sub(r'def compute_state_action_diffs.*?^def calculate_closure_degree', 'def calculate_closure_degree', content, flags=re.MULTILINE|re.DOTALL)

# Remove check_sustained_joint_differences
content = re.sub(r'def check_sustained_joint_differences.*?^def pick_identify', 'def pick_identify', content, flags=re.MULTILINE|re.DOTALL)

# Remove thresholds from pick_identify
content = content.replace('    thresholds: dict,\n', '')
content = content.replace('    thresholds: dict,', '')

# Remove redundant import logs
imports = """from pathlib import Path
from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Hand configurations
HAND_CONFIG_BASE = {
    'right': {
        'right_hand_fingers': [
            'R_pinky_proximal_joint', 'R_ring_proximal_joint',
            'R_middle_proximal_joint', 'R_index_proximal_joint',
            'R_thumb_proximal_pitch_joint'
        ],
        'additional_joints': ['right_elbow_pitch_joint'],
        'joint_direction_coefficients': {
            'R_pinky_proximal_joint': -1.0,
            'R_ring_proximal_joint': -1.0,
            'R_middle_proximal_joint': -1.0,
            'R_index_proximal_joint': -1.0,
            'R_thumb_proximal_pitch_joint': 1.0,
        },
    },
    'left': {
        'right_hand_fingers': [
            'L_pinky_proximal_joint', 'L_ring_proximal_joint',
            'L_middle_proximal_joint', 'L_index_proximal_joint',
            'L_thumb_proximal_pitch_joint'
        ],
        'additional_joints': ['left_elbow_pitch_joint'],
        'joint_direction_coefficients': {
            'L_pinky_proximal_joint': -1.0,
            'L_ring_proximal_joint': -1.0,
            'L_middle_proximal_joint': -1.0,
            'L_index_proximal_joint': -1.0,
            'L_thumb_proximal_pitch_joint': 1.0,
        },
    }
}
"""
content = re.sub(r'from pathlib import Path.*logger = logging\.getLogger\(__name__\)\n\n# No external local module imports needed\n', '', content, flags=re.MULTILINE|re.DOTALL)

# Replace the first imports with new ones
content = re.sub(r'from pathlib import Path.*?logger = logging\.getLogger\(__name__\)', imports, content, flags=re.MULTILINE|re.DOTALL)

with open('src/engines/pnp_detector/data_detector.py', 'w') as f:
    f.write(content)
