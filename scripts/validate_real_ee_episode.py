#!/usr/bin/env python3
"""Validate one real episode with EEActionValidator using only episode_id."""
import sys

REPO_ROOT = "/home/shitw1/Desktop/haishang/evaluation"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.acceptance_service.validators import EEActionValidator, ValidatorConfig

DEFAULT_EPISODE_ID = "1049211"


def main() -> None:
    episode_id = DEFAULT_EPISODE_ID

    config = ValidatorConfig()
    validator = EEActionValidator(config)
    result = validator.validate(episode_id)

    for key, value in result.to_dict().items():
        if key == "issues":
            print(f"{key}:")
            for issue in value:
                print(f"  - {issue}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
