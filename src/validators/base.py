"""兼容旧路径的验证器基类导出。"""

from src.acceptance_service.validators.core.base import (
    BaseValidator,
    IssueLevel,
    ValidationIssue,
    ValidationResult,
    ValidatorConfig,
)

__all__ = [
    "BaseValidator",
    "IssueLevel",
    "ValidationIssue",
    "ValidationResult",
    "ValidatorConfig",
]
