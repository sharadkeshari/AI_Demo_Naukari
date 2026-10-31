from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


class TransientToolError(Exception):
    """A declared temporary failure that may be retried."""


class PermanentToolError(Exception):
    """A permanent failure that must not be retried."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: object
    tool_call_id: str


@dataclass(frozen=True)
class ToolSpec:
    handler: Callable[..., Any]
    parameters: Mapping[str, type]
    description: str = ""
    transient_errors: tuple[type[Exception], ...] = (TransientToolError,)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error_code: str | None = None
    message: str | None = None

    @classmethod
    def valid(cls) -> "ValidationResult":
        return cls(ok=True)

    @classmethod
    def invalid(cls, error_code: str, message: str) -> "ValidationResult":
        return cls(ok=False, error_code=error_code, message=message)


@dataclass(frozen=True)
class ExecutionOutcome:
    ok: bool
    value: Any = None
    error_code: str | None = None
    message: str | None = None
    attempts: int = 0

    @classmethod
    def succeeded(cls, value: Any, attempts: int) -> "ExecutionOutcome":
        return cls(ok=True, value=value, attempts=attempts)

    @classmethod
    def failed(
        cls,
        error_code: str,
        message: str,
        attempts: int,
    ) -> "ExecutionOutcome":
        return cls(
            ok=False,
            error_code=error_code,
            message=message,
            attempts=attempts,
        )


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    ok: bool
    output: Any = None
    error_code: str | None = None
    message: str | None = None
    attempts: int = 0


def validate_arguments(
    arguments: object,
    parameters: Mapping[str, type],
) -> ValidationResult:
    """Validate an argument mapping in a deterministic order."""
    if not isinstance(arguments, Mapping):
        return ValidationResult.invalid("invalid_arguments", "Arguments must be a mapping.")

    expected_names = set(parameters)
    provided_names = set(arguments)
    missing = sorted(expected_names - provided_names)
    if missing:
        return ValidationResult.invalid("missing_argument", f"Missing argument: {missing[0]}")
    unexpected = sorted(provided_names - expected_names)
    if unexpected:
        return ValidationResult.invalid(
            "unexpected_argument",
            f"Unexpected argument: {unexpected[0]}",
        )
    for name in sorted(expected_names):
        if type(arguments[name]) is not parameters[name]:
            return ValidationResult.invalid(
                "invalid_argument_type",
                f"Invalid type for argument: {name}",
            )
    return ValidationResult.valid()
