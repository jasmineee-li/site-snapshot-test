"""Registry for eval awareness judges."""

from eval_awareness_experiments.judges.base import BaseJudge

_JUDGE_REGISTRY: dict[str, type[BaseJudge]] = {}


def register_judge(name: str | None = None):
    """Decorator to register a judge class."""

    def decorator(cls: type[BaseJudge]) -> type[BaseJudge]:
        judge_name = name or cls.name
        _JUDGE_REGISTRY[judge_name] = cls
        return cls

    return decorator


def get_judge(name: str, **kwargs) -> BaseJudge:
    """Get a judge instance by name."""
    if name not in _JUDGE_REGISTRY:
        available = ", ".join(_JUDGE_REGISTRY.keys())
        raise ValueError(f"Unknown judge: {name}. Available: {available}")
    return _JUDGE_REGISTRY[name](**kwargs)


def list_judges() -> list[str]:
    """List all registered judge names."""
    return list(_JUDGE_REGISTRY.keys())
