from .cli import app

# Keep in sync with `version` in pyproject.toml. (The package is usually run
# from a source checkout without being pip-installed, so a hardcoded value is
# the single source of truth here; importlib.metadata would only resolve when
# the distribution is actually installed.)
__version__ = "0.6.0"
__all__ = ["app"]
