# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Pytest configuration for Fly DSL tests.

Supports both the new Fly dialect (build-fly/) and legacy build paths.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[1]

# New Fly dialect build
_fly_pkg_dir = _repo_root / "build-fly" / "python_packages"
if _fly_pkg_dir.exists():
    _p = str(_fly_pkg_dir)
    _already = _p in sys.path or any(os.path.isdir(ep) and os.path.samefile(ep, _p) for ep in sys.path if ep)
    if not _already:
        sys.path.insert(0, _p)

# Legacy: .flydsl/build or build/
for _legacy in [
    _repo_root / ".flydsl" / "build" / "python_packages" / "flydsl",
    _repo_root / "build" / "python_packages" / "flydsl",
    _repo_root / "build" / "lib.linux-x86_64-cpython-312",
]:
    if _legacy.exists():
        _p = str(_legacy)
        if _p not in sys.path:
            sys.path.append(_p)
        break

# Legacy: in-tree flydsl source (for old API tests)
_src_py_dir = _repo_root / "flydsl" / "src"
if _src_py_dir.exists() and (_src_py_dir / "flydsl").exists():
    _p = str(_src_py_dir)
    if _p not in sys.path:
        sys.path.append(_p)


def _flydsl_origin():
    """Directory `import flydsl` resolves to, or None if it is not importable.

    None of the paths above point at `python/`, so an unbuilt checkout silently
    falls back to whatever `flydsl` is installed system-wide. `pip install -e .`
    is what links this checkout in (it symlinks `python/flydsl/_mlir` at the
    build output); without it the suite exercises a foreign package.
    """
    try:
        spec = importlib.util.find_spec("flydsl")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _flydsl_is_in_repo(origin):
    return origin is not None and _repo_root in origin.parents


# Try importing new or old context setup
_ensure_extensions = None
try:
    from flydsl.compiler.context import ensure_flydsl_python_extensions

    _ensure_extensions = ensure_flydsl_python_extensions
except ImportError:
    pass

try:
    from flydsl._mlir.ir import Context, InsertionPoint, Location, Module
except ImportError:
    try:
        from _mlir.ir import Context, InsertionPoint, Location, Module
    except ImportError:
        Context = Location = Module = InsertionPoint = None


@pytest.fixture
def ctx():
    """Provide a fresh MLIR context for each test."""
    if Context is None:
        pytest.skip("MLIR Python bindings not available")
    with Context() as context:
        if _ensure_extensions:
            _ensure_extensions(context)
        with Location.unknown(context):
            module = Module.create()
            yield type(
                "MLIRContext",
                (),
                {
                    "context": context,
                    "module": module,
                    "location": Location.unknown(context),
                },
            )()


@pytest.fixture
def module(ctx):
    """Provide module from context."""
    return ctx.module


@pytest.fixture
def insert_point(ctx):
    """Provide insertion point for the module body."""
    with InsertionPoint(ctx.module.body):
        yield InsertionPoint.current


def pytest_addoption(parser):
    """Add FlyDSL test-session options that map to env variables."""
    group = parser.getgroup("flydsl")
    group.addoption(
        "--flydsl-compile-backend",
        action="store",
        default=None,
        help="Set FLYDSL_COMPILE_BACKEND for this pytest session.",
    )
    group.addoption(
        "--flydsl-compile-arch",
        action="store",
        default=None,
        help="Set ARCH for this pytest session.",
    )


def pytest_report_header(config):
    """Report which flydsl the suite is about to exercise."""
    origin = _flydsl_origin()
    if origin is None:
        return "flydsl: not importable"
    if _flydsl_is_in_repo(origin):
        return f"flydsl: {origin}"
    return f"flydsl: {origin} (outside this checkout)"


def pytest_configure(config):
    """Apply FlyDSL env overrides from CLI options.

    Note: marker registration lives in pytest.ini (single source of truth).
    """
    origin = _flydsl_origin()
    if origin is not None and not _flydsl_is_in_repo(origin):
        # The header is suppressed under -q, so warn as well: a run against a
        # foreign flydsl reports results that have nothing to do with the
        # working tree, which is far more confusing than an import error.
        config.issue_config_time_warning(
            pytest.PytestConfigWarning(
                f"`import flydsl` resolves to {origin}, outside {_repo_root}. "
                f"This run does not exercise your working tree. "
                f"Run `pip install -e .` in this checkout, or set PYTHONPATH to its `python/` directory."
            ),
            stacklevel=2,
        )

    backend = config.getoption("--flydsl-compile-backend")
    arch = config.getoption("--flydsl-compile-arch")
    # Intentionally set process-level env vars so downstream code (env.py)
    # picks them up. The pytest process exits after the session, so no cleanup needed.
    if backend:
        os.environ["FLYDSL_COMPILE_BACKEND"] = backend
    if arch:
        os.environ["ARCH"] = arch


def pytest_sessionfinish(session, exitstatus):
    """Prevent pytest from erroring on empty test files."""
    if exitstatus == 5:
        session.exitstatus = 0
