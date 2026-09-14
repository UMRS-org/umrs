"""
Tests for Phase 5 — Robustness & Polish.

Covers:
  - Thread-safe cached step imports (_step_class_cache)
  - importlib.util loading without sys.path mutation
  - Import error handling (missing module, bad class, non-Step class)
  - Cache invalidation across different script dirs
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from umrs.pipeline_sdk.step import Step
from umrs.pipeline_sdk.workflow import (
    ScriptLoadError,
    StepSpec,
    WorkflowDefinition,
    _import_step_class,
    _step_cache_lock,
    _step_class_cache,
    load_steps,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def _clear_import_cache():
    """Clear the step import cache before each test."""
    with _step_cache_lock:
        _step_class_cache.clear()
    yield
    with _step_cache_lock:
        _step_class_cache.clear()


def _write_step_module(tmp_path: Path, module_name: str, class_name: str, extra: str = "") -> Path:
    """Write a minimal Step subclass into tmp_path/<module_name>.py."""
    code = textwrap.dedent(f"""\
        from umrs.pipeline_sdk.step import Step, StepContext, StepResult

        class {class_name}(Step):
            name = "{class_name.lower()}"
            {extra}

            def run(self, ctx: StepContext) -> StepResult:
                return StepResult(output_keys=list(ctx.input_keys))
    """)
    filepath = tmp_path / f"{module_name}.py"
    filepath.write_text(code)
    return filepath


# =========================================================================
# importlib-based loading
# =========================================================================


class TestImportlibLoading:
    def test_loads_step_from_file(self, tmp_path: Path):
        _write_step_module(tmp_path, "my_mod", "MyStep")
        spec = StepSpec(name="s1", entrypoint="my_mod:MyStep", config={})
        cls = _import_step_class(spec, tmp_path)
        assert issubclass(cls, Step)
        assert cls.name == "mystep"

    def test_no_sys_path_mutation(self, tmp_path: Path):
        import sys

        _write_step_module(tmp_path, "no_path_mod", "NoPath")
        spec = StepSpec(name="s1", entrypoint="no_path_mod:NoPath", config={})
        before = list(sys.path)
        _import_step_class(spec, tmp_path)
        after = list(sys.path)
        # sys.path should not have script_dir appended/prepended
        assert str(tmp_path) not in after or str(tmp_path) in before

    def test_missing_module_raises(self, tmp_path: Path):
        spec = StepSpec(name="s1", entrypoint="nonexistent:Cls", config={})
        with pytest.raises(ScriptLoadError, match="cannot import"):
            _import_step_class(spec, tmp_path)

    def test_missing_class_raises(self, tmp_path: Path):
        _write_step_module(tmp_path, "modA", "Actual")
        spec = StepSpec(name="s1", entrypoint="modA:Missing", config={})
        with pytest.raises(ScriptLoadError, match="not found"):
            _import_step_class(spec, tmp_path)

    def test_non_step_class_raises(self, tmp_path: Path):
        code = textwrap.dedent("""\
            class NotAStep:
                pass
        """)
        (tmp_path / "bad_mod.py").write_text(code)
        spec = StepSpec(name="s1", entrypoint="bad_mod:NotAStep", config={})
        with pytest.raises(ScriptLoadError, match="does not inherit"):
            _import_step_class(spec, tmp_path)

    def test_invalid_entrypoint_format(self, tmp_path: Path):
        spec = StepSpec(name="s1", entrypoint="no_colon_here", config={})
        with pytest.raises(ScriptLoadError, match="invalid entrypoint"):
            _import_step_class(spec, tmp_path)


# =========================================================================
# Import cache
# =========================================================================


class TestImportCache:
    def test_cached_after_first_import(self, tmp_path: Path):
        _write_step_module(tmp_path, "cache_mod", "CacheStep")
        spec = StepSpec(name="s1", entrypoint="cache_mod:CacheStep", config={})
        cls1 = _import_step_class(spec, tmp_path)
        cls2 = _import_step_class(spec, tmp_path)
        assert cls1 is cls2

    def test_cache_key_includes_dir(self, tmp_path: Path):
        """Same module_name in different dirs → different cache entries."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_step_module(dir_a, "shared", "StepA")
        _write_step_module(dir_b, "shared", "StepB")
        spec_a = StepSpec(name="sa", entrypoint="shared:StepA", config={})
        spec_b = StepSpec(name="sb", entrypoint="shared:StepB", config={})
        cls_a = _import_step_class(spec_a, dir_a)
        cls_b = _import_step_class(spec_b, dir_b)
        assert cls_a is not cls_b
        assert cls_a.name == "stepa"
        assert cls_b.name == "stepb"

    def test_cache_populated(self, tmp_path: Path):
        _write_step_module(tmp_path, "pop_mod", "PopStep")
        spec = StepSpec(name="s1", entrypoint="pop_mod:PopStep", config={})
        _import_step_class(spec, tmp_path)
        assert any(
            key[0] == "pop_mod:PopStep" and key[1] == str(tmp_path) for key in _step_class_cache
        )


# =========================================================================
# load_steps with new importer
# =========================================================================


class TestLoadStepsIntegration:
    def test_load_steps_from_file(self, tmp_path: Path):
        _write_step_module(tmp_path, "ls_mod", "LsStep")
        wf_dict = {
            "name": "test",
            "source": {"type": "local_path", "path": str(tmp_path)},
            "steps": [
                {"name": "s1", "entrypoint": "ls_mod:LsStep", "config": {}},
            ],
        }
        wf = WorkflowDefinition.from_dict(wf_dict)
        steps = load_steps(wf)
        assert len(steps) == 1
        assert isinstance(steps[0], Step)

    def test_load_steps_caches(self, tmp_path: Path):
        _write_step_module(tmp_path, "lsc_mod", "LscStep")
        wf_dict = {
            "name": "test",
            "source": {"type": "local_path", "path": str(tmp_path)},
            "steps": [
                {"name": "s1", "entrypoint": "lsc_mod:LscStep"},
                {"name": "s2", "entrypoint": "lsc_mod:LscStep"},
            ],
        }
        wf = WorkflowDefinition.from_dict(wf_dict)
        steps = load_steps(wf)
        # Both steps should be different instances but same class
        assert type(steps[0]) is type(steps[1])
