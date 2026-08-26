import asyncio

import pytest

import aiolab
from aiolab import cli


def test_every_demo_covers_one_module():
    assert set(cli.DEMOS) == {
        "basics",
        "races",
        "tasks",
        "queue",
        "streams",
        "retry",
        "ratelimit",
    }
    assert "all" not in cli.DEMOS  # "all" is a parser choice, not a demo


@pytest.mark.parametrize("name", sorted(cli.DEMOS))
def test_each_demo_exits_zero(name, capsys):
    assert cli.main([name]) == 0
    assert capsys.readouterr().out.strip()


def test_unknown_demo_is_rejected():
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2


def test_default_runs_everything(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in ("sequential", "asyncio.Lock", "siblings cancelled", "attempts"):
        assert name in out


def test_public_api_is_importable():
    for name in aiolab.__all__:
        assert hasattr(aiolab, name), name


async def test_the_pipeline_example_runs_end_to_end():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "examples" / "pipeline_demo.py"
    spec = importlib.util.spec_from_file_location("pipeline_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert await module.main() == 0
