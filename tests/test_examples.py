"""The sample datafiles and scripts installed with the package."""

import os

import pytest

from pysurfaceevolver import Evolver, examples, make_datafile


def test_samples_are_installed():
    names = examples.names()
    assert "cube.fe" in names and len(names) == 25
    assert "obj.cmd" in examples.names("*.cmd")
    assert os.path.isfile(examples.path("cube.fe"))
    with pytest.raises(FileNotFoundError):
        examples.path("nope.fe")


def test_samples_are_on_evolverpath():
    entries = os.environ["EVOLVERPATH"].split(os.pathsep)
    assert str(examples.directory()) in entries


def test_load_sample_by_name_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ev = Evolver("cube.fe")
    assert ev.counts["facets"] == 24
    ev = Evolver("cube")          # Evolver adds .fe itself
    assert ev.counts["facets"] == 24


def test_working_directory_comes_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cube.fe").write_text(
        make_datafile([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]))
    assert Evolver("cube.fe").counts["facets"] == 1


def test_sample_scripts_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ev = Evolver("cube.fe")
    ev.iterate(3)
    ev.command('read "obj.cmd"')
    ev.command(f'obj >>> "{tmp_path / "cube.obj"}"')
    lines = (tmp_path / "cube.obj").read_text().splitlines()
    assert sum(line.startswith("v ") for line in lines) >= ev.counts["vertices"]
    assert any(line.startswith("f ") for line in lines)
