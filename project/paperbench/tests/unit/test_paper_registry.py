from pathlib import Path

import pytest

from paperbench.paper_registry import PaperRegistry


def _add_paper(data_dir: Path, registry_name: str, paper_id: str) -> Path:
    paper_dir = data_dir / registry_name / paper_id
    paper_dir.mkdir(parents=True)
    (paper_dir / "config.yaml").write_text(f"id: {paper_id}\ntitle: Test paper\n")
    return paper_dir


def test_registry_discovers_built_in_and_nips26_rebuttal_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_in_dir = _add_paper(tmp_path, "papers", "built-in-paper")
    rebuttal_paper_dir = _add_paper(tmp_path, "nips26_rebuttal", "rebuttal-paper")
    monkeypatch.setattr("paperbench.paper_registry.get_paperbench_data_dir", lambda: tmp_path)
    registry = PaperRegistry()

    assert registry.list_paper_ids() == ["built-in-paper", "rebuttal-paper"]
    assert registry.get_paper_dir("built-in-paper") == built_in_dir
    assert registry.get_paper_dir("rebuttal-paper") == rebuttal_paper_dir
    assert (
        registry.get_paper("rebuttal-paper").rubric == rebuttal_paper_dir / "rubric.json"
    )


def test_registry_rejects_duplicate_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_paper(tmp_path, "papers", "duplicate-paper")
    _add_paper(tmp_path, "nips26_rebuttal", "duplicate-paper")
    monkeypatch.setattr("paperbench.paper_registry.get_paperbench_data_dir", lambda: tmp_path)
    registry = PaperRegistry()

    with pytest.raises(ValueError, match="registered more than once"):
        registry.list_paper_ids()

    with pytest.raises(ValueError, match="registered more than once"):
        registry.get_paper("duplicate-paper")


def test_registry_reports_unknown_paper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("paperbench.paper_registry.get_paperbench_data_dir", lambda: tmp_path)

    with pytest.raises(ValueError, match="was not found"):
        PaperRegistry().get_paper("unknown-paper")
