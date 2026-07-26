from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog.stdlib

from paperbench.utils import get_paperbench_data_dir, load_yaml_dict

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    paper_pdf: Path
    paper_md: Path
    addendum: Path
    judge_addendum: Path
    assets: Path
    blacklist: Path
    rubric: Path

    def __post_init__(self) -> None:
        assert isinstance(self.id, str), "Paper id must be a string."
        assert isinstance(self.title, str), "Paper title must be a string."
        assert isinstance(self.paper_pdf, Path), "Paper PDF must be a Path."
        assert isinstance(self.paper_md, Path), "Paper MD must be a Path."
        assert isinstance(self.addendum, Path), "Addendum must be a Path."
        assert isinstance(self.judge_addendum, Path), "Judge addendum must be a Path."
        assert isinstance(self.assets, Path), "Assets must be a Path."
        assert isinstance(self.rubric, Path), "Rubric must be a Path."
        assert isinstance(self.blacklist, Path), "Blacklist must be a Path."
        assert len(self.id) > 0, "Paper id cannot be empty."
        assert len(self.title) > 0, "Paper title cannot be empty."

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Paper:
        try:
            return Paper(
                id=data["id"],
                title=data["title"],
                paper_pdf=data["paper_pdf"],
                paper_md=data["paper_md"],
                addendum=data["addendum"],
                judge_addendum=data["judge_addendum"],
                assets=data["assets"],
                rubric=data["rubric"],
                blacklist=data["blacklist"],
            )
        except KeyError as e:
            raise ValueError(f"Missing key in paper config! {e}") from e


class PaperRegistry:
    def get_paper(self, paper_id: str) -> Paper:
        """Fetch the paper from the registry."""

        paper_dir = self.get_paper_dir(paper_id)
        config_path = paper_dir / "config.yaml"
        config = load_yaml_dict(config_path)

        paper_pdf = paper_dir / "paper.pdf"
        paper_md = paper_dir / "paper.md"
        addendum = paper_dir / "addendum.md"
        judge_addendum = paper_dir / "judge.addendum.md"
        assets = paper_dir / "assets"
        rubric = paper_dir / "rubric.json"
        blacklist = paper_dir / "blacklist.txt"
        return Paper.from_dict(
            {
                **config,
                "paper_pdf": paper_pdf,
                "paper_md": paper_md,
                "addendum": addendum,
                "judge_addendum": judge_addendum,
                "assets": assets,
                "rubric": rubric,
                "blacklist": blacklist,
            }
        )

    def get_paper_dir(self, paper_id: str) -> Path:
        """Return the unique directory containing a registered paper."""

        matches = [
            registry_dir / paper_id
            for registry_dir in self.get_registry_dirs()
            if (registry_dir / paper_id / "config.yaml").is_file()
        ]

        if not matches:
            registry_dirs = ", ".join(str(path) for path in self.get_registry_dirs())
            raise ValueError(
                f"Paper '{paper_id}' was not found in registry directories: {registry_dirs}"
            )
        if len(matches) > 1:
            locations = ", ".join(str(path) for path in matches)
            raise ValueError(f"Paper ID '{paper_id}' is registered more than once: {locations}")

        return matches[0]

    def get_papers_dir(self) -> Path:
        """Return the directory containing PaperBench's built-in papers."""

        return get_paperbench_data_dir() / "papers"

    def get_nips26_rebuttal_dir(self) -> Path:
        """Return the directory containing the NeurIPS 2026 rebuttal papers."""

        return get_paperbench_data_dir() / "nips26_rebuttal"

    def get_registry_dirs(self) -> tuple[Path, ...]:
        """Return paper registry roots in built-in-first lookup order."""

        return self.get_papers_dir(), self.get_nips26_rebuttal_dir()

    def list_paper_ids(self) -> list[str]:
        """List all paper IDs available in the registry, sorted alphabetically."""

        paper_locations: dict[str, Path] = {}
        for registry_dir in self.get_registry_dirs():
            for config_path in registry_dir.glob("*/config.yaml"):
                paper_id = config_path.parent.name
                if paper_id in paper_locations:
                    first_location = paper_locations[paper_id].parent
                    raise ValueError(
                        f"Paper ID '{paper_id}' is registered more than once: "
                        f"{first_location}, {config_path.parent}"
                    )
                paper_locations[paper_id] = config_path

        return sorted(paper_locations)


paper_registry = PaperRegistry()
