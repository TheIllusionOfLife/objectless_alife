"""Run all PR #26 follow-up analyses and emit a single manifest.

Usage:
    uv run python scripts/run_pr26_followups.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/pr26_followups
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from scripts.no_filter_analysis import main as run_no_filter
from scripts.phenotype_taxonomy import main as run_taxonomy
from scripts.pr26_followups_manifest_paths import collect_manifest_output_paths
from scripts.ranking_stability import main as run_ranking_stability
from scripts.synchronous_ablation import main as run_sync_ablation
from scripts.te_null_analysis import main as run_te_null


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run all PR #26 follow-up analyses")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/pr26_followups"))
    parser.add_argument("--quick", action="store_true", help="Run quick presets for all analyses")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    checksums_path = out_dir / "checksums.sha256"

    def _run_output(command: list[str]) -> str:
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    branch_name = _run_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    command_line = " ".join(
        [
            "uv",
            "run",
            "python",
            "scripts/run_pr26_followups.py",
            *[shlex.quote(arg) for arg in (argv or [])],
        ]
    )

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "quick": args.quick,
        "git_commit": "unknown",
        "git_branch": branch_name,
        "command_line": command_line,
        "python_version": sys.version.split(" ", maxsplit=1)[0],
        "uv_version": _run_output(["uv", "--version"]),
        "platform": platform.platform(),
        "commands": {},
        "outputs": {},
        "analysis_status": {},
        "zenodo": None,
    }

    def _write_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_checksums() -> None:
        out_dir_resolved = out_dir.resolve()
        targets: list[Path] = [manifest_path]
        output_targets, _skipped = collect_manifest_output_paths(
            manifest,
            manifest_path,
            base_dir=out_dir_resolved,
        )
        for resolved in output_targets:
            if resolved not in targets:
                targets.append(resolved)
        lines = [
            f"{_sha256(path)}  {path.resolve().relative_to(out_dir_resolved)}" for path in targets
        ]
        checksums_path.write_text("\n".join(lines) + "\n")

    def _run_analysis(
        key: str,
        script_name: str,
        func: Callable[[list[str]], None],
        target_dir: Path,
        extra_args: list[str] | None = None,
        custom_outputs: dict[str, str] | None = None,
    ) -> None:
        cmd_args = list(extra_args or []) + ["--out-dir", str(target_dir)]
        if args.quick:
            cmd_args.append("--quick")

        manifest["commands"][key] = [
            "uv",
            "run",
            "python",
            f"scripts/{script_name}.py",
            *cmd_args,
        ]

        try:
            func(cmd_args)
            manifest["analysis_status"][key] = "success"
        except Exception as exc:
            manifest["analysis_status"][key] = f"failed: {exc}"

        if custom_outputs:
            manifest["outputs"][key] = custom_outputs
        else:
            manifest["outputs"][key] = {
                "json": str(target_dir / "summary.json"),
                "csv": str(target_dir / "summary.csv"),
            }
        _write_manifest()

    _run_analysis(
        "no_filter",
        "no_filter_analysis",
        run_no_filter,
        out_dir / "no_filter",
    )

    _run_analysis(
        "synchronous_ablation",
        "synchronous_ablation",
        run_sync_ablation,
        out_dir / "synchronous_ablation",
    )

    _run_analysis(
        "ranking_stability",
        "ranking_stability",
        run_ranking_stability,
        out_dir / "ranking_stability",
    )

    _run_analysis(
        "te_null",
        "te_null_analysis",
        run_te_null,
        out_dir / "te_null",
        extra_args=["--data-dir", str(args.data_dir)],
    )

    taxonomy_dir = out_dir / "phenotypes"
    _run_analysis(
        "phenotype_taxonomy",
        "phenotype_taxonomy",
        run_taxonomy,
        taxonomy_dir,
        extra_args=["--data-dir", str(args.data_dir)],
        custom_outputs={
            "json": str(taxonomy_dir / "taxonomy.json"),
            "csv": str(taxonomy_dir / "taxonomy.csv"),
        },
    )

    git_commit = _run_output(["git", "rev-parse", "HEAD"])

    manifest["git_commit"] = git_commit
    _write_manifest()
    _write_checksums()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
