import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock modules that might import pyarrow or be heavy
mock_no_filter = MagicMock()
mock_taxonomy = MagicMock()
mock_ranking = MagicMock()
mock_sync = MagicMock()
mock_te_null = MagicMock()

# Set up the mocks in sys.modules
sys.modules["scripts.no_filter_analysis"] = mock_no_filter
sys.modules["scripts.phenotype_taxonomy"] = mock_taxonomy
sys.modules["scripts.ranking_stability"] = mock_ranking
sys.modules["scripts.synchronous_ablation"] = mock_sync
sys.modules["scripts.te_null_analysis"] = mock_te_null

# Now import the module under test
from scripts.run_pr26_followups import main as run_pr26_main  # noqa: E402


def test_run_pr26_followups_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup directories
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "output"
    data_dir.mkdir()

    # Mock subprocess run for git commands and uv version
    mock_subprocess_run = MagicMock()
    mock_subprocess_run.return_value.stdout = "mock_output"
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    # Run the main function
    args = ["--data-dir", str(data_dir), "--out-dir", str(out_dir), "--quick"]

    # We need to make sure the mocked mains don't raise exceptions by default
    mock_no_filter.main.return_value = None
    mock_taxonomy.main.return_value = None
    mock_ranking.main.return_value = None
    mock_sync.main.return_value = None
    mock_te_null.main.return_value = None

    run_pr26_main(args)

    # Verify calls to analysis scripts
    mock_no_filter.main.assert_called_once()
    assert "--quick" in mock_no_filter.main.call_args[0][0]
    assert str(out_dir / "no_filter") in mock_no_filter.main.call_args[0][0]

    mock_sync.main.assert_called_once()
    assert "--quick" in mock_sync.main.call_args[0][0]
    assert str(out_dir / "synchronous_ablation") in mock_sync.main.call_args[0][0]

    mock_ranking.main.assert_called_once()
    assert "--quick" in mock_ranking.main.call_args[0][0]
    assert str(out_dir / "ranking_stability") in mock_ranking.main.call_args[0][0]

    mock_te_null.main.assert_called_once()
    assert "--quick" in mock_te_null.main.call_args[0][0]
    assert str(out_dir / "te_null") in mock_te_null.main.call_args[0][0]
    assert str(data_dir) in mock_te_null.main.call_args[0][0]

    mock_taxonomy.main.assert_called_once()
    assert "--quick" in mock_taxonomy.main.call_args[0][0]
    assert str(out_dir / "phenotypes") in mock_taxonomy.main.call_args[0][0]
    assert str(data_dir) in mock_taxonomy.main.call_args[0][0]

    # Verify manifest creation
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["commands"]["no_filter"] is not None
    assert manifest["analysis_status"]["no_filter"] == "success"
    assert manifest["outputs"]["no_filter"]["json"].endswith("summary.json")

    assert manifest["analysis_status"]["synchronous_ablation"] == "success"
    assert manifest["analysis_status"]["ranking_stability"] == "success"
    assert manifest["analysis_status"]["te_null"] == "success"
    assert manifest["analysis_status"]["phenotype_taxonomy"] == "success"
