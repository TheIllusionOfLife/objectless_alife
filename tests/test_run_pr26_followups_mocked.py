import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Define mocks as fixtures or inside the test to avoid global pollution
@pytest.fixture
def mock_modules():
    mock_no_filter = MagicMock()
    mock_taxonomy = MagicMock()
    mock_ranking = MagicMock()
    mock_sync = MagicMock()
    mock_te_null = MagicMock()

    # We mock modules that are imported by scripts.run_pr26_followups
    modules = {
        "scripts.no_filter_analysis": mock_no_filter,
        "scripts.phenotype_taxonomy": mock_taxonomy,
        "scripts.ranking_stability": mock_ranking,
        "scripts.synchronous_ablation": mock_sync,
        "scripts.te_null_analysis": mock_te_null,
    }
    return modules, (mock_no_filter, mock_taxonomy, mock_ranking, mock_sync, mock_te_null)


def test_run_pr26_followups_mocked(
    tmp_path: Path, mock_modules, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules_dict, (mock_no_filter, mock_taxonomy, mock_ranking, mock_sync, mock_te_null) = (
        mock_modules
    )

    # Setup directories
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "output"
    data_dir.mkdir()

    # Mock subprocess run for git commands and uv version
    mock_subprocess_run = MagicMock()
    mock_subprocess_run.return_value.stdout = "mock_output"
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    # Use patch.dict to safely mock sys.modules for the duration of this context
    with patch.dict(sys.modules, modules_dict):
        # Import the module inside the patch context so it picks up the mocked modules
        # If it was already imported, we must reload it or remove it first to force re-import
        if "scripts.run_pr26_followups" in sys.modules:
            del sys.modules["scripts.run_pr26_followups"]

        from scripts.run_pr26_followups import main as run_pr26_main

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
