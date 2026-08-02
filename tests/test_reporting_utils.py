from pathlib import Path

from examples.reporting_utils import format_dataframe_for_report, write_run_report


def test_write_run_report_creates_report_file(tmp_path):
    report_path = write_run_report(
        script_name="workflow_dual_horizon_pro2.py",
        experiment_name="Dual_Horizon_1D_Short_Term",
        summary_lines=["summary line"],
        metrics={"IC": 0.12},
        details_lines=["prediction_preview:", "score", "latest_recommendations:", "instrument score"],
        output_dir=tmp_path,
    )

    assert isinstance(report_path, Path)
    assert report_path.exists()
    assert report_path.parent == tmp_path
    content = report_path.read_text(encoding="utf-8")
    assert "workflow_dual_horizon_pro2.py" in content
    assert "Dual_Horizon_1D_Short_Term" in content
    assert "summary line" in content
    assert "IC: 0.12" in content
    assert "prediction_preview:" in content
    assert "latest_recommendations:" in content


def test_format_dataframe_for_report_uses_flat_table():
    frame = {"datetime": ["2025-07-01"], "instrument": ["SH600105"], "score": [0.95]}
    formatted = format_dataframe_for_report(frame)
    assert "datetime" in formatted
    assert "instrument" in formatted
    assert "score" in formatted
