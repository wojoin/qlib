from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd
from termcolor import colored


def print_colored_block(message: str, color: str = "red", attrs=None) -> None:
    attrs = attrs or ["bold"]
    print(colored(message, color, attrs=attrs))


def format_dataframe_for_report(frame: pd.DataFrame | dict) -> str:
    """Render a DataFrame as a flat table suitable for text reports."""
    if isinstance(frame, dict):
        frame = pd.DataFrame(frame)
    if getattr(frame, "empty", False):
        return "(empty)"
    return frame.reset_index().to_string(index=False)


def print_latest_recommendations(pred, name: str, topk: int = 5):
    """Print the latest recommendations for a given prediction frame."""
    latest_date = pred.index.get_level_values("datetime").max()
    latest_pred = (
        pred.xs(latest_date, level="datetime")
        .sort_values("score", ascending=False)
        .head(topk)
    )

    print_colored_block(f"\n[{name}] 最新行业推荐 ({latest_date.date()}):")
    for rank, (instrument, row) in enumerate(latest_pred.iterrows(), start=1):
        print(colored(f"{rank}. {instrument}", "red") + colored(f"  score={row['score']:.6f}", "red"))
    return latest_pred


def print_final_results(recorder, name: str, stage: str = "final", metrics_filter=None):
    """Print recorder metrics and optionally the backtest summary table."""
    print_colored_block(f"\n[{name}] {stage} 结果")
    metrics = recorder.list_metrics()
    filters = metrics_filter or ["IC", "annualized_return", "information_ratio", "max_drawdown", "l2.", "ffr"]
    if metrics:
        for key in sorted(metrics):
            value = metrics[key]
            if any(token in key for token in filters):
                print(colored(f"  - {key}: {value}", "red"))
            else:
                print(f"  - {key}: {value}")
    else:
        print(colored("  - 无可用指标", "red"))

    if stage != "回测分析":
        return

    for report_path in ["portfolio_analysis/port_analysis_1day.pkl", "port_analysis_1day.pkl"]:
        try:
            portfolio_report = recorder.load_object(report_path)
            if isinstance(portfolio_report, pd.DataFrame):
                print(colored("\n  回测摘要表:", "red", attrs=["bold"]))
                print(portfolio_report.to_string())
                return
        except Exception:
            continue

    print(colored("\n  未找到可用的回测摘要数据", "red"))


def write_run_report(
    script_name: str,
    experiment_name: str,
    summary_lines: Iterable[str],
    metrics: Optional[Mapping[str, object]] = None,
    details_lines: Optional[Iterable[str]] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Persist a run summary under the examples/reports directory."""
    output_dir = Path(output_dir or Path(__file__).resolve().parent / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_script = Path(script_name).stem
    report_path = output_dir / f"{safe_script}_{timestamp}.txt"

    lines = [
        f"script={script_name}",
        f"experiment={experiment_name}",
        f"timestamp={timestamp}",
        "summary:",
        *[f"- {line}" for line in summary_lines],
        "metrics:",
    ]
    if metrics:
        lines.extend([f"- {key}: {value}" for key, value in sorted(metrics.items())])
    else:
        lines.append("- none")

    if details_lines:
        lines.extend(["details:"])
        for detail in details_lines:
            lines.append(detail)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
