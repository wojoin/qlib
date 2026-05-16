"""Inspect key artifacts from the CPO workflow MLflow run."""

from __future__ import annotations

import argparse
import ast
import html
import json
import importlib.util
import smtplib
import textwrap
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EXPERIMENT_DIR = Path("mlruns") / "331389798673625822"
DEFAULT_EXPORT_DIR = Path("examples") / "cpo"
MLRUNS_DIR = Path("mlruns")
DEFAULT_WORKFLOW_PATH = Path("examples") / "workflow_dual_horizon.py"
DEFAULT_CPO_UNIVERSE = "all"
DEFAULT_FROM_EMAIL = "1003257670@qq.com"
DEFAULT_TO_EMAILS = ["1003257670@qq.com", "wojoin@163.com"]
DEFAULT_EMAIL_SUBJECT = "CPO PCB 半导体 存储 光纤"
EMAIL_CELL_WRAP_CHARS = 80


def default_cpo_provider_uri(current_date: date | None = None) -> Path:
    current_date = current_date or date.today()
    return Path.home() / "qlib" / "examples" / "data" / current_date.strftime("%Y%m%d") / "qlib_data"


DEFAULT_CPO_PROVIDER_URI = default_cpo_provider_uri()


DEFAULT_TRAIN_RESULTS_DIR = Path("examples") / "data_train_results"


def dated_email_subject(subject: str, current_date: date | None = None, exp_label: str | None = None) -> str:
    current_date = current_date or date.today()
    date_prefix = current_date.strftime("%Y%m%d")
    if subject.startswith(date_prefix):
        return subject
    parts = [date_prefix]
    if exp_label:
        parts.append(exp_label)
    parts.append(subject)
    return " ".join(parts)


def default_export_dir_for_run(run_dir_spec: str | None, run_dir: Path) -> Path:
    """Return examples/data_train_results/<YYYYMMDD>/<name> where <name> is derived from --run-dir.

    - experiment name shorthand (e.g. 5D_Mid_Term)  → used as-is
    - explicit path (contains /)                     → run_id (last path component)
    - omitted (None)                                 → MLflow experiment name from run_dir's parent meta.yaml
    """
    if run_dir_spec is None:
        meta = run_dir.parent / "meta.yaml"
        name = _read_experiment_name(meta) if meta.exists() else None
        name = name or run_dir.parent.name
    elif "/" in run_dir_spec or "\\" in run_dir_spec:
        name = Path(run_dir_spec).name
    else:
        name = run_dir_spec
    date_str = date.today().strftime("%Y%m%d")
    return DEFAULT_TRAIN_RESULTS_DIR / date_str / name


def latest_run_dir(experiment_dir: Path = DEFAULT_EXPERIMENT_DIR) -> Path:
    run_dirs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and (path / "artifacts").is_dir()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No MLflow run directories found under {experiment_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _read_experiment_name(meta_path: Path) -> str | None:
    for line in meta_path.read_text().splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def find_experiment_dir(name: str, mlruns_dir: Path = MLRUNS_DIR) -> Path:
    """Find an experiment directory whose MLflow name contains *name* as a substring."""
    import difflib

    all_metas = list(mlruns_dir.glob("*/meta.yaml"))
    available = [n for m in all_metas if (n := _read_experiment_name(m))]
    matches = [
        meta.parent
        for meta in all_metas
        if (exp_name := _read_experiment_name(meta)) and name in exp_name
    ]
    if not matches:
        suggestions = difflib.get_close_matches(name, available, n=3, cutoff=0.4)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise FileNotFoundError(
            f"No MLflow experiment found matching '{name}'.{hint} "
            f"Available: {available}"
        )
    if len(matches) > 1:
        names = [_read_experiment_name(m / "meta.yaml") for m in matches]
        raise ValueError(
            f"Ambiguous experiment name '{name}' matches multiple experiments: {names}. "
            f"Use a more specific name or --exp-name mlruns/<experiment_id>/<run_id>."
        )
    return matches[0]


def resolve_run_dir(spec: str | None) -> Path:
    """Resolve --run-dir to an actual MLflow run directory.

    Accepts:
      - None              → latest run under DEFAULT_EXPERIMENT_DIR
      - existing path     → use directly (run dir) or pick latest run (experiment dir)
      - experiment name   → substring-match against mlruns/*/meta.yaml, then latest run
    """
    if spec is None:
        return latest_run_dir()
    path = Path(spec)
    if path.exists():
        if (path / "artifacts").is_dir():
            return path
        return latest_run_dir(path)
    # Not an existing filesystem path — treat as experiment name
    exp_dir = find_experiment_dir(spec)
    return latest_run_dir(exp_dir)


def strategy_kwargs_from_workflow(workflow_path: Path = DEFAULT_WORKFLOW_PATH) -> dict[str, int]:
    tree = ast.parse(workflow_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        items = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if items.get("class") and isinstance(items["class"], ast.Constant) and items["class"].value == "TopkDropoutStrategy":
            kwargs_node = items.get("kwargs")
            if isinstance(kwargs_node, ast.Dict):
                kwargs = {
                    key.value: value.value
                    for key, value in zip(kwargs_node.keys, kwargs_node.values)
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in {"topk", "n_drop"}
                    and isinstance(value, ast.Constant)
                }
                return {
                    "topk": int(kwargs.get("topk", 5)),
                    "n_drop": int(kwargs.get("n_drop", 1)),
                }
    return {"topk": 5, "n_drop": 1}


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "params": run_dir / "artifacts" / "params.pkl",
        "pred": run_dir / "artifacts" / "pred.pkl",
        "cpo_latest_recommendations": run_dir / "artifacts" / "cpo_latest_recommendations.pkl",
        "positions": run_dir
        / "artifacts"
        / "portfolio_analysis"
        / "positions_normal_1day.pkl",
        "port_analysis": run_dir
        / "artifacts"
        / "portfolio_analysis"
        / "port_analysis_1day.pkl",
        "indicator_analysis": run_dir
        / "artifacts"
        / "portfolio_analysis"
        / "indicator_analysis_1day.pkl",
    }


def load_cpo_stock_list(provider_uri: Path = DEFAULT_CPO_PROVIDER_URI, universe: str = DEFAULT_CPO_UNIVERSE) -> list[str]:
    instruments_path = provider_uri / "instruments" / f"{universe}.txt"
    if not instruments_path.exists():
        raise FileNotFoundError(f"CPO instruments file not found: {instruments_path}")
    stocks = []
    for line in instruments_path.read_text().splitlines():
        if line.strip():
            stocks.append(line.split()[0])
    if not stocks:
        raise ValueError(f"CPO instruments file is empty: {instruments_path}")
    return stocks


def validate_cpo_index(obj: pd.DataFrame | pd.Series, cpo_stock_list: list[str], context: str) -> pd.DataFrame | pd.Series:
    if not isinstance(obj.index, pd.MultiIndex) or "instrument" not in obj.index.names:
        return obj
    cpo_stock_set = set(cpo_stock_list)
    instruments = obj.index.get_level_values("instrument")
    non_cpo = sorted(set(instruments) - cpo_stock_set)
    # if non_cpo:
    #     raise ValueError(f"{context} contains non-CPO instruments: {non_cpo}")
    return obj[instruments.isin(cpo_stock_set)]


def normalize_code(code: str) -> str:
    code = str(code).upper()
    if code.startswith(("SH", "SZ")):
        return code
    if code.startswith("6"):
        return f"SH{code}"
    return f"SZ{code}"


def describe_dataframe(df: pd.DataFrame, rows: int) -> str:
    parts = [
        f"shape: {df.shape}",
        "columns: " + ", ".join(map(str, df.columns)),
        "head:",
        df.head(rows).to_string(),
        "tail:",
        df.tail(rows).to_string(),
    ]
    return "\n".join(parts)


def describe_series(series: pd.Series, rows: int) -> str:
    parts = [
        f"shape: {series.shape}",
        f"name: {series.name}",
        "head:",
        series.head(rows).to_string(),
        "tail:",
        series.tail(rows).to_string(),
    ]
    return "\n".join(parts)


def describe_object(obj: Any, rows: int) -> str:
    if isinstance(obj, pd.DataFrame):
        return describe_dataframe(obj, rows)
    if isinstance(obj, pd.Series):
        return describe_series(obj, rows)
    if isinstance(obj, dict):
        keys = list(obj.keys())
        preview_keys = keys[:rows]
        lines = [f"type: dict", f"keys: {len(keys)}", f"first keys: {preview_keys}"]
        for key in preview_keys:
            lines.append(f"{key!r}: {obj[key]!r}")
        return "\n".join(lines)
    return f"type: {type(obj).__name__}\nvalue:\n{obj!r}"


def position_to_frame(position: Any) -> pd.DataFrame:
    if hasattr(position, "position"):
        raw_position = position.position
    else:
        raw_position = position.get("position", position)

    rows = []
    for instrument, value in raw_position.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "instrument": instrument,
                "weight": float(value.get("weight", 0.0)),
                "amount": float(value.get("amount", 0.0)),
                "price": float(value.get("price", 0.0)),
                "count_day": int(value.get("count_day", 0)),
            }
        )
    return pd.DataFrame(rows)


FEATURE_LABELS = {
    "OPEN0": "开盘相对收盘",
    "HIGH0": "日内高点相对收盘",
    "LOW0": "日内低点相对收盘",
    "VWAP0": "VWAP相对收盘",
    "MA5": "5日均线",
    "MA10": "10日均线",
    "MA20": "20日均线",
    "MA60": "60日均线",
    "VROC1": "当日量能变化",
    "VOL_RATIO20": "放量倍数",
    "PRICE_VOLUME_CORR10": "10日量价相关",
    "RETURN_PER_VOLUME5": "单位量能收益",
    "VOL_CV20": "成交量波动",
}


def model_feature_importance(model: Any, feature_names: list[str]) -> pd.Series:
    booster = getattr(model, "model", None)
    if booster is None or not hasattr(booster, "feature_importance"):
        return pd.Series(1.0, index=feature_names)
    values = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    if names and all(str(name).startswith("Column_") for name in names):
        names = feature_names[: len(values)]
    importance = pd.Series(values, index=names, dtype="float64").reindex(feature_names).fillna(0.0)
    if importance.sum() <= 0:
        return pd.Series(1.0, index=feature_names)
    return importance / importance.sum()


def build_feature_diagnostics(
    pred_score: pd.Series,
    infer_features: pd.DataFrame,
    raw_features: pd.DataFrame | None = None,
    previous_raw_features: pd.DataFrame | None = None,
    importance: pd.Series | None = None,
) -> dict[str, Any]:
    features = infer_features.apply(pd.to_numeric, errors="coerce")
    pred_score = pred_score.reindex(features.index)
    raw_features = raw_features.reindex(features.index) if raw_features is not None else features
    zscore = (features - features.mean()) / features.std(ddof=0).replace(0, 1)
    percentile = features.rank(pct=True)
    corr = features.apply(
        lambda col: 0.0 if col.std(ddof=0) == 0 or pred_score.std(ddof=0) == 0 else col.corr(pred_score)
    ).fillna(0.0)
    importance = importance.reindex(features.columns).fillna(0.0) if importance is not None else corr.abs()
    if importance.sum() <= 0:
        importance = pd.Series(1.0, index=features.columns)
    importance = importance / importance.sum()
    contribution = zscore.mul(corr, axis=1).mul(importance, axis=1)
    return {
        "raw": raw_features,
        "previous_raw": previous_raw_features,
        "infer": features,
        "percentile": percentile,
        "corr": corr,
        "importance": importance,
        "contribution": contribution,
    }


def feature_phrase(instrument: str, feature: str, diagnostics: dict[str, Any]) -> str:
    raw = diagnostics["raw"]
    percentile = diagnostics["percentile"]
    corr = diagnostics["corr"]
    raw_value = raw.loc[instrument, feature] if instrument in raw.index and feature in raw.columns else float("nan")
    pct_value = percentile.loc[instrument, feature] if instrument in percentile.index and feature in percentile.columns else float("nan")
    preference = "偏好高值" if corr.get(feature, 0.0) >= 0 else "偏好低值"
    label = FEATURE_LABELS.get(feature, feature)

    if feature == "VOL_RATIO20":
        return f"{label} {raw_value:.2f}倍，行业分位 {pct_value:.0%}，模型当日{preference}"
    if feature == "VROC1":
        return f"{label} {raw_value:.2%}，行业分位 {pct_value:.0%}，模型当日{preference}"
    if feature == "PRICE_VOLUME_CORR10":
        return f"{label} {raw_value:.2f}，行业分位 {pct_value:.0%}，模型当日{preference}"
    if feature == "RETURN_PER_VOLUME5":
        return f"{label} {raw_value:.4f}，行业分位 {pct_value:.0%}，模型当日{preference}"
    if feature == "VOL_CV20":
        return f"{label} {raw_value:.2f}，行业分位 {pct_value:.0%}，模型当日{preference}"
    if feature in {"MA5", "MA10", "MA20", "MA60"}:
        ma5 = raw.loc[instrument, "MA5"] if "MA5" in raw.columns and instrument in raw.index else None
        ma10 = raw.loc[instrument, "MA10"] if "MA10" in raw.columns and instrument in raw.index else None
        relation = ""
        if ma5 is not None and ma10 is not None:
            prev = diagnostics.get("previous_raw")
            if (
                prev is not None
                and instrument in prev.index
                and "MA5" in prev.columns
                and "MA10" in prev.columns
                and prev.loc[instrument, "MA5"] <= prev.loc[instrument, "MA10"]
                and ma5 > ma10
            ):
                relation = "，MA5上穿MA10"
            elif (
                prev is not None
                and instrument in prev.index
                and "MA5" in prev.columns
                and "MA10" in prev.columns
                and prev.loc[instrument, "MA5"] >= prev.loc[instrument, "MA10"]
                and ma5 < ma10
            ):
                relation = "，MA5下穿MA10"
            else:
                relation = "，MA5高于MA10" if ma5 > ma10 else "，MA5低于MA10"
        return f"{label}相对收盘 {raw_value:.3f}{relation}，行业分位 {pct_value:.0%}，模型当日{preference}"
    return f"{label}={raw_value:.3f}，行业分位 {pct_value:.0%}，模型当日{preference}"


def explain_instrument(instrument: str, diagnostics: dict[str, Any] | None, mode: str = "support", limit: int = 3) -> str:
    if diagnostics is None or instrument not in diagnostics["contribution"].index:
        return ""
    contributions = diagnostics["contribution"].loc[instrument].dropna()
    if contributions.empty:
        return ""
    selected = contributions.nlargest(limit) if mode == "support" else contributions.nsmallest(limit)
    direction = "支撑评分" if mode == "support" else "拖累评分"
    phrases = [feature_phrase(instrument, feature, diagnostics) for feature in selected.index]
    return f"{direction}: " + "；".join(phrases)


def candidate_top10(
    pred_score: pd.Series,
    current_holding: set[str],
    diagnostics: dict[str, Any] | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    rows = []
    for rank, (instrument, score) in enumerate(pred_score.sort_values(ascending=False).head(limit).items(), start=1):
        rows.append(
            {
                "rank": rank,
                "instrument": instrument,
                "score": float(score),
                "is_current_holding": instrument in current_holding,
                "feature_reason": explain_instrument(instrument, diagnostics, "support"),
            }
        )
    return pd.DataFrame(rows)


def action_plan(
    pred_score: pd.Series,
    positions: pd.DataFrame,
    topk: int = 5,
    n_drop: int = 1,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    pred_score = pred_score.sort_values(ascending=False)
    held = positions["instrument"].tolist()
    held_ranked = pred_score.reindex(held).sort_values(ascending=False).index
    buy_candidates = pred_score[~pred_score.index.isin(held_ranked)].index[: n_drop + topk - len(held_ranked)]
    combined = pred_score.reindex(held_ranked.union(pd.Index(buy_candidates))).sort_values(ascending=False).index
    sell = set(held_ranked[held_ranked.isin(list(combined)[-n_drop:])])
    buy = set(buy_candidates[: len(sell) + topk - len(held_ranked)])

    rows = []
    for instrument in held:
        current = positions.loc[positions["instrument"] == instrument].iloc[0].to_dict()
        action = "SELL" if instrument in sell else "HOLD"
        score = float(pred_score.get(instrument, float("nan")))
        if action == "SELL":
            interpretation = f"建议卖出 {instrument}，释放当前约 {current['weight']:.2%} 的仓位。"
            feature_reason = explain_instrument(instrument, diagnostics, "drag")
            reason = f"在当前持仓与新候选标的合并排序后，{instrument} 属于评分最低的 {n_drop} 只，需要按策略规则调出。"
        else:
            interpretation = f"建议继续持有 {instrument}，当前仓位约 {current['weight']:.2%}。"
            feature_reason = explain_instrument(instrument, diagnostics, "support")
            reason = f"{instrument} 仍保留在组合内，评分没有低到触发本轮卖出。"
        if feature_reason:
            reason = f"{reason} {feature_reason}"
        rows.append(
            {
                "action": action,
                "instrument": instrument,
                "score": score,
                "weight": current["weight"],
                "amount": current["amount"],
                "price": current["price"],
                "count_day": current["count_day"],
                "interpretation": interpretation,
                "feature_reason": feature_reason,
                "reason": reason,
            }
        )
    for instrument in buy:
        score = float(pred_score[instrument])
        feature_reason = explain_instrument(instrument, diagnostics, "support")
        reason = f"{instrument} 是当前未持仓股票中评分最高的候选标的，按 topk={topk}, n_drop={n_drop} 的策略规则进入买入名单。"
        if feature_reason:
            reason = f"{reason} {feature_reason}"
        rows.append(
            {
                "action": "BUY",
                "instrument": instrument,
                "score": score,
                "weight": 0.0,
                "amount": 0.0,
                "price": 0.0,
                "count_day": 0,
                "interpretation": f"建议新买入 {instrument}，用卖出资金补入组合。",
                "feature_reason": feature_reason,
                "reason": reason,
            }
        )
    action_order = {"SELL": 0, "BUY": 1, "HOLD": 2}
    columns = [
        "action",
        "instrument",
        "score",
        "weight",
        "amount",
        "price",
        "count_day",
        "interpretation",
        "feature_reason",
        "reason",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["action", "score"],
            key=lambda s: s.map(action_order) if s.name == "action" else s,
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return pd.read_pickle(path)


def load_sig_analysis(run_dir: Path) -> pd.DataFrame | None:
    sig_dir = run_dir / "artifacts" / "sig_analysis"
    ic_path = sig_dir / "ic.pkl"
    ric_path = sig_dir / "ric.pkl"
    if not ic_path.exists() or not ric_path.exists():
        return None
    ic = pd.read_pickle(ic_path).dropna()
    ric = pd.read_pickle(ric_path).dropna()
    return pd.DataFrame(
        {
            "value": {
                "IC": ic.mean(),
                "Rank IC": ric.mean(),
                "ICIR": ic.mean() / ic.std(),
                "Rank ICIR": ric.mean() / ric.std(),
            }
        }
    )


def inspect_sig_analysis(run_dir: Path) -> pd.DataFrame | None:
    sig = load_sig_analysis(run_dir)
    if sig is None:
        return None
    print("=" * 88)
    print("sig_analysis (IC / Rank IC / ICIR / Rank ICIR)")
    print(sig.to_string())
    return sig


def load_feature_frames(pred_date: pd.Timestamp, workflow_path: Path = DEFAULT_WORKFLOW_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    spec = importlib.util.spec_from_file_location("workflow_by_cpo_for_artifacts", workflow_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow config from {workflow_path}")
    workflow_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow_module)

    qlib.init(
        provider_uri=str(DEFAULT_CPO_PROVIDER_URI.resolve()),
        region=REG_CN,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file://" + str(MLRUNS_DIR.resolve()),
                "default_exp_name": "Experiment",
            },
        },
    )
    dataset = init_instance_by_config(workflow_module.CPO_TASK["dataset"])
    raw_all = dataset.handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    infer_all = dataset.handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    dates = raw_all.index.get_level_values("datetime").unique()
    dates = dates[dates <= pred_date]
    current_date = dates[-1]
    previous_date = dates[-2] if len(dates) > 1 else None
    raw = raw_all.loc(axis=0)[current_date, :]
    infer = infer_all.loc(axis=0)[current_date, :]
    previous_raw = raw_all.loc(axis=0)[previous_date, :] if previous_date is not None else None
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.droplevel("datetime")
    if isinstance(infer.index, pd.MultiIndex):
        infer = infer.droplevel("datetime")
    if previous_raw is not None and isinstance(previous_raw.index, pd.MultiIndex):
        previous_raw = previous_raw.droplevel("datetime")
    return raw, infer, previous_raw


def load_diagnostics(run_dir: Path, pred_score: pd.Series, pred_date: pd.Timestamp) -> dict[str, Any] | None:
    try:
        raw_features, infer_features, previous_raw_features = load_feature_frames(pred_date)
        model = load_pickle(artifact_paths(run_dir)["params"])
        importance = model_feature_importance(model, list(infer_features.columns))
        return build_feature_diagnostics(
            pred_score,
            infer_features,
            raw_features,
            previous_raw_features=previous_raw_features,
            importance=importance,
        )
    except Exception as exc:
        print(f"feature diagnostics skipped: {exc}")
        return None


def export_csv(name: str, obj: Any, out_dir: Path) -> Path | None:
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.csv"
        obj.to_csv(out_path)
        return out_path
    return None


def inspect_artifacts(run_dir: Path, rows: int, export_dir: Path | None, cpo_stock_list: list[str]) -> None:
    paths = artifact_paths(run_dir)
    for name, path in paths.items():
        if name in {"params", "positions"}:
            continue
        if not path.exists():
            continue
        print("=" * 88)
        print(f"{name}: {path}")
        obj = load_pickle(path)
        if name == "pred":
            obj = validate_cpo_index(obj, cpo_stock_list, "pred.pkl")
        print(describe_object(obj, rows))
        if export_dir is not None:
            exported = export_csv(name, obj, export_dir)
            if exported is not None:
                print(f"csv: {exported}")
            else:
                print("csv: skipped; object is not a pandas DataFrame/Series")


def inspect_action_plan(run_dir: Path, topk: int, n_drop: int, export_dir: Path | None, cpo_stock_list: list[str]) -> None:
    paths = artifact_paths(run_dir)
    pred = load_pickle(paths["pred"])
    pred = validate_cpo_index(pred, cpo_stock_list, "pred.pkl")
    positions = load_pickle(paths["positions"])
    pred_date = pred.index.get_level_values("datetime").max()
    position_date = max(positions)
    pred_score = pred.loc[pred_date].iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred.loc[pred_date]
    pred_score = pred_score.reindex(cpo_stock_list).dropna()
    current_positions = position_to_frame(positions[position_date])
    current_positions = current_positions[current_positions["instrument"].isin(cpo_stock_list)]
    diagnostics = load_diagnostics(run_dir, pred_score, pred_date)
    plan = action_plan(pred_score, current_positions, topk=topk, n_drop=n_drop, diagnostics=diagnostics)
    candidates = candidate_top10(pred_score, set(current_positions["instrument"]), diagnostics=diagnostics, limit=10)

    print("=" * 88)
    print(f"plain action plan based on {position_date.date()} positions and {pred_date.date()} scores")
    print(f"strategy approximation: TopkDropoutStrategy(topk={topk}, n_drop={n_drop})")
    print(plan.to_string(index=False))
    print("\ntop 10 candidates")
    print(candidates.to_string(index=False))
    if export_dir is not None:
        exported = export_csv("action_plan", plan, export_dir)
        print(f"csv: {exported}")
        exported = export_csv("candidate_top10", candidates, export_dir)
        print(f"csv: {exported}")
    return plan, candidates


def email_cell_html(value: Any, wrap_chars: int = EMAIL_CELL_WRAP_CHARS) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    wrapped_lines: list[str] = []
    for line in text.splitlines() or [""]:
        if len(line) <= wrap_chars:
            wrapped_lines.append(html.escape(line))
            continue
        wrapped_lines.extend(
            html.escape(part)
            for part in textwrap.wrap(
                line,
                width=wrap_chars,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    return "<br>".join(wrapped_lines)


def email_table_html(df: pd.DataFrame, include_index: bool = True) -> str:
    display_df = df.reset_index() if include_index else df
    scroll_style = (
        "display:block;width:100%;max-width:100%;overflow-x:auto;overflow-y:hidden;"
        "-webkit-overflow-scrolling:touch;border:1px solid #d8dee4;margin:8px 0 18px 0;"
    )
    table_style = (
        "border-collapse:collapse;width:max-content;min-width:100%;font-family:Arial,sans-serif;"
        "font-size:13px;line-height:1.35;"
    )
    th_style = (
        "border:1px solid #d8dee4;background:#f6f8fa;color:#24292f;padding:6px 8px;"
        "text-align:left;vertical-align:top;white-space:nowrap;font-weight:600;"
    )
    td_style = (
        "border:1px solid #d8dee4;color:#24292f;padding:6px 8px;text-align:left;"
        "vertical-align:top;white-space:nowrap;"
    )

    header_cells = "".join(
        f'<th style="{th_style}">{email_cell_html(column)}</th>' for column in display_df.columns
    )
    body_rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f'<td style="{td_style}">{email_cell_html(value)}</td>' for value in row)
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        f'<div style="{scroll_style}">'
        f'<table role="table" cellpadding="0" cellspacing="0" style="{table_style}">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def html_from_obj(obj: Any, title: str, max_rows: int = 30, note: str = "") -> str:
    title_html = f"<h2>{html.escape(title)}</h2>"
    note_html = f"<p><strong>{html.escape(note)}</strong></p>" if note else ""
    if isinstance(obj, pd.DataFrame):
        display_df = obj if len(obj) <= max_rows else pd.concat([obj.head(max_rows // 2), obj.tail(max_rows // 2)])
        return title_html + note_html + email_table_html(display_df, include_index=True)
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=obj.name).reset_index()
        return title_html + note_html + email_table_html(df, include_index=False)
    return title_html + note_html + f"<pre>{html.escape(str(obj))}</pre>"


def sibling_instruments(export_dir: Path | None, filename: str) -> set[str]:
    """Collect instruments from the same CSV file in sibling experiment directories."""
    if export_dir is None or not export_dir.parent.exists():
        return set()
    instruments: set[str] = set()
    for sibling in export_dir.parent.iterdir():
        if sibling == export_dir or not sibling.is_dir():
            continue
        csv_path = sibling / filename
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            if "instrument" in df.columns:
                instruments.update(df["instrument"].dropna().tolist())
        except Exception:
            pass
    return instruments


def build_email_body(
    run_dir: Path,
    plan: pd.DataFrame,
    candidates: pd.DataFrame,
    indicator_analysis: Any,
    port_analysis: Any,
    both_plan: list[str] | None = None,
    both_candidates: list[str] | None = None,
    sig_analysis: Any = None,
) -> str:
    trade_date = date.today().strftime("%Y%m%d")
    plan_note = f"Both: {' '.join(both_plan)}" if both_plan else ""
    candidates_note = f"Both: {' '.join(both_candidates)}" if both_candidates else ""
    parts = [
        f"<h1>交易日: {trade_date} Workflow 日报</h1>",
        f"<p><strong>Run directory:</strong> {html.escape(str(run_dir))}</p>",
        html_from_obj(plan, "Action Plan", note=plan_note),
        html_from_obj(candidates, "Candidate Top10", note=candidates_note),
        html_from_obj(indicator_analysis, "Indicator Analysis"),
        html_from_obj(port_analysis, "Port Analysis"),
    ]
    if sig_analysis is not None:
        parts.append(html_from_obj(sig_analysis, "Signal Analysis (IC / Rank IC / ICIR / Rank ICIR)"))
    return "<html><body>" + "".join(parts) + "</body></html>"


def send_email(
    subject: str,
    body_html: str,
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    to_emails: list[str],
) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = ", ".join(to_emails)
    message.set_content("QLib workflow report. If you see this text, please use an HTML-capable email client.")
    message.add_alternative(body_html, subtype="html")

    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(message)
    else:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View pred.pkl, port_analysis_1day.pkl, and indicator_analysis_1day.pkl from a CPO MLflow run."
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help=(
            "Experiment name (e.g. 5D_Mid_Term, 1D_Short_Term) to use the latest run, "
            "or an explicit path mlruns/<experiment_id>/<run_id> for a specific run. "
            "Defaults to latest CPO run."
        ),
    )
    parser.add_argument("--rows", type=int, default=8, help="Rows to show from head and tail.")
    parser.add_argument(
        "--action-plan",
        action="store_true",
        help="Show plain-language current holdings, sell, buy, and hold lists.",
    )
    parser.add_argument("--topk", type=int, default=None, help="TopkDropoutStrategy topk value.")
    parser.add_argument("--n-drop", type=int, default=None, help="TopkDropoutStrategy n_drop value.")
    parser.add_argument(
        "--show-codes",
        nargs="*",
        default=[],
        help="Show latest score and rank for specific codes, e.g. 300308 300502 002281.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export pandas artifacts to CSV under examples/data_train_results/<run-dir> by default.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Override the export directory when --export-csv is set.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file for email settings (default: config/config.json).",
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send this run result by email when SMTP credentials and recipients are provided.",
    )
    parser.add_argument(
        "--smtp-server",
        type=str,
        default=None,
        help="SMTP server address.",
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=None,
        help="SMTP server port.",
    )
    parser.add_argument(
        "--smtp-user",
        type=str,
        default=None,
        help="SMTP login user (usually the sender email address).",
    )
    parser.add_argument(
        "--smtp-password",
        type=str,
        default=None,
        help="SMTP login password or auth code.",
    )
    parser.add_argument(
        "--from-email",
        type=str,
        default=None,
        help="Sender email address. Defaults to smtp-user if omitted.",
    )
    parser.add_argument(
        "--to-emails",
        nargs="*",
        default=None,
        help="Recipient email addresses.",
    )
    parser.add_argument(
        "--email-subject",
        type=str,
        default=None,
        help="Email subject line.",
    )
    return parser.parse_args()


def inspect_specific_codes(run_dir: Path, codes: list[str], cpo_stock_list: list[str]) -> None:
    if not codes:
        return
    pred = validate_cpo_index(load_pickle(artifact_paths(run_dir)["pred"]), cpo_stock_list, "pred.pkl")
    pred_date = pred.index.get_level_values("datetime").max()
    pred_score = pred.loc[pred_date].iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred.loc[pred_date]
    ranked = pred_score.reindex(cpo_stock_list).dropna().sort_values(ascending=False)

    rows = []
    for raw_code in codes:
        code = normalize_code(raw_code)
        rows.append(
            {
                "query": raw_code,
                "instrument": code,
                "in_cpo_universe": code in cpo_stock_list,
                "rank": int(ranked.index.get_loc(code) + 1) if code in ranked.index else None,
                "score": float(ranked.loc[code]) if code in ranked.index else None,
            }
        )
    result = pd.DataFrame(rows)
    print("=" * 88)
    print(f"specific CPO code ranks based on {pred_date.date()} scores")
    print(result.to_string(index=False))


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        default_paths = [Path("config") / "config.json", Path("examples") / "config" / "config.json"]
        for path in default_paths:
            if path.exists():
                config_path = path
                break
    if config_path is None or not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def merge_email_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    email_subject = args.email_subject if args.email_subject is not None else DEFAULT_EMAIL_SUBJECT
    exp_label = args.exp_name.replace("_Term", "") if args.exp_name else None
    settings = {
        "smtp_server": args.smtp_server if args.smtp_server is not None else config.get("smtp_server", "smtp.qq.com"),
        "smtp_port": args.smtp_port if args.smtp_port is not None else config.get("smtp_port", 465),
        "smtp_user": args.smtp_user if args.smtp_user is not None else config.get("smtp_user"),
        "smtp_password": args.smtp_password if args.smtp_password is not None else config.get("smtp_password"),
        "from_email": args.from_email if args.from_email is not None else DEFAULT_FROM_EMAIL,
        "to_emails": args.to_emails if args.to_emails is not None else DEFAULT_TO_EMAILS,
        "email_subject": dated_email_subject(email_subject, exp_label=exp_label),
    }
    if isinstance(settings["to_emails"], str):
        settings["to_emails"] = [email.strip() for email in settings["to_emails"].split(",") if email.strip()]
    if settings["from_email"] is None:
        settings["from_email"] = settings["smtp_user"]
    return settings


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(args.exp_name)
    strategy_kwargs = strategy_kwargs_from_workflow()
    topk = args.topk if args.topk is not None else strategy_kwargs["topk"]
    n_drop = args.n_drop if args.n_drop is not None else strategy_kwargs["n_drop"]
    export_dir = (args.export_dir or default_export_dir_for_run(args.exp_name, run_dir)) if args.export_csv else None
    cpo_stock_list = load_cpo_stock_list()
    print(f"run_dir: {run_dir}")
    print(f"CPO universe: {len(cpo_stock_list)} stocks from {DEFAULT_CPO_PROVIDER_URI / 'instruments' / (DEFAULT_CPO_UNIVERSE + '.txt')}")
    if args.action_plan:
        print(f"strategy: TopkDropoutStrategy(topk={topk}, n_drop={n_drop})")
    inspect_artifacts(run_dir, args.rows, export_dir, cpo_stock_list)
    sig_analysis = inspect_sig_analysis(run_dir)
    inspect_specific_codes(run_dir, args.show_codes, cpo_stock_list)
    if args.action_plan:
        plan, candidates = inspect_action_plan(run_dir, topk, n_drop, export_dir, cpo_stock_list)
    else:
        plan, candidates = None, None

    if args.send_email:
        settings = merge_email_settings(args, config)
        if not settings["smtp_user"] or not settings["smtp_password"] or not settings["to_emails"]:
            raise ValueError("--send-email requires smtp_user, smtp_password, and to_emails from args or config")
        if plan is None:
            raise ValueError("--send-email requires --action-plan to generate action plan and candidate top1 content.")
        paths = artifact_paths(run_dir)
        indicator_analysis = load_pickle(paths["indicator_analysis"]) if paths["indicator_analysis"].exists() else None
        port_analysis = load_pickle(paths["port_analysis"]) if paths["port_analysis"].exists() else None
        sibling_plan = sibling_instruments(export_dir, "action_plan.csv")
        sibling_cands = sibling_instruments(export_dir, "candidate_top10.csv")
        plan_instruments = set(plan["instrument"].tolist()) if "instrument" in plan.columns else set()
        cand_instruments = set(candidates["instrument"].tolist()) if "instrument" in candidates.columns else set()
        both_plan = sorted(plan_instruments & sibling_plan)
        both_candidates = sorted(cand_instruments & sibling_cands)
        body_html = build_email_body(run_dir, plan, candidates, indicator_analysis, port_analysis,
                                     both_plan=both_plan, both_candidates=both_candidates,
                                     sig_analysis=sig_analysis)
        send_email(
            subject=settings["email_subject"],
            body_html=body_html,
            smtp_server=settings["smtp_server"],
            smtp_port=settings["smtp_port"],
            smtp_user=settings["smtp_user"],
            smtp_password=settings["smtp_password"],
            from_email=settings["from_email"],
            to_emails=settings["to_emails"],
        )
        print(f"Email sent to: {', '.join(settings['to_emails'])}")


if __name__ == "__main__":
    main()
