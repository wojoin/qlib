"""Inspect key artifacts from the CPO workflow MLflow run."""

from __future__ import annotations

import argparse
import ast
import importlib.util
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EXPERIMENT_DIR = Path("mlruns") / "331389798673625822"
DEFAULT_EXPORT_DIR = Path("examples") / "cpo"
DEFAULT_WORKFLOW_PATH = Path("examples") / "workflow_by_cpo_latest.py"
DEFAULT_CPO_PROVIDER_URI = Path("examples") / "data" / "parquet" / "20260507" / "qlib_data"
DEFAULT_CPO_UNIVERSE = "all"


def default_export_dir(current_date: date | None = None) -> Path:
    current_date = current_date or date.today()
    return DEFAULT_EXPORT_DIR / current_date.strftime("%Y%m%d")


def latest_run_dir(experiment_dir: Path = DEFAULT_EXPERIMENT_DIR) -> Path:
    run_dirs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and (path / "artifacts").is_dir()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No MLflow run directories found under {experiment_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


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
    if non_cpo:
        raise ValueError(f"{context} contains non-CPO instruments: {non_cpo}")
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

    qlib.init(provider_uri=str(DEFAULT_CPO_PROVIDER_URI.resolve()), region=REG_CN)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View pred.pkl, port_analysis_1day.pkl, and indicator_analysis_1day.pkl from a CPO MLflow run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="MLflow run directory, for example mlruns/<experiment_id>/<run_id>. Defaults to latest CPO run.",
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
        help="Export pandas artifacts to CSV under examples/cpo/YYYYMMDD by default.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory for CSV exports when --export-csv is set.",
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


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    strategy_kwargs = strategy_kwargs_from_workflow()
    topk = args.topk if args.topk is not None else strategy_kwargs["topk"]
    n_drop = args.n_drop if args.n_drop is not None else strategy_kwargs["n_drop"]
    export_dir = (args.export_dir or default_export_dir()) if args.export_csv else None
    cpo_stock_list = load_cpo_stock_list()
    print(f"run_dir: {run_dir}")
    print(f"CPO universe: {len(cpo_stock_list)} stocks from {DEFAULT_CPO_PROVIDER_URI / 'instruments' / (DEFAULT_CPO_UNIVERSE + '.txt')}")
    if args.action_plan:
        print(f"strategy: TopkDropoutStrategy(topk={topk}, n_drop={n_drop})")
    inspect_artifacts(run_dir, args.rows, export_dir, cpo_stock_list)
    inspect_specific_codes(run_dir, args.show_codes, cpo_stock_list)
    if args.action_plan:
        inspect_action_plan(run_dir, topk, n_drop, export_dir, cpo_stock_list)


if __name__ == "__main__":
    main()
