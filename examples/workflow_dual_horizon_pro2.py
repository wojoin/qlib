from pathlib import Path

import numpy as np
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.model.base import Model
import lightgbm as lgb

try:
    from examples.dual_horizon_dates import get_data_dates
    from examples.reporting_utils import (
        format_dataframe_for_report,
        print_colored_block,
        print_final_results,
        print_latest_recommendations,
        write_run_report,
    )
except ModuleNotFoundError:
    from dual_horizon_dates import get_data_dates
    from reporting_utils import (
        format_dataframe_for_report,
        print_colored_block,
        print_final_results,
        print_latest_recommendations,
        write_run_report,
    )

# ==================== 🛠️ 自定义：工业级 LambdaRank 排序模型 ====================
class LGBLambdaModel(Model):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    def _prepare_data_for_rank(self, dataset, segment):
        """将 Qlib 数据集清洗并平滑转换为标准 LambdaRank 截面档位格式"""
        df = dataset.prepare(segment, col_set=["feature", "label"])
        df = df.sort_index(level="datetime")
        
        x = df["feature"]
        raw_y = df["label"].iloc[:, 0]
        
        # 将标准收益率映射到 0-9 档的非负整数，用于 NDCG 排序优化
        y_pct = raw_y.groupby(level="datetime").rank(pct=True, ascending=True)
        y_int = (y_pct * 9).fillna(0).astype(int)
        
        # 计算 LambdaRank 必须的截面每日 group 股票数量
        group = df.groupby(level="datetime").size().values
        return x, y_int, group

    def fit(self, dataset, **kwargs):
        x_train, y_train, group_train = self._prepare_data_for_rank(dataset, "train")
        x_valid, y_valid, group_valid = self._prepare_data_for_rank(dataset, "valid")
        
        train_dataset = lgb.Dataset(x_train, label=y_train, group=group_train)
        valid_dataset = lgb.Dataset(x_valid, label=y_valid, group=group_valid, reference=train_dataset)
        
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 64,
            "num_threads": 12,
            "verbosity": -1,
            "label_gain": [0, 1, 2, 3, 4, 15, 30, 60, 120, 250] # 强化 Top 5 选股惩罚
        }
        params.update(self.params)
        
        num_iterations = params.pop("num_iterations", 800)
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if early_stopping_rounds else []
        
        self.model = lgb.train(
            params,
            train_set=train_dataset,
            num_boost_round=num_iterations,
            valid_sets=[valid_dataset],
            callbacks=callbacks
        )

    def predict(self, dataset, **kwargs):
        if self.model is None:
            raise ValueError("模型尚未进行训练！")
        
        df_test = dataset.prepare("test", col_set="feature")
        raw_preds = self.model.predict(df_test)
        
        # 将原始预测分转换为每天截面内部标准的 0.0~1.0 相对排名，消除跨期数值漂移
        pred_series = pd.Series(raw_preds, index=df_test.index)
        final_score = pred_series.groupby(level="datetime").rank(pct=True)
        return final_score


TECH_UNIVERSE = "all"
RECOMMEND_TOPK = 5
NDROP = 2
INITIAL_CAPITAL = 300000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
POSITION_STATE_DIR = EXAMPLES_DIR / "position_state"
METRIC_HISTORY_DIR = EXAMPLES_DIR / "metric_history"
METRIC_HISTORY_PATH = METRIC_HISTORY_DIR / "strategy_metrics_history.pkl"
ADAPTIVE_STRATEGY_NAME = "Adaptive_Exit"
STRATEGY_NAMES = ["1D_Short_Term", "5D_Mid_Term", ADAPTIVE_STRATEGY_NAME]
ADAPTIVE_1D_WEIGHT = 0.40
ADAPTIVE_5D_WEIGHT = 0.60
ADAPTIVE_MIN_HOLD_DAYS = 2
ADAPTIVE_MAX_HOLD_DAYS = 10
ADAPTIVE_EXIT_RANK = 10
ADAPTIVE_EXIT_SCORE = 0.55
ADAPTIVE_EMERGENCY_EXIT_SCORE = 0.35

POSITION_COLUMNS = [
    "strategy",
    "instrument",
    "entry_date",
    "last_updated",
    "holding_days",
    "score",
    "rank",
    "weight",
    "amount",
]
ACTION_COLUMNS = [
    "strategy",
    "trade_date",
    "action",
    "instrument",
    "entry_date",
    "holding_days",
    "score",
    "rank",
    "amount",
    "reason",
]
METRIC_COLUMNS = ["trade_date", "strategy", "IC", "ICIR", "Sharpe", "Sortino", "Calmar"]


def calculate_risk_metrics(report: pd.DataFrame, annualization_factor: int = 238) -> pd.DataFrame:
    """Calculate risk-adjusted ratios from the strategy's net daily returns."""
    required_columns = {"return", "cost"}
    missing_columns = required_columns.difference(report.columns)
    if missing_columns:
        raise ValueError(f"回测报告缺少必要字段: {', '.join(sorted(missing_columns))}")

    net_returns = pd.to_numeric(report["return"] - report["cost"], errors="coerce")
    net_returns = net_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if net_returns.empty:
        raise ValueError("回测报告中没有可用于计算风险指标的有效收益率")

    mean_return = net_returns.mean()
    volatility = net_returns.std(ddof=1)
    downside_deviation = np.sqrt(np.square(net_returns.clip(upper=0.0)).mean())

    equity_curve = (1.0 + net_returns).cumprod()
    max_drawdown = (equity_curve / equity_curve.cummax() - 1.0).min()
    annualized_return = equity_curve.iloc[-1] ** (annualization_factor / len(net_returns)) - 1.0

    sharpe = mean_return / volatility * np.sqrt(annualization_factor) if volatility > 0 else np.nan
    sortino = (
        mean_return / downside_deviation * np.sqrt(annualization_factor)
        if downside_deviation > 0
        else np.nan
    )
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return pd.DataFrame(
        {"value": {"Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar}}
    )


def calculate_strategy_metrics(
    ic: pd.Series,
    risk_metrics: pd.DataFrame,
    strategy: str,
    trade_date: object,
) -> pd.DataFrame:
    """Build one comparable daily snapshot of the five strategy metrics."""
    clean_ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ic_std = clean_ic.std(ddof=1)
    values = risk_metrics["value"]
    return pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp(trade_date).normalize(),
                "strategy": strategy,
                "IC": clean_ic.mean(),
                "ICIR": clean_ic.mean() / ic_std if ic_std > 0 else np.nan,
                "Sharpe": values.get("Sharpe", np.nan),
                "Sortino": values.get("Sortino", np.nan),
                "Calmar": values.get("Calmar", np.nan),
            }
        ],
        columns=METRIC_COLUMNS,
    )


def persist_metrics_history(
    snapshot: pd.DataFrame,
    history_path: Path = METRIC_HISTORY_PATH,
) -> pd.DataFrame:
    """Append metric snapshots, replacing duplicate strategy/trade-date rows."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history = pd.read_pickle(history_path)
        history = history.loc[:, METRIC_COLUMNS]
    else:
        history = pd.DataFrame(columns=METRIC_COLUMNS)

    if history.empty:
        combined = snapshot.loc[:, METRIC_COLUMNS].copy()
    else:
        combined = pd.concat([history, snapshot.loc[:, METRIC_COLUMNS]], ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"]).dt.normalize()
    combined = (
        combined.drop_duplicates(["trade_date", "strategy"], keep="last")
        .sort_values(["trade_date", "strategy"])
        .reset_index(drop=True)
    )
    combined.to_pickle(history_path)
    return combined


def format_strategy_console_summary(
    positions_by_strategy: dict[str, pd.DataFrame],
    metrics_history: pd.DataFrame,
) -> str:
    """Format the same core position and five-metric information used by email."""
    sections = ["\n" + "=" * 24 + " 三种策略持仓与五项指标汇总 " + "=" * 24]
    for strategy in STRATEGY_NAMES:
        positions = positions_by_strategy.get(strategy, pd.DataFrame(columns=POSITION_COLUMNS))
        strategy_metrics = metrics_history.loc[metrics_history["strategy"] == strategy]
        latest_metrics = (
            strategy_metrics.sort_values("trade_date").tail(1).loc[:, METRIC_COLUMNS]
            if not strategy_metrics.empty
            else pd.DataFrame(columns=METRIC_COLUMNS)
        )
        sections.extend(
            [
                f"\n[{strategy}] 当前持仓",
                positions.to_string(index=False) if not positions.empty else "(无当前持仓)",
                f"\n[{strategy}] 最新五项指标（IC / ICIR / Sharpe / Sortino / Calmar）",
                latest_metrics.to_string(index=False) if not latest_metrics.empty else "(暂无五项指标)",
            ]
        )
    return "\n".join(sections)


def latest_scores(pred: pd.DataFrame | pd.Series) -> tuple[pd.Timestamp, pd.Series]:
    """Return the latest cross-sectional score series from a Qlib prediction artifact."""
    if not isinstance(pred.index, pd.MultiIndex) or "datetime" not in pred.index.names:
        raise ValueError("预测结果必须使用包含 datetime 的 MultiIndex")
    latest_date = pd.Timestamp(pred.index.get_level_values("datetime").max()).normalize()
    latest = pred.xs(latest_date, level="datetime")
    if isinstance(latest, pd.DataFrame):
        if "score" not in latest.columns:
            raise ValueError("预测结果缺少 score 字段")
        latest = latest["score"]
    return latest_date, pd.to_numeric(latest, errors="coerce").dropna().sort_values(ascending=False)


def prediction_score_series(pred: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        if "score" not in pred.columns:
            raise ValueError("预测结果缺少 score 字段")
        pred = pred["score"]
    return pd.to_numeric(pred, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def load_current_positions(strategy: str, state_dir: Path = POSITION_STATE_DIR) -> pd.DataFrame:
    """Load the position snapshot saved by the previous successful workflow run."""
    path = state_dir / f"{strategy}_current.pkl"
    if not path.exists():
        return pd.DataFrame(columns=POSITION_COLUMNS)
    positions = pd.read_pickle(path)
    if not isinstance(positions, pd.DataFrame):
        raise ValueError(f"持仓状态文件格式错误: {path}")
    if "amount" not in positions.columns and "weight" in positions.columns:
        positions["amount"] = pd.to_numeric(positions["weight"], errors="coerce") * INITIAL_CAPITAL
    missing = set(POSITION_COLUMNS).difference(positions.columns)
    if missing:
        raise ValueError(f"持仓状态文件缺少字段 {sorted(missing)}: {path}")
    return positions.loc[:, POSITION_COLUMNS].copy()


def _position_row(
    strategy: str,
    instrument: str,
    entry_date: object,
    trade_date: pd.Timestamp,
    holding_days: int,
    score: float,
    rank: int,
    topk: int,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "instrument": instrument,
        "entry_date": pd.Timestamp(entry_date).normalize(),
        "last_updated": trade_date,
        "holding_days": int(holding_days),
        "score": float(score),
        "rank": int(rank),
        "weight": 1.0 / topk,
        "amount": INITIAL_CAPITAL / topk,
    }


def _action_row(position: dict[str, object], trade_date: pd.Timestamp, action: str, reason: str) -> dict[str, object]:
    return {
        "strategy": position["strategy"],
        "trade_date": trade_date,
        "action": action,
        "instrument": position["instrument"],
        "entry_date": position["entry_date"],
        "holding_days": position["holding_days"],
        "score": position["score"],
        "rank": position["rank"],
        "amount": position["amount"],
        "reason": reason,
    }


def update_fixed_horizon_positions(
    previous: pd.DataFrame,
    scores: pd.Series,
    trade_date: object,
    strategy: str,
    holding_period: int,
    topk: int = RECOMMEND_TOPK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll a fixed 1D/5D paper portfolio forward from its recorded position state."""
    trade_date = pd.Timestamp(trade_date).normalize()
    ranked = scores.dropna().sort_values(ascending=False)
    ranks = ranked.rank(method="first", ascending=False).astype(int)
    current_rows: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    sold_today: set[str] = set()

    for old in previous.to_dict("records"):
        instrument = str(old["instrument"])
        same_day = pd.Timestamp(old["last_updated"]).normalize() == trade_date
        holding_days = int(old["holding_days"]) + (0 if same_day else 1)
        score = float(ranked.get(instrument, np.nan))
        rank = int(ranks.get(instrument, len(ranked) + 1))
        position = _position_row(
            strategy, instrument, old["entry_date"], trade_date, holding_days, score, rank, topk
        )
        if not same_day and holding_days > holding_period:
            sold_today.add(instrument)
            actions.append(_action_row(position, trade_date, "SELL", f"已完成固定持有期 {holding_period} 天"))
        else:
            current_rows.append(position)
            actions.append(_action_row(position, trade_date, "HOLD", f"固定持有期 {holding_period} 天内继续持有"))

    held = {str(row["instrument"]) for row in current_rows}
    for instrument, score in ranked.items():
        instrument = str(instrument)
        if len(current_rows) >= topk:
            break
        if instrument in held or instrument in sold_today:
            continue
        position = _position_row(
            strategy, instrument, trade_date, trade_date, 1, float(score), int(ranks[instrument]), topk
        )
        current_rows.append(position)
        actions.append(_action_row(position, trade_date, "BUY", f"进入最新综合排名前 {topk}"))
        held.add(instrument)

    return pd.DataFrame(current_rows, columns=POSITION_COLUMNS), pd.DataFrame(actions, columns=ACTION_COLUMNS)


def update_adaptive_positions(
    previous: pd.DataFrame,
    scores: pd.Series,
    trade_date: object,
    topk: int = RECOMMEND_TOPK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update positions using signal deterioration and holding-day based exits."""
    trade_date = pd.Timestamp(trade_date).normalize()
    ranked = scores.dropna().sort_values(ascending=False)
    ranks = ranked.rank(method="first", ascending=False).astype(int)
    current_rows: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    sold_today: set[str] = set()

    for old in previous.to_dict("records"):
        instrument = str(old["instrument"])
        same_day = pd.Timestamp(old["last_updated"]).normalize() == trade_date
        holding_days = int(old["holding_days"]) + (0 if same_day else 1)
        score = float(ranked.get(instrument, np.nan))
        rank = int(ranks.get(instrument, len(ranked) + 1))
        position = _position_row(
            ADAPTIVE_STRATEGY_NAME,
            instrument,
            old["entry_date"],
            trade_date,
            holding_days,
            score,
            rank,
            topk,
        )

        sell_reason = None
        if not np.isfinite(score):
            sell_reason = "最新交易日没有有效评分"
        elif score < ADAPTIVE_EMERGENCY_EXIT_SCORE:
            sell_reason = f"综合评分 {score:.3f} 低于紧急退出线 {ADAPTIVE_EMERGENCY_EXIT_SCORE:.2f}"
        elif not same_day and holding_days > ADAPTIVE_MAX_HOLD_DAYS:
            sell_reason = f"持有 {holding_days} 天，超过最长持有期 {ADAPTIVE_MAX_HOLD_DAYS} 天"
        elif holding_days >= ADAPTIVE_MIN_HOLD_DAYS and (
            rank > ADAPTIVE_EXIT_RANK or score < ADAPTIVE_EXIT_SCORE
        ):
            sell_reason = (
                f"持有已满 {ADAPTIVE_MIN_HOLD_DAYS} 天且排名/评分转弱："
                f"rank={rank}, score={score:.3f}"
            )

        if sell_reason is not None:
            sold_today.add(instrument)
            actions.append(_action_row(position, trade_date, "SELL", sell_reason))
        else:
            current_rows.append(position)
            actions.append(_action_row(position, trade_date, "HOLD", "尚未触发动态退出条件"))

    held = {str(row["instrument"]) for row in current_rows}
    for instrument, score in ranked.items():
        instrument = str(instrument)
        if len(current_rows) >= topk:
            break
        if instrument in held or instrument in sold_today:
            continue
        position = _position_row(
            ADAPTIVE_STRATEGY_NAME,
            instrument,
            trade_date,
            trade_date,
            1,
            float(score),
            int(ranks[instrument]),
            topk,
        )
        current_rows.append(position)
        actions.append(_action_row(position, trade_date, "BUY", f"综合评分进入前 {topk}"))
        held.add(instrument)

    return pd.DataFrame(current_rows, columns=POSITION_COLUMNS), pd.DataFrame(actions, columns=ACTION_COLUMNS)


def persist_position_state(
    strategy: str,
    current: pd.DataFrame,
    actions: pd.DataFrame,
    state_dir: Path = POSITION_STATE_DIR,
) -> tuple[Path, Path]:
    """Persist the latest snapshot and an idempotent per-trade-date action history."""
    state_dir.mkdir(parents=True, exist_ok=True)
    current_path = state_dir / f"{strategy}_current.pkl"
    history_path = state_dir / f"{strategy}_history.pkl"
    current.to_pickle(current_path)

    if history_path.exists():
        history = pd.read_pickle(history_path)
        if not history.empty and not actions.empty:
            history_dates = pd.to_datetime(history["trade_date"]).dt.normalize()
            new_actions = actions.loc[~pd.to_datetime(actions["trade_date"]).dt.normalize().isin(history_dates.unique())]
        else:
            new_actions = actions
        history = pd.concat([history, new_actions], ignore_index=True)
    else:
        history = actions.copy()
    history.to_pickle(history_path)
    return current_path, history_path


def combine_adaptive_scores(scores_by_horizon: dict[str, pd.Series]) -> pd.Series:
    required = {"1D_Short_Term", "5D_Mid_Term"}
    missing = required.difference(scores_by_horizon)
    if missing:
        raise ValueError(f"动态退出策略缺少预测结果: {sorted(missing)}")
    combined = pd.concat(
        {
            "1D": scores_by_horizon["1D_Short_Term"],
            "5D": scores_by_horizon["5D_Mid_Term"],
        },
        axis=1,
    ).dropna()
    return (
        combined["1D"] * ADAPTIVE_1D_WEIGHT
        + combined["5D"] * ADAPTIVE_5D_WEIGHT
    ).sort_values(ascending=False)


def evaluate_adaptive_history(
    score_history: pd.Series,
    forward_returns: pd.Series,
    topk: int = RECOMMEND_TOPK,
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Backtest the adaptive exit rules with equal weights and next-day realized returns."""
    def normalize_index(series: pd.Series, context: str) -> pd.Series:
        required_levels = {"datetime", "instrument"}
        if not isinstance(series.index, pd.MultiIndex) or not required_levels.issubset(series.index.names):
            raise ValueError(f"{context}必须使用包含 datetime 和 instrument 的 MultiIndex")
        return series.reorder_levels(["datetime", "instrument"]).sort_index()

    score_history = normalize_index(score_history, "动态策略历史评分")
    forward_returns = normalize_index(forward_returns, "未来收益")

    aligned = pd.concat(
        [score_history.rename("score"), forward_returns.rename("forward_return")],
        axis=1,
    ).dropna()
    ic_values = {}
    report_rows = []
    previous = pd.DataFrame(columns=POSITION_COLUMNS)

    for trade_date in sorted(aligned.index.get_level_values("datetime").unique()):
        daily = aligned.xs(trade_date, level="datetime")
        if len(daily) < 2:
            continue
        daily_scores = daily["score"].sort_values(ascending=False)
        ic_values[pd.Timestamp(trade_date)] = daily["score"].corr(daily["forward_return"])
        current, actions = update_adaptive_positions(previous, daily_scores, trade_date, topk=topk)
        held_returns = daily["forward_return"].reindex(current["instrument"]).dropna()
        gross_return = held_returns.mean() if not held_returns.empty else 0.0
        buy_count = int((actions["action"] == "BUY").sum())
        sell_count = int((actions["action"] == "SELL").sum())
        cost = buy_count / topk * open_cost + sell_count / topk * close_cost
        report_rows.append({"datetime": pd.Timestamp(trade_date), "return": gross_return, "cost": cost})
        previous = current

    ic = pd.Series(ic_values, name="IC", dtype="float64").sort_index().dropna()
    if not report_rows:
        raise ValueError("动态策略没有足够的历史评分和未来收益用于回测")
    report = pd.DataFrame(report_rows).set_index("datetime").sort_index()
    risk_metrics = calculate_risk_metrics(report)
    return ic, report, risk_metrics

def run_dual_research():
    today, yesterday = get_data_dates()

    # yesterday = "20260730"
    # today = "20260731"

    # 🌟 极致优化 1：启动高级物理缓存组件
    provider_uri = EXAMPLES_DIR / "data" / today / "qlib_data"
    qlib.init(
        provider_uri=str(provider_uri),
        region=REG_CN,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": MLRUNS_DIR.as_uri(),
                "default_exp_name": "Experiment",
            },
        },
    )
    
    cpo_stock_list = D.instruments(market=TECH_UNIVERSE)

    # 🌟 保持你原本完全能跑通的时序公式定义（不带空格）
    horizons = {
        "1D_Short_Term": "Ref($close,-2)/Ref($close,-1)-1",
        "5D_Mid_Term": "Ref($close,-6)/Ref($close,-1)-1"
    }

    model_params = {
        "class": LGBLambdaModel,
        "kwargs": {
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 64,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "lambda_l1": 5.0,
            "lambda_l2": 15.0,
            "num_threads": 12,
            "num_iterations": 800,
            "early_stopping_rounds": 50,
        },
    }

    scores_by_horizon: dict[str, pd.Series] = {}
    score_history_by_horizon: dict[str, pd.Series] = {}
    latest_trade_dates: dict[str, pd.Timestamp] = {}

    for name, label_formula in horizons.items():
        print(f"\n{'='*20} 开始处理 {name} 任务 {'='*20}")
        
        task_config = {
            "model": model_params,
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": "2020-01-01",
                            "end_time": f"{today}",
                            "fit_start_time": "2020-01-01",
                            "fit_end_time": "2024-12-31",
                            "instruments": TECH_UNIVERSE,
                            "infer_processors": [
                                # 🌟 核心修复：彻底拿掉高危的 DropCol 处理器！
                                # 这样数据框最顶层的特征和标签多层索引结构就会保持绝对纯净。
                                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                                # 特征截面中性化，斩断科技股大盘集体暴涨暴跌的扰动
                                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}}
                            ],
                            "learn_processors": [
                                # 🌟 此时 Pandas 可以 100% 顺畅找到 "label" 组名，完美避开 KeyError！
                                {"class": "DropnaLabel"},
                                # 用后置截面处理器对时序标签执行 (Label - CSMean) / CSStd
                                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}
                            ],
                            "label": [label_formula],
                        },
                    },
                    "segments": {
                        "train": ("2020-01-01", "2024-12-31"),
                        "valid": ("2025-01-01", "2025-06-30"),
                        "test": ("2025-07-01", f"{today}"),
                    },
                },
            },
        }

        # 执行实验流程
        with R.start(experiment_name=f"Dual_Horizon_{name}"):
            R.log_params(**flatten_dict(task_config))
            
            model = init_instance_by_config(task_config["model"])
            dataset = init_instance_by_config(task_config["dataset"])
            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})
            
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate(save=True)
            
            pred = recorder.load_object("pred.pkl")
            latest_recommendations = print_latest_recommendations(pred, name)
            recorder.save_objects(**{f"{name}.pkl": latest_recommendations})
            latest_trade_date, score_series = latest_scores(pred)
            scores_by_horizon[name] = score_series
            score_history_by_horizon[name] = prediction_score_series(pred)
            latest_trade_dates[name] = latest_trade_date

            holding_period = 1 if name.startswith("1D") else 5
            previous_positions = load_current_positions(name)
            current_positions, holding_actions = update_fixed_horizon_positions(
                previous_positions,
                score_series,
                latest_trade_date,
                strategy=name,
                holding_period=holding_period,
            )
            _, holding_history_path = persist_position_state(name, current_positions, holding_actions)
            holding_history = pd.read_pickle(holding_history_path)
            recorder.save_objects(
                **{
                    "current_positions.pkl": current_positions,
                    "holding_actions.pkl": holding_actions,
                    "holding_history.pkl": holding_history,
                }
            )
            print_colored_block(
                f"\n{name} 当前策略持仓:\n{current_positions.to_string(index=False)}"
                f"\n{name} 今日持仓动作:\n{holding_actions.to_string(index=False)}"
            )
            print_colored_block(f"\n已完成 {name} 的最新推荐结果保存。")

            sar = SigAnaRecord(recorder)
            sar.generate()
            print_final_results(recorder, name, stage="信号分析")
            
            if "5D" in name or "1D" in name:
                print(f"\n正在对 {name} 进行组合回测分析...")
                port_analysis_config = {
                    "strategy": {
                        "class": "TopkDropoutStrategy",
                        "module_path": "qlib.contrib.strategy.signal_strategy",
                        "kwargs": {
                            "signal": (model, dataset),
                            "topk": RECOMMEND_TOPK,
                            "n_drop": NDROP,
                        },
                    },
                    "backtest": {
                        "start_time": "2025-07-01",
                        "end_time": f"{yesterday}",
                        "account": INITIAL_CAPITAL,
                        "benchmark": cpo_stock_list,
                        "exchange_kwargs": {
                            "freq": "day",
                            "limit_threshold": 0.095,
                            "deal_price": "close",
                            "open_cost": 0.0005,
                            "close_cost": 0.0015,
                        },
                    },
                }

                par = PortAnaRecord(recorder, port_analysis_config, "day")
                par.generate()

                report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
                risk_metrics = calculate_risk_metrics(report)
                ic_series = recorder.load_object("sig_analysis/ic.pkl")
                strategy_metrics = calculate_strategy_metrics(
                    ic_series,
                    risk_metrics,
                    strategy=name,
                    trade_date=latest_trade_date,
                )
                metrics_history = persist_metrics_history(strategy_metrics)
                recorder.save_objects(
                    **{
                        "risk_metrics.pkl": risk_metrics,
                        "strategy_metrics.pkl": strategy_metrics,
                        "metrics_history.pkl": metrics_history,
                    }
                )
                print_colored_block(
                    f"\n{name} 风险调整指标（扣除交易成本）:\n{risk_metrics.to_string()}"
                    f"\n{name} 五项指标快照:\n{strategy_metrics.to_string(index=False)}"
                )
                print_final_results(recorder, name, stage="回测分析")

                summary_lines = [
                    f"task={name}",
                    f"topk={RECOMMEND_TOPK}",
                    f"n_drop={NDROP}",
                    "backtest=enabled",
                ]
                details_lines = [
                    "prediction_preview:",
                    format_dataframe_for_report(pred.head(5)),
                    "latest_recommendations:",
                    format_dataframe_for_report(latest_recommendations),
                    "current_positions:",
                    format_dataframe_for_report(current_positions),
                    "holding_actions:",
                    format_dataframe_for_report(holding_actions),
                    "five_metrics:",
                    format_dataframe_for_report(strategy_metrics),
                ]
                report_metrics = recorder.list_metrics()
                report_metrics.update(
                    {
                        metric: float(strategy_metrics.iloc[-1][metric])
                        for metric in ("Sharpe", "Sortino", "Calmar")
                    }
                )
                report_path = write_run_report(
                    script_name=Path(__file__).name,
                    experiment_name=f"Dual_Horizon_{name}",
                    summary_lines=summary_lines,
                    metrics=report_metrics,
                    details_lines=details_lines,
                    output_dir=Path(__file__).resolve().parent / "reports",
                )
                print_colored_block(f"[{name}] 实验完成！请通过 mlflow ui 查看结果。")
                print_colored_block(f"运行报告已保存至 {report_path}")

    adaptive_scores = combine_adaptive_scores(scores_by_horizon)
    adaptive_score_history = combine_adaptive_scores(score_history_by_horizon)
    adaptive_trade_date = min(latest_trade_dates.values())
    adaptive_previous = load_current_positions(ADAPTIVE_STRATEGY_NAME)
    adaptive_positions, adaptive_actions = update_adaptive_positions(
        adaptive_previous,
        adaptive_scores,
        adaptive_trade_date,
    )
    _, adaptive_history_path = persist_position_state(
        ADAPTIVE_STRATEGY_NAME, adaptive_positions, adaptive_actions
    )
    adaptive_history = pd.read_pickle(adaptive_history_path)
    adaptive_dates = adaptive_score_history.index.get_level_values("datetime")
    adaptive_return_frame = D.features(
        cpo_stock_list,
        [horizons["1D_Short_Term"]],
        start_time=adaptive_dates.min(),
        end_time=adaptive_dates.max(),
        freq="day",
    )
    adaptive_forward_returns = adaptive_return_frame.iloc[:, 0]
    adaptive_ic, adaptive_report, adaptive_risk_metrics = evaluate_adaptive_history(
        adaptive_score_history,
        adaptive_forward_returns,
    )
    adaptive_strategy_metrics = calculate_strategy_metrics(
        adaptive_ic,
        adaptive_risk_metrics,
        strategy=ADAPTIVE_STRATEGY_NAME,
        trade_date=adaptive_trade_date,
    )
    metrics_history = persist_metrics_history(adaptive_strategy_metrics)
    with R.start(experiment_name=f"Dual_Horizon_{ADAPTIVE_STRATEGY_NAME}"):
        R.log_params(
            strategy=ADAPTIVE_STRATEGY_NAME,
            topk=RECOMMEND_TOPK,
            weight_1d=ADAPTIVE_1D_WEIGHT,
            weight_5d=ADAPTIVE_5D_WEIGHT,
            min_hold_days=ADAPTIVE_MIN_HOLD_DAYS,
            max_hold_days=ADAPTIVE_MAX_HOLD_DAYS,
            exit_rank=ADAPTIVE_EXIT_RANK,
            exit_score=ADAPTIVE_EXIT_SCORE,
            emergency_exit_score=ADAPTIVE_EMERGENCY_EXIT_SCORE,
        )
        adaptive_artifacts = {
            "current_positions.pkl": adaptive_positions,
            "holding_actions.pkl": adaptive_actions,
            "holding_history.pkl": adaptive_history,
            f"{ADAPTIVE_STRATEGY_NAME}.pkl": adaptive_positions,
            "adaptive_report.pkl": adaptive_report,
            "risk_metrics.pkl": adaptive_risk_metrics,
            "strategy_metrics.pkl": adaptive_strategy_metrics,
            "metrics_history.pkl": metrics_history,
        }
        R.save_objects(**adaptive_artifacts)
    print_colored_block(
        f"\n{ADAPTIVE_STRATEGY_NAME} 当前策略持仓:\n{adaptive_positions.to_string(index=False)}"
        f"\n{ADAPTIVE_STRATEGY_NAME} 今日持仓动作:\n{adaptive_actions.to_string(index=False)}"
        f"\n{ADAPTIVE_STRATEGY_NAME} 五项指标快照:\n{adaptive_strategy_metrics.to_string(index=False)}"
    )
    adaptive_report_path = write_run_report(
        script_name=Path(__file__).name,
        experiment_name=f"Dual_Horizon_{ADAPTIVE_STRATEGY_NAME}",
        summary_lines=[
            f"task={ADAPTIVE_STRATEGY_NAME}",
            f"topk={RECOMMEND_TOPK}",
            f"weight_1d={ADAPTIVE_1D_WEIGHT}",
            f"weight_5d={ADAPTIVE_5D_WEIGHT}",
            "backtest=adaptive_equal_weight",
        ],
        metrics={
            metric: float(adaptive_strategy_metrics.iloc[-1][metric])
            for metric in ("IC", "ICIR", "Sharpe", "Sortino", "Calmar")
        },
        details_lines=[
            "current_positions:",
            format_dataframe_for_report(adaptive_positions),
            "holding_actions:",
            format_dataframe_for_report(adaptive_actions),
            "five_metrics:",
            format_dataframe_for_report(adaptive_strategy_metrics),
        ],
        output_dir=Path(__file__).resolve().parent / "reports",
    )
    print_colored_block(f"{ADAPTIVE_STRATEGY_NAME} 运行报告已保存至 {adaptive_report_path}")
    positions_by_strategy = {
        strategy: load_current_positions(strategy) for strategy in STRATEGY_NAMES
    }
    print_colored_block(
        format_strategy_console_summary(positions_by_strategy, metrics_history)
    )

if __name__ == "__main__":
    run_dual_research()
