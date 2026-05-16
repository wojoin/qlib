# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
workflow_by_akshare_single.py
------------------------------
单只股票：LSTM vs GRU 对比实验。

与 LightGBM（Alpha158）的核心区别：
  ┌─────────────────┬──────────────────────────────┬──────────────────────────────┐
  │                 │  LightGBM + Alpha158          │  LSTM / GRU（本脚本）        │
  ├─────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ 特征构造         │ 截面特征（需多股票）            │ 时间序列特征（仅用自身历史）    │
  │ 截面标准化       │ CSZScoreNorm（截面排名）        │ RobustZScoreNorm（时序 Z）   │
  │ 数据集类型       │ DatasetH（每行独立）           │ TSDatasetH（滑动窗口序列）     │
  │ 输入形状         │ [batch, n_feat]               │ [batch, step_len, n_feat]    │
  │ 参数量           │ 树模型，少                     │ 深度学习，多                  │
  └─────────────────┴──────────────────────────────┴──────────────────────────────┘

TSDatasetH 工作原理：
  - 以 step_len=20 天为滑动窗口，每个样本是连续 20 天的特征矩阵
  - label 来自最后一天的 Ref($close,-2)/Ref($close,-1)-1（次日收益）
  - LSTM/GRU 学习 20 天时序模式来预测次日收益

运行方法：
  python examples/workflow_by_akshare_single.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
# PyTorch 2.x multi-threaded LSTM segfaults on macOS Apple Silicon within deep call stacks.
# Single-threaded mode avoids this; for CPU-only training the speed difference is negligible.
torch.set_num_threads(1)

from pathlib import Path
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# ------------------------------------------------------------------ #
# 1. qlib 初始化
# ------------------------------------------------------------------ #
PROVIDER_URI = str(Path(__file__).parent / "data" / "qlib_data")
qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

# ------------------------------------------------------------------ #
# 2. 时间段（两个模型共用，确保对比公平）
# ------------------------------------------------------------------ #
TRAIN_START = "2022-01-27"
TRAIN_END   = "2023-12-29"
VALID_START = "2024-01-02"
VALID_END   = "2024-12-31"
TEST_START  = "2025-01-02"
TEST_END    = "2026-04-30"

# ------------------------------------------------------------------ #
# 3. 纯时间序列特征（不依赖截面，单只股票可用）
#    共 28 个特征，对应模型 d_feat=28
# ------------------------------------------------------------------ #
FEATURE_FIELDS = (
    # --- 价格动量 (5) ---
    ["Ref($close,1)/$close-1",
     "Ref($close,5)/$close-1",
     "Ref($close,10)/$close-1",
     "Ref($close,20)/$close-1",
     "Ref($close,60)/$close-1",
    # --- 均线偏离 (6) ---
     "$close/Mean($close,5)-1",
     "$close/Mean($close,10)-1",
     "$close/Mean($close,20)-1",
     "$close/Mean($close,60)-1",
     "Mean($close,5)/Mean($close,20)-1",
     "Mean($close,10)/Mean($close,60)-1",
    # --- 波动率 (5) ---
     "Std($close,5)/$close",
     "Std($close,10)/$close",
     "Std($close,20)/$close",
     "Std($close,60)/$close",
     "($close-Mean($close,20))/Std($close,20)",   # 布林带位置
    # --- 成交量 (5) ---
     "$volume/Mean($volume,5)-1",
     "$volume/Mean($volume,10)-1",
     "$volume/Mean($volume,20)-1",
     "Std($volume,5)/Mean($volume,5)",
     "$change*($volume/Mean($volume,5))",          # 量价配合
    # --- K线形态 (5) ---
     "($high-$low)/$close",                        # 振幅
     "($close-$low)/($high-$low)",                 # 收盘位置
     "($close-$open)/$close",                      # 实体
     "($high-Greater($close,$open))/$close",       # 上影线
     "(Less($close,$open)-$low)/$close",           # 下影线
    # --- 换手率 (2) ---
     "$turnover",
     "$turnover/Mean($turnover,10)-1",
    ]
)
FEATURE_NAMES = (
    ["MOM1","MOM5","MOM10","MOM20","MOM60",
     "MA5_DEV","MA10_DEV","MA20_DEV","MA60_DEV","MA5_20","MA10_60",
     "STD5","STD10","STD20","STD60","BBAND20",
     "VOL5_DEV","VOL10_DEV","VOL20_DEV","VOL_STD5","VP_CORR",
     "AMPLITUDE","CLOSE_POS","BODY","UPPER_SHADOW","LOWER_SHADOW",
     "TURN","TURN10_DEV",
    ]
)
D_FEAT = len(FEATURE_NAMES)   # 28

LABEL_FIELDS = ["Ref($close,-2)/Ref($close,-1)-1"]
LABEL_NAMES  = ["LABEL0"]

# ------------------------------------------------------------------ #
# 4. 共享 DataHandlerLP 配置（LSTM 和 GRU 使用完全相同的特征和标准化）
#    RobustZScoreNorm：用 median/IQR 代替 mean/std，对股价尖峰厚尾更鲁棒
# ------------------------------------------------------------------ #
HANDLER_CONFIG = {
    "class": "DataHandlerLP",
    "module_path": "qlib.data.dataset.handler",
    "kwargs": {
        "instruments": ["SZ301217"],
        "start_time": TRAIN_START,
        "end_time":   TEST_END,
        "data_loader": {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (FEATURE_FIELDS, FEATURE_NAMES),
                    "label":   (LABEL_FIELDS, LABEL_NAMES),
                },
                "freq": "day",
            },
        },
        "learn_processors": [
            {"class": "DropnaLabel"},
            {
                "class": "RobustZScoreNorm",
                "kwargs": {
                    "fit_start_time": TRAIN_START,
                    "fit_end_time":   TRAIN_END,
                    "fields_group": "feature",
                    "clip_outlier": True,
                },
            },
            {"class": "Fillna"},
        ],
        "infer_processors": [
            {
                "class": "RobustZScoreNorm",
                "kwargs": {
                    "fit_start_time": TRAIN_START,
                    "fit_end_time":   TRAIN_END,
                    "fields_group": "feature",
                    "clip_outlier": True,
                },
            },
            {"class": "Fillna"},
        ],
    },
}

# TSDatasetH：将 Handler 输出的 DataFrame 切割成滑动窗口序列
# step_len=20：每个训练样本是连续 20 天的特征矩阵，预测第 20 天的次日收益
DATASET_CONFIG = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": HANDLER_CONFIG,
        "segments": {
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test":  (TEST_START,  TEST_END),
        },
        "step_len": 20,
    },
}

# ------------------------------------------------------------------ #
# 5. 模型配置
#    GPU=-1  → 强制 CPU（无 CUDA；Apple M 系列芯片的 MPS 暂不被 qlib 自动识别）
#    batch_size=64  → 单股票训练样本少（~450），小 batch 避免梯度噪声过大
#    n_epochs=100   → 配合 early_stop=20，足够收敛
# ------------------------------------------------------------------ #
LSTM_CONFIG = {
    "class": "LSTM",
    "module_path": "qlib.contrib.model.pytorch_lstm_ts",
    "kwargs": {
        "d_feat":      D_FEAT,   # 必须与 handler 输出的特征数完全一致
        "hidden_size": 64,
        "num_layers":  2,
        "dropout":     0.3,      # 单股票样本少，适当 dropout 防过拟合
        "n_epochs":    100,
        "lr":          1e-3,
        "early_stop":  20,
        "batch_size":  64,
        "metric":      "loss",
        "loss":        "mse",
        "n_jobs":      0,  # macOS 上 DataLoader 用主进程加载（避免 spawn 问题）
        "GPU":         -1,       # -1 = CPU
        "seed":        42,
    },
}

GRU_CONFIG = {
    "class": "GRU",
    "module_path": "qlib.contrib.model.pytorch_gru_ts",
    "kwargs": {
        "d_feat":      D_FEAT,
        "hidden_size": 64,
        "num_layers":  2,
        "dropout":     0.3,
        "n_epochs":    100,
        "lr":          1e-3,
        "early_stop":  20,
        "batch_size":  64,
        "metric":      "loss",
        "loss":        "mse",
        "n_jobs":      0,  # macOS 上 DataLoader 用主进程加载（避免 spawn 问题）
        "GPU":         -1,
        "seed":        42,
    },
}

# ------------------------------------------------------------------ #
# 6. 回测配置（两个模型共用）
# ------------------------------------------------------------------ #
PORT_ANALYSIS_CONFIG = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {"signal": "<PRED>", "topk": 1, "n_drop": 0},
    },
    "backtest": {
        "start_time": None,
        "end_time":   None,
        "account":    1_000_000,
        "benchmark":  "SZ301217",   # 持有不动 vs 策略
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost":  0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

# ------------------------------------------------------------------ #
# 工具函数：运行一个完整实验并返回 IC / 回测指标
# ------------------------------------------------------------------ #
def run_experiment(model_name: str, model_config: dict, dataset_config: dict) -> dict:
    """Run full pipeline for one model, return key metrics."""
    import copy
    print(f"\n{'='*60}")
    print(f"  Running: {model_name}")
    print(f"{'='*60}")

    model = init_instance_by_config(model_config)
    # Fresh dataset per run — avoids internal state leaking between LSTM and GRU
    dataset = init_instance_by_config(dataset_config)
    # Deep-copy config so PortAnaRecord cannot mutate the shared "<PRED>" placeholder
    port_cfg = copy.deepcopy(PORT_ANALYSIS_CONFIG)

    with R.start(experiment_name=f"single_{model_name.lower()}"):
        R.log_params(**flatten_dict(model_config))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()

        # 5.4 预测信号
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 5.5 信号分析
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 5.6 回测
        par = PortAnaRecord(recorder, port_cfg, "day")
        par.generate()
        port = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    # 提取对比指标
    # 单只股票无法计算截面 IC（每天只有 1 个数据点，方差为 0）。
    # 改用时序 IC：将预测分数与实际收益在测试期做整体 Pearson 相关。
    import pandas as pd
    try:
        pred_df = recorder.load_object("pred.pkl")
        # pred.pkl is a DataFrame with (datetime, instrument) index and "score" column
        pred_scores = pred_df.iloc[:, 0]  # extract first column as Series

        label_df = dataset.prepare("test", col_set="label", data_key="infer")
        # label_df may have MultiIndex columns like (instrument, field) → flatten to (field,)
        if isinstance(label_df.columns, pd.MultiIndex):
            label_df = label_df.droplevel(axis=1, level=0)
        label_ser = label_df.iloc[:, 0]

        # Align by index: both have (datetime, instrument) MultiIndex
        merged = pred_scores.rename("pred").to_frame().join(label_ser.rename("label"), how="inner")
        ic_mean = float(merged["pred"].corr(merged["label"])) if len(merged) >= 2 else float("nan")
    except Exception as e:
        print(f"IC computation failed: {e}")
        ic_mean = float("nan")

    # port 是 MultiIndex DataFrame，行索引为 (分析类型, 指标名)，列为 "risk"
    def _port_val(port_df, analysis_type, metric):
        try:
            return float(port_df.loc[(analysis_type, metric), "risk"])
        except Exception:
            return float("nan")

    return {
        "model":              model_name,
        "IC":                 round(ic_mean, 4) if ic_mean == ic_mean else float("nan"),
        "ICIR":               float("nan"),
        "excess_annual_ret":  round(_port_val(port, "excess_return_with_cost", "annualized_return"), 4),
        "excess_IR":          round(_port_val(port, "excess_return_with_cost", "information_ratio"), 4),
        "excess_max_dd":      round(_port_val(port, "excess_return_with_cost", "max_drawdown"), 4),
    }


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    print(f"Feature count (d_feat): {D_FEAT}")
    print(f"Train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"Valid: {VALID_START} ~ {VALID_END}")
    print(f"Test:  {TEST_START}  ~ {TEST_END}")

    results = []
    for name, cfg in [("LSTM", LSTM_CONFIG), ("GRU", GRU_CONFIG)]:
        metrics = run_experiment(name, cfg, DATASET_CONFIG)
        results.append(metrics)

    # ---------------------------------------------------------------- #
    # 对比结果汇总
    # ---------------------------------------------------------------- #
    print("\n" + "="*60)
    print("  LSTM vs GRU — Comparison Summary")
    print("="*60)
    header = f"{'Metric':<25} {'LSTM':>12} {'GRU':>12}"
    print(header)
    print("-" * len(header))

    keys = [
        ("IC",               "IC (预测质量)"),
        ("ICIR",             "ICIR (信号稳定性)"),
        ("excess_annual_ret","超额年化收益"),
        ("excess_IR",        "超额 IR (Sharpe)"),
        ("excess_max_dd",    "超额最大回撤"),
    ]
    lstm_r, gru_r = results[0], results[1]
    for key, label in keys:
        v_lstm = lstm_r[key]
        v_gru  = gru_r[key]
        # 标注胜者（最大回撤越小越好，其余越大越好）
        if key == "excess_max_dd":
            winner = "LSTM ✓" if v_lstm > v_gru else ("GRU  ✓" if v_gru > v_lstm else "tie")
        else:
            winner = "LSTM ✓" if v_lstm > v_gru else ("GRU  ✓" if v_gru > v_lstm else "tie")
        print(f"{label:<25} {str(v_lstm):>12} {str(v_gru):>12}   {winner}")

    print("="*60)
    print("\n详细 artifacts 保存在 examples/mlruns/ 中，可用 mlflow ui 查看。")
