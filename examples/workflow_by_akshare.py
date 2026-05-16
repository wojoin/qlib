# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
workflow_by_akshare.py
-----------------------
多只股票的量化研究 workflow，与 workflow_by_code.py 结构完全一致。

适用场景：本地 qlib_data 中已有 50 只以上股票（由 akshareToBin.py 批量转换）。

与 workflow_by_code.py 的差异：
  1. provider_uri → examples/data/qlib_data（akshareToBin.py 生成）
  2. instruments  → "all"（读取 instruments/all.txt，覆盖全部本地股票）
  3. 时间段        → 根据本地数据实际范围调整（需按实际修改）
  4. benchmark    → 所有本地股票的等权日均收益（替代 CSI300）
  5. topk / n_drop → 根据股票数量自动调整（见下方 AUTO_TOPK）

如果本地只有少量股票（< 10 只），请改用 workflow_by_akshare_single.py。

添加更多股票数据：
  python scripts/akshareToBin.py \
      --src /path/to/your/parquet/dir \
      --qlib_dir examples/data/qlib_data

运行方法：
  python examples/workflow_by_akshare.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# ------------------------------------------------------------------ #
# 1. qlib 初始化
# ------------------------------------------------------------------ #
PROVIDER_URI = str(Path(__file__).parent / "data" / "qlib_data")
qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

# ------------------------------------------------------------------ #
# 2. 动态获取本地股票数量，自动设置 topk
#    topk 通常设为股票总数的 10%~20%，最少 5 只
# ------------------------------------------------------------------ #
all_instruments = D.instruments("all")
all_stock_list  = D.list_instruments(all_instruments, as_list=True)
n_stocks = len(all_stock_list)
AUTO_TOPK  = max(5, n_stocks // 10)
AUTO_NDROP = max(1, AUTO_TOPK // 5)

print(f"Local instruments: {n_stocks} stocks  →  topk={AUTO_TOPK}, n_drop={AUTO_NDROP}")

# ------------------------------------------------------------------ #
# 3. 任务配置
#    Handler: Alpha158（158 个截面特征）+ CSZScoreNorm（截面标准化）
#    需要 ≥ 50 只股票才能让截面特征有意义
# ------------------------------------------------------------------ #
AKSHARE_TASK = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 4,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    # "all" 读取 instruments/all.txt 中的全部本地股票
                    "instruments": "all",
                    # ⚠️ 根据你的实际数据范围修改以下时间
                    "start_time": "2020-01-01",
                    "end_time":   "2026-04-30",
                    "fit_start_time": "<dataset.kwargs.segments.train.0>",
                    "fit_end_time":   "<dataset.kwargs.segments.train.1>",
                },
            },
            "segments": {
                # ⚠️ 根据你的数据范围调整（建议训练集 ≥ 3 年）
                "train": ("2020-01-01", "2022-12-31"),
                "valid": ("2023-01-01", "2023-12-31"),
                "test":  ("2024-01-01", "2026-04-30"),
            },
        },
    },
}

# ------------------------------------------------------------------ #
# 4. 回测配置
#    benchmark: 所有本地股票的等权日均收益（替代 CSI300）
#    topk: 自动根据股票总数计算
# ------------------------------------------------------------------ #
PORT_ANALYSIS_CONFIG = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",   # PortAnaRecord 自动替换为 pred.pkl
            "topk":   AUTO_TOPK,
            "n_drop": AUTO_NDROP,
        },
    },
    "backtest": {
        "start_time": None,        # 从 pred.pkl 自动推断
        "end_time":   None,        # 从 pred.pkl 自动推断并后退一天
        "account": 100_000_000,   # 初始资金 1 亿
        # 所有本地股票的等权平均收益作为基准（替代 CSI300）
        "benchmark": all_stock_list,
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
# 主流程
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    # 2. 实例化模型和数据集
    model   = init_instance_by_config(AKSHARE_TASK["model"])
    dataset = init_instance_by_config(AKSHARE_TASK["dataset"])

    # 4. 预览训练集特征矩阵（可选）
    example_df = dataset.prepare("train")
    print(f"\nTrain set shape: {example_df.shape}")
    print(example_df.head())

    with R.start(experiment_name="workflow_akshare"):

        # 5.1 记录超参数
        R.log_params(**flatten_dict(AKSHARE_TASK))

        # 5.2 训练 LightGBM（需要多股票的截面数据才能收敛）
        model.fit(dataset)

        # 5.3 保存模型
        R.save_objects(**{"params.pkl": model})

        # 5.4 生成预测信号
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 5.5 信号质量分析
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 5.6 组合回测
        par = PortAnaRecord(recorder, PORT_ANALYSIS_CONFIG, "day")
        par.generate()
