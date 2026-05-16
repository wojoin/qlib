import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.config import CSI300_BENCH

# 自定义 CPO 任务配置
CPO_TASK = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 10,  # 你的 M5 Pro 核心多，可以适当增加
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158", # 使用完整 Alpha158 数据处理器，内部会配置 Alpha158DL
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2020-01-01",
                    "end_time": "2026-05-06",
                    "fit_start_time": "2020-01-01",
                    "fit_end_time": "2024-12-31",
                    "instruments": "cpo_sector", # 读取 ~/.qlib/qlib_data/cn_data/instruments/cpo_sector.txt
                },
            },
            "segments": {
                "train": ("2020-01-01", "2024-12-31"),
                "valid": ("2025-01-01", "2025-06-30"),
                "test": ("2025-07-01", "2026-05-06"), # 包含你提到的最新数据点
            },
        },
    },
}

# # 更新后的 CPO 任务配置，集成了进阶量价因子
# CPO_TASK = {
#     "model": {
#         "class": "LGBModel",
#         "module_path": "qlib.contrib.model.gbdt",
#         "kwargs": {
#             "loss": "mse",
#             "colsample_bytree": 0.8879,
#             "learning_rate": 0.0421,
#             "subsample": 0.8789,
#             "lambda_l1": 205.6999,
#             "lambda_l2": 580.9768,
#             "max_depth": 8,
#             "num_leaves": 210,
#             "num_threads": 10,
#         },
#     },
#     "dataset": {
#         "class": "DatasetH",
#         "module_path": "qlib.data.dataset",
#         "kwargs": {
#             "handler": {
#                 "class": "DataHandlerLP",
#                 "module_path": "qlib.data.dataset.handler",
#                 "kwargs": {
#                     "instruments": "cpo_sector",
#                     "start_time": "2020-01-01",
#                     "end_time": "2026-05-06",
#                     "data_loader": {
#                         "class": "QlibDataLoader",
#                         "module_path": "qlib.data.dataset.loader",
#                         "kwargs": {
#                             "config": {
#                                 "feature": (
#                                     [
#                                         "$open/$close",
#                                         "$high/$close",
#                                         "$low/$close",
#                                         "$vwap/$close",
#                                         "Mean($close, 5)/$close",
#                                         "Mean($close, 10)/$close",
#                                         "Mean($close, 20)/$close",
#                                         "Mean($close, 60)/$close",
#                                         "$volume / Ref($volume, 1) - 1",
#                                         "$volume / Mean($volume, 20)",
#                                         "Corr($close, Log($volume + 1), 10)",
#                                         "($close / Ref($close, 5) - 1) / ($volume / Mean($volume, 5))",
#                                         "Std($volume, 20) / Mean($volume, 20)",
#                                     ],
#                                     [
#                                         "OPEN0",
#                                         "HIGH0",
#                                         "LOW0",
#                                         "VWAP0",
#                                         "MA5",
#                                         "MA10",
#                                         "MA20",
#                                         "MA60",
#                                         "VROC1",
#                                         "VOL_RATIO20",
#                                         "PRICE_VOLUME_CORR10",
#                                         "RETURN_PER_VOLUME5",
#                                         "VOL_CV20",
#                                     ],
#                                 ),
#                                 "label": (
#                                     ["Ref($close, -2) / Ref($close, -1) - 1"],
#                                     ["LABEL0"],
#                                 ),
#                             }
#                         },
#                     },
#                     "infer_processors": [
#                         {
#                             "class": "ZScoreNorm",
#                             "kwargs": {
#                                 "fit_start_time": "2020-01-01",
#                                 "fit_end_time": "2024-12-31",
#                                 "fields_group": "feature",
#                             },
#                         },
#                         {"class": "Fillna"},
#                     ],
#                     "learn_processors": [
#                         {"class": "DropnaLabel"},
#                         {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
#                     ],
#                 },
#             },
#             "segments": {
#                 "train": ("2020-01-01", "2024-12-31"),
#                 "valid": ("2025-01-01", "2025-06-30"),
#                 "test": ("2025-07-01", "2026-05-06"),
#             },
#         },
#     },
# }

if __name__ == "__main__":
    # provider_uri = "~/.qlib/qlib_data/cn_data"
    provider_uri = "/Users/joseph/qlib/examples/data/20260506/qlib_data"  # 假设你已经按照前面的说明准备好了这个目录
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 初始化模型和数据集
    model = init_instance_by_config(CPO_TASK["model"])
    dataset = init_instance_by_config(CPO_TASK["dataset"])

    # 针对 CPO 板块调整回测配置
    # 由于 CPO 个股数量较少，我们将 topk 从 50 调低到 5-10
    port_analysis_config = {
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
                "signal": (model, dataset),
                "topk": 5,  # 仅持有行业内得分最高的 5 只股
                "n_drop": 2,
            },
        },
        "backtest": {
            "start_time": "2025-07-01",
            "end_time": "2026-05-06",
            "account": 10000000, # 1000万资金量
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # 开始实验
    with R.start(experiment_name="CPO_Strategy_Research"):
        R.log_params(**flatten_dict(CPO_TASK))
        
        print("正在训练 CPO 行业模型...")
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # 预测与信号分析
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        sar.generate()

        # 回测
        print("开始 CPO 行业历史回测...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        print(f"实验完成！请通过 mlflow ui 查看结果。")
