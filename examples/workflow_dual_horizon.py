from pathlib import Path

import qlib
import pandas as pd
from termcolor import colored
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

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

TECH_UNIVERSE = "all"
RECOMMEND_TOPK = 5
NDROP = 2



def run_dual_research():
    today, yesterday = get_data_dates()

    # 初始化 Qlib
    # qlib.init(provider_uri="~/.qlib/qlib_data/my_cpo_data", region=REG_CN)
    qlib.init(provider_uri=f"~/ai/qlib/examples/data/{today}/qlib_data", region="cn")


    # 初始化 Qlib - 注入 M5 Pro 专属极致缓存
    # qlib.init(
    #     provider_uri=f"~/qlib/examples/data/{today}/qlib_data",
    #     region="cn",
    #     mem_cache_size_limit=0,   # 0 代表无上限！48G 内存足够把 605 只股的 5 年因子全塞进 RAM，二次运行秒开
    #     expression_cache=True,    # 开启表达式缓存，避免 1D 和 5D 任务重复计算相同的 Alpha 基础因子
    # )
    
    # 自动获取当前池子中的所有股票作为 benchmark 参考
    ai_stock_list = D.instruments(market=TECH_UNIVERSE)

    # 定义双周期任务：1-day 和 5-day
    horizons = {
        "1D_Short_Term": "Ref($close, -2) / Ref($close, -1) - 1",
        "5D_Mid_Term": "Ref($close, -6) / Ref($close, -1) - 1"
        # 升级为截面中性化（减去当天这 605 只股票的平均涨幅，只赚取超越科技股平均水平的超额收益）：
        # "5D_Mid_Term": "(Ref($close, -5) / Ref($close, -1) - 1) - CSMean(Ref($close, -5) / Ref($close, -1) - 1)"
    }

    # 公用的模型参数 (LightGBM)
    model_params = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        # "kwargs": {
        #     "loss": "mse",
        #     "colsample_bytree": 0.8879,
        #     "learning_rate": 0.2,
        #     "subsample": 0.8789,
        #     "lambda_l1": 205.6,
        #     "lambda_l2": 580.9,
        #     "max_depth": 8,
        #     "num_leaves": 210,
        #     "num_threads": 12, # 充分利用 M5 Pro 核心
        # },
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.85,
            "learning_rate": 0.05,    # 🌟 降低学习率（从0.2降到0.05），让模型学得更稳
            "subsample": 0.85,
            "lambda_l1": 10.0,        # 🌟 科技股行业集中，不需要全市场那么恐怖的 L1 正则，大幅调低它
            "lambda_l2": 50.0,        # 🌟 适当降低 L2 正则，松绑模型对强趋势科技股的拟合
            "max_depth": 6,           # 🌟 限制树深（6层足够），防止过拟合
            "num_leaves": 64,         # 🌟 叶子数与树深配套（2^6=64），保证基础泛化
            "num_threads": 12,
    },
    }

    for name, label_formula in horizons.items():
        print(colored(f"\n{'='*20} 开始处理 {name} 任务 {'='*20}", "red", attrs=["bold"]))
        
        task_config = {
            "model": model_params,
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158", # 使用完整 Alpha158 数据处理器
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": "2020-01-01",
                            "end_time": f"{today}",
                            "fit_start_time": "2020-01-01",
                            "fit_end_time": "2024-12-31",
                            "instruments": TECH_UNIVERSE,
                            "infer_processors": [
                                {"class": "DropCol", "kwargs": {"col_list": ["Ref($close, -1)/$close - 1"]}},
                                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                                {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
                            ],
                            "learn_processors": [
                                {"class": "DropnaLabel"},
                                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}
                            ],
                            # 5-day label: buy at T+1 close and sell at T+6 close.
                            "label": [label_formula],
                        },
                    },
                    "segments": {
                        "train": ("2020-01-01", "2024-12-31"),
                        "valid": ("2025-01-01", "2025-06-30"),
                        "test": ("2025-07-01", f"{today}"), # 包含你提到的最新数据点
                    },
                },
            },
        }

        # 执行实验流程
        with R.start(experiment_name=f"Dual_Horizon_{name}"):
            R.log_params(**flatten_dict(task_config))
            
            # 1. 训练
            model = init_instance_by_config(task_config["model"])
            dataset = init_instance_by_config(task_config["dataset"])
            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})
            
            # 2. 预测与指标分析 (IC/Rank IC/ICIR/Rank ICIR)并保存
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate(save=True)
            
            # 3. 打印最新推荐结果
            pred = recorder.load_object("pred.pkl")
            latest_recommendations = print_latest_recommendations(pred, name)
            recorder.save_objects(**{f"{name}.pkl": latest_recommendations})
            print(f"\n已完成 {name} 的最新推荐结果保存。")

            sar = SigAnaRecord(recorder)
            sar.generate()
            print_final_results(recorder, name, stage="信号分析")
            
            # 4. 回测分析 (仅 5D 任务运行回测以节省时间，1D 仅看信号)
            if "5D" in name or "1D" in name:
                print(f"\n正在对 {name} 进行组合回测分析...")
                # 简单回测配置
                port_analysis_config = {
                    "strategy": {
                        "class": "TopkDropoutStrategy",
                        "module_path": "qlib.contrib.strategy.signal_strategy",
                        "kwargs": {
                            "signal": (model, dataset),
                            "topk": RECOMMEND_TOPK, # 仅持有行业内得分最高的 5 只股
                            "n_drop": NDROP,
                        },
                    },
                    "backtest": {
                        "start_time": "2025-07-01",
                        # 回测需要访问下一交易日来生成交易区间，不能设置为本地日历最后一天
                        "end_time": f"{yesterday}",
                        "account": 300000,
                        "benchmark": ai_stock_list,
                        "exchange_kwargs": {
                            "freq": "day",
                            "limit_threshold": 0.095,
                            "deal_price": "close",
                            "open_cost": 0.0005,
                            "close_cost": 0.0015,
                        },
                    },
                }

                # 回测分析
                par = PortAnaRecord(recorder, port_analysis_config, "day")
                par.generate()
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
                ]
                report_path = write_run_report(
                    script_name=Path(__file__).name,
                    experiment_name=f"Dual_Horizon_{name}",
                    summary_lines=summary_lines,
                    metrics=recorder.list_metrics(),
                    details_lines=details_lines,
                    output_dir=Path(__file__).resolve().parent / "reports",
                )
                print_colored_block(f"[{name}] 实验完成！请通过 mlflow ui 查看结果。")
                print_colored_block(f"运行报告已保存至 {report_path}")

if __name__ == "__main__":
    run_dual_research()
