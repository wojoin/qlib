from pathlib import Path

import qlib
import pandas as pd
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.model.base import Model

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.model.base import Model

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.model.base import Model

# class LGBLambdaModel(Model):
#     def __init__(self, **kwargs):
#         self.params = kwargs
#         self.model = None

#     def _prepare_data_for_rank(self, dataset, segment):
#         """核心校准：精准对齐截面标准化后的 Label 档位"""
#         df = dataset.prepare(segment, col_set=["feature", "label"])
        
#         # 确保时间连续
#         df = df.sort_index(level="datetime")
        
#         x = df["feature"]
#         raw_y = df["label"].iloc[:, 0]
        
#         # 🌟 校准 1：显式指定 ascending=True，但将档位反转，让涨幅最大的股票（Z-Score最大）分到最高的级别 9
#         # LightGBM LambdaRank 的 Label 越大，代表相关性越高（越应该排在前面）
#         y_pct = raw_y.groupby(level="datetime").rank(pct=True, ascending=True)
#         y_int = (y_pct * 9).fillna(0).astype(int)
        
#         # 计算截面 group 数量
#         group = df.groupby(level="datetime").size().values
        
#         return x, y_int, group

#     def fit(self, dataset, **kwargs):
#         x_train, y_train, group_train = self._prepare_data_for_rank(dataset, "train")
#         x_valid, y_valid, group_valid = self._prepare_data_for_rank(dataset, "valid")
        
#         train_dataset = lgb.Dataset(x_train, label=y_train, group=group_train)
#         valid_dataset = lgb.Dataset(x_valid, label=y_valid, group=group_valid, reference=train_dataset)
        
#         params = {
#             "objective": "lambdarank",
#             "metric": "ndcg",
#             "ndcg_eval_at": [5, 10],
#             "learning_rate": 0.05,
#             "max_depth": 6,
#             "num_leaves": 64,
#             "num_threads": 12,
#             "verbosity": -1,
#             # 🌟 校准 2：显式指定 LambdaRank 的增益权重，强化最高档位（第9档，即最头部的科技股）的惩罚力度
#             "label_gain": [0, 1, 2, 3, 4, 15, 30, 60, 120, 250] 
#         }
#         params.update(self.params)
        
#         num_iterations = params.pop("num_iterations", 800)
#         early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        
#         callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if early_stopping_rounds else []
        
#         self.model = lgb.train(
#             params,
#             train_set=train_dataset,
#             num_boost_round=num_iterations,
#             valid_sets=[valid_dataset],
#             callbacks=callbacks
#         )

#     def predict(self, dataset, **kwargs):
#         if self.model is None:
#             raise ValueError("模型尚未进行训练！")
        
#         df_test = dataset.prepare("test", col_set="feature")
#         preds = self.model.predict(df_test)
        
#         # 🌟 返回标准结果
#         return pd.Series(preds, index=df_test.index)


import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.model.base import Model

class LGBLambdaModel(Model):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    def _prepare_data_for_rank(self, dataset, segment):
        """标准截面数据切片与档位转换"""
        df = dataset.prepare(segment, col_set=["feature", "label"])
        df = df.sort_index(level="datetime")
        
        x = df["feature"]
        raw_y = df["label"].iloc[:, 0]
        
        # 严格将连续收益率平滑映射到 0~9 档的非负整数
        y_pct = raw_y.groupby(level="datetime").rank(pct=True, ascending=True)
        y_int = (y_pct * 9).fillna(0).astype(int)
        
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
            "label_gain": [0, 1, 2, 3, 4, 15, 30, 60, 120, 250]
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
        # 得到原生 LambdaRank 的原始概率分
        raw_preds = self.model.predict(df_test)
        
        pred_series = pd.Series(raw_preds, index=df_test.index)
        
        # 🌟 关键解毒步：将无量纲的原始分，强制转换为“每天截面内部的百分比排名”
        # 这样做能彻底消除跨期数值漂移，将其强制驯化为 Qlib 策略最喜欢的 0.0 ~ 1.0 单调性得分
        final_score = pred_series.groupby(level="datetime").rank(pct=True)
        
        return final_score


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
    cpo_stock_list = D.instruments(market=TECH_UNIVERSE)

    # 定义双周期任务：1-day 和 5-day
    horizons = {
        "1D_Short_Term": "Ref($close, -2) / Ref($close, -1) - 1",
        # "5D_Mid_Term": "Ref($close, -6) / Ref($close, -1) - 1"
        # 升级为截面中性化（减去当天这 605 只股票的平均涨幅，只赚取超越科技股平均水平的超额收益）：
        "5D_Mid_Term": ["(Ref($close, -5) / Ref($close, -1) - 1) - CSMean(Ref($close, -5) / Ref($close, -1) - 1)"]
    }

    # 公用的模型参数 (LightGBM)
    # model_params = {
    #     "class": "LGBModel",
    #     "module_path": "qlib.contrib.model.gbdt",
    #     # "kwargs": {
    #     #     "loss": "mse",
    #     #     "colsample_bytree": 0.8879,
    #     #     "learning_rate": 0.2,
    #     #     "subsample": 0.8789,
    #     #     "lambda_l1": 205.6,
    #     #     "lambda_l2": 580.9,
    #     #     "max_depth": 8,
    #     #     "num_leaves": 210,
    #     #     "num_threads": 12, # 充分利用 M5 Pro 核心
    #     # },
    #     "kwargs": {
    #         "loss": "mse",
    #         "colsample_bytree": 0.85,
    #         "learning_rate": 0.05,    # 🌟 降低学习率（从0.2降到0.05），让模型学得更稳
    #         "subsample": 0.85,
    #         "lambda_l1": 10.0,        # 🌟 科技股行业集中，不需要全市场那么恐怖的 L1 正则，大幅调低它
    #         "lambda_l2": 50.0,        # 🌟 适当降低 L2 正则，松绑模型对强趋势科技股的拟合
    #         "max_depth": 6,           # 🌟 限制树深（6层足够），防止过拟合
    #         "num_leaves": 64,         # 🌟 叶子数与树深配套（2^6=64），保证基础泛化
    #         "num_threads": 12,
    #     },
    # }

    model_params = {
        "class": LGBLambdaModel,  # 🌟 注意：直接传入类名对象，千万不要加引号变成字符串！
        # "module_path": ...     # 🌟 注意：这一行直接删掉，不需要了！
        "kwargs": {
            "learning_rate": 0.05,             # 降低学习率，排序梯度更平稳
            "max_depth": 6,                    # 限制树深，防止科技股小样本过拟合
            "num_leaves": 64,                  # 叶子节点数与树深配套
            "min_data_in_leaf": 30,            # 保证分裂点具备统计学意义
            "num_iterations": 800,             # 给树模型充分组合因子的空间
            "early_stopping_rounds": 50,       # 连续 50 步 NDCG 不提升则提前停止
            "subsample": 0.85,                 # 随机行抽样，对抗科技股高波动
            "colsample_bytree": 0.80,          # 随机列抽样
            "lambda_l1": 5.0,                  # 大幅松绑的 L1 正则
            "lambda_l2": 15.0,                 # 大幅松绑的 L2 正则
            "num_threads": 12,                 # 充分跑满 M5 Pro 的核心
            "verbosity": -1,                   # 关闭冗长日志
        },
    }

    for name, label_formula in horizons.items():
        print(f"\n{'='*20} 开始处理 {name} 任务 {'='*20}")
        
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
                        "account": 10000000,
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
