from pathlib import Path

import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.model.base import Model
# 🌟 核心引入：引入 Alpha158 的时序特征字典，以及干净的通用处理器基类
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP
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
        df = dataset.prepare(segment, col_set=["feature", "label"])
        df = df.sort_index(level="datetime")
        
        x = df["feature"]
        raw_y = df["label"].iloc[:, 0]
        
        # 将标准收益率映射到 0-9 档的非负整数，用于 NDCG 排序优化
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
        raw_preds = self.model.predict(df_test)
        
        pred_series = pd.Series(raw_preds, index=df_test.index)
        final_score = pred_series.groupby(level="datetime").rank(pct=True)
        return final_score


TECH_UNIVERSE = "all"
RECOMMEND_TOPK = 5
NDROP = 2

def run_dual_research():
    today, yesterday = get_data_dates()

    qlib.init(
        provider_uri=f"~/qlib/examples/data/{today}/qlib_data",
        region="cn"
    )
    
    cpo_stock_list = D.instruments(market=TECH_UNIVERSE)

    # 🌟 统一公式写法（无空格标准时序表达式）
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

    # 🌟 绝招：直接把 Alpha158 经典的 158 个时序特征定义拉出来，丢给通用的 DataHandlerLP
    alpha158_instance = Alpha158(instruments=TECH_UNIVERSE, fit_start_time="2020-01-01", fit_end_time="2024-12-31")
    alpha158_pure_features = alpha158_instance.get_feature_config()

    for name, label_formula in horizons.items():
        print(f"\n{'='*20} 开始处理 {name} 任务 {'='*20}")
        
        task_config = {
            "model": model_params,
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        # 🌟 放弃 Alpha158 类，改用绝对纯净的 DataHandlerLP
                        "class": "DataHandlerLP",
                        "module_path": "qlib.data.dataset.handler",
                        "kwargs": {
                            "start_time": "2020-01-01",
                            "end_time": f"{today}",
                            "fit_start_time": "2020-01-01",
                            "fit_end_time": "2024-12-31",
                            "instruments": TECH_UNIVERSE,
                            "infer_processors": [
                                # 🌟 工业级未来函数绝杀法：直接把整个 label 组从推理矩阵中抹去，简单暴击
                                {"class": "DropCol", "kwargs": {"col_list": ["label"]}},
                                {
                                    "class": "RobustZScoreNorm",
                                    "kwargs": {
                                        "fit_start_time": "2020-01-01",
                                        "fit_end_time": "2024-12-31",
                                        "fields_group": "feature",
                                        "clip_outlier": True,
                                    },
                                },
                                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                                # 特征截面中性化：剥离科技行业整体 Beta
                                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}}
                            ],
                            "learn_processors": [
                                {"class": "DropnaLabel"},
                                # 用后置截面处理器对时序标签执行中性化，数学上完美等价于去中心化
                                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}
                            ],
                            "feature": alpha158_pure_features,
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
