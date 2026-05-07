#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""

import copy
from datetime import datetime
from pathlib import Path

import qlib
from qlib.config import C
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir

    # 1. download data and qlib initialization, if the data already exists, it will be skipped.
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    log_dir = Path("examples/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"workflow_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging_config = copy.deepcopy(C.logging_config)
    logging_config["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "level": "DEBUG",
        "formatter": "logger_format",
        "filename": str(log_path),
        "encoding": "utf-8",
        "filters": ["field_not_found"],
    }
    logging_config["loggers"]["qlib"]["handlers"] = ["console", "file"]

    qlib.init(provider_uri=provider_uri, region=REG_CN, logging_config=logging_config)
    print(f"Qlib log file: {log_path}")
    # qlib.init(provider_uri="/Users/joseph/qlib/examples/data/qlib_data", region=REG_CN)

    # 2. Instantiate model and dataset from config dicts using qlib's dependency injection pattern.
    # CSI300_GBDT_TASK["model"] contains {"class": "LGBModel", "module_path": ..., "kwargs": {...}}
    # CSI300_GBDT_TASK["dataset"] contains {"class": "DatasetH", ...} with Alpha158 handler + train/valid/test splits.
    # init_instance_by_config dynamically imports the class and calls it with kwargs — no hardcoded imports needed.
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    # 3. Define the backtest configuration (not executed yet, just a dict — used later by PortAnaRecord).
    #    Three sections:
    #    - executor: SimulatorExecutor simulates day-by-day order execution with portfolio metrics tracking.
    #    - strategy: TopkDropoutStrategy holds top-50 stocks by model score;
    #                n_drop=5 randomly drops 5 of the current holdings each rebalancing to reduce rank sensitivity.
    #    - backtest: simulation period (2017-2020), initial capital 1亿RMB, benchmark CSI300,
    #                exchange rules: A-share 涨跌停 9.5%, 以收盘价成交, 买0.05%/卖0.15% 佣金, 最低5元.
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
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
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

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    # 4. dataset.prepare("train") returns the training split as a DataFrame with MultiIndex (datetime, instrument).
    #    Columns are the 158 Alpha features after preprocessing (zscore normalization, fillna, cross-sectional rank).
    #    Useful for inspecting what the model actually sees, or for writing a custom training loop outside qlib.
    example_df = dataset.prepare("train")
    print(example_df.head())

    # start exp
    # 5. Open a new MLflow experiment run named "workflow". Everything inside this block (params, metrics,
    #    artifacts, model files) is persisted and queryable via MLflow UI (`mlflow ui` in terminal).
    #    R is the global QlibRecorder singleton — analogous to mlflow.start_run().
    with R.start(experiment_name="workflow"):
        # 5.1 Log all task hyperparameters (model params + dataset config) to MLflow as flat key-value pairs.
        #     flatten_dict converts nested dicts into dot-separated keys, e.g. {"model.kwargs.n_estimators": 200}.
        #     This ensures every run is reproducible and comparable in the MLflow experiment dashboard.
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        # 5.2 Train the LightGBM model. Internally calls dataset.prepare(["train", "valid"]) to get feature matrices,
        #     trains with early stopping on the validation set, and stores feature importances internally.
        #     The label (training target) is the next-day return defined in the dataset handler config.
        model.fit(dataset)
        # 5.3 Serialize and save the trained model as params.pkl to the MLflow artifact store (local filesystem by default).
        #     This allows reloading the model later with R.load_object("params.pkl") without retraining.
        R.save_objects(**{"params.pkl": model})

        # prediction
        # 5.4 Generate and persist model predictions on the test set.
        #     R.get_recorder() returns the current active MLflow Recorder (the handle to this run's storage).
        #     SignalRecord calls model.predict(dataset) and saves the result as pred.pkl in the recorder.
        #     pred.pkl is a Series with MultiIndex (datetime, instrument) — each value is the model's
        #     predicted score (proxy for expected future return) for that stock on that day.
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        # 5.5 Analyze the quality of the prediction signal, reading pred.pkl saved in 5.4.
        #     Computes and saves: IC (correlation between predicted score and actual next-day return),
        #     ICIR (IC / std(IC), measures stability), Rank IC, and IC decay curve over multiple horizons.
        #     Rule of thumb: IC > 0.03 is useful, ICIR > 0.5 is stable. Results saved under sig_analysis/.
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        # 5.6 Run the full portfolio backtest using the strategy config defined in step 3.
        #     PortAnaRecord drives TopkDropoutStrategy with the saved predictions, simulates daily rebalancing,
        #     and generates performance reports saved under port_analysis/:
        #       - portfolio_metrics_daily.pkl: daily returns, turnover, cost
        #       - analysis_df.pkl: Sharpe, annualized return, max drawdown vs CSI300 benchmark
        #     "day" is the frequency — matches time_per_step in the executor config.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
