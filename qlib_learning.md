# Qlib 库架构解析

Qlib 是微软开源的量化投资平台，整体架构分为 **基础设施层 → 数据层 → 模型层 → 工作流层 → 回测/策略层**，并有贡献库（contrib）作为具体实现的集合。

---

## 一、整体架构分层

```
┌──────────────────────────────────────────────────────────────┐
│                     用户入口 / 配置                           │
│  qlib/__init__.py (init/auto_init)  qlib/config.py (C)       │
│  qlib/constant.py  qlib/log.py  qlib/typehint.py             │
└───────────────────────────┬──────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐
│  数据层       │   │  模型层       │   │  工作流/实验管理层    │
│  qlib/data/  │   │  qlib/model/ │   │  qlib/workflow/      │
└──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘
       │                  │                       │
       └──────────────────┼───────────────────────┘
                          ▼
              ┌────────────────────────┐
              │  回测 & 策略执行层      │
              │  qlib/backtest/        │
              │  qlib/strategy/        │
              └────────────┬───────────┘
                           │
              ┌────────────▼───────────┐
              │  强化学习层            │
              │  qlib/rl/              │
              └────────────────────────┘
                           │
              ┌────────────▼───────────┐
              │  贡献库（具体实现）     │
              │  qlib/contrib/         │
              └────────────────────────┘
```

---

## 二、各模块详解

### 1. 基础设施层

| 文件/模块 | 职责 |
|-----------|------|
| `__init__.py` | 提供 `init()` / `auto_init()` 入口；负责挂载 NFS、注册 ops、初始化 recorder |
| `config.py` | 全局单例 `C (QlibConfig)`，管理 provider_uri、cache、redis、exp_manager 等所有配置；支持 client/server 两种模式 |
| `constant.py` | 常量定义：REG_CN/US/TW、EPS、INF、时间 Timedelta |
| `log.py` | 自定义 `QlibLogger` + `_QLibLoggerManager`，统一管理日志级别；`TimeInspector` 做性能计时 |
| `typehint.py` | 通用类型提示：`InstConf` / `InstDictConf`，统一描述如何从配置实例化对象 |
| `utils/` | 工具函数集：时间重采样、文件 IO、并行化、序列化、索引数据结构等 |

---

### 2. 数据层 (`qlib/data/`)

这是 Qlib 的核心，采用 **Provider 模式 + 表达式引擎 + 多级缓存** 架构：

```
用户 API (D.features / D.calendar)
        │
   LocalProvider / ClientProvider
        │
  ┌─────┴───────────────────────────┐
  │  CalendarProvider               │ ← 交易日历
  │  InstrumentProvider             │ ← 股票列表/过滤
  │  FeatureProvider                │ ← 原始特征 OHLCV
  │  ExpressionProvider             │ ← 表达式计算（Alpha因子）
  │  DatasetProvider                │ ← 多股票数据集组装
  └─────────────────────────────────┘
        │
  Cache 层 (DiskExpressionCache / DiskDatasetCache / MemCache)
        │
  Storage 层 (本地二进制文件 / 远程服务器)
```

#### 数据层关键文件说明

| 文件 | 职责 |
|------|------|
| `base.py` | `Expression` 抽象基类，支持算术/比较/逻辑运算符重载，是所有 Alpha 因子的基类 |
| `ops.py` | 内置运算符：元素级（Abs/Log/Sign）、双操作数（Add/Sub/Mul/Corr）、滚动窗口（Mean/Std/Rank/EMA/WMA）等 40+ 个算子 |
| `pit.py` | Point-In-Time 数据支持（财务数据防未来泄露） |
| `filter.py` | 股票过滤器 `NameDFilter` / `ExpressionDFilter` |
| `cache.py` | 多级缓存体系（内存/磁盘/Redis） |
| `data.py` | Provider 接口定义及 Local/Client 实现 |
| `client.py` | 远程 Client 连接实现 |

#### `dataset/` 子目录 — Dataset 三层管道

```
DatasetH (Dataset with Handler)
    │
    Handler (特征工程管道)
        │── Loader       ← 从 Provider 加载原始数据
        │── Processor[]  ← 数据预处理（归一化/fillna/CSRankNorm 等）
        └── 输出 DataFrame (instrument × feature)
```

---

### 3. 模型层 (`qlib/model/`)

定义模型的抽象接口，采用极简设计：

```
BaseModel (Serializable)
    └── Model
          ├── fit(dataset)         ← 训练
          ├── predict(dataset)     ← 预测
          └── ModelFT
                └── finetune(dataset)   ← 微调

Trainer                            ← 训练编排器
    ├── TrainerR                   ← 基于 Recorder 的训练
    ├── DelayTrainerR              ← 延迟训练
    ├── TrainerRM                  ← 带 Task Manager 的训练
    └── DelayTrainerRM             ← 延迟 + Task Manager
```

#### 模型层子目录

| 子目录 | 内容 |
|--------|------|
| `meta/` | 元学习（DDG-DA 等动态场景适应） |
| `ens/` | 模型集成（Ensemble） |
| `riskmodel/` | 风险模型（结构化协方差估计） |
| `interpret/` | 模型可解释性 |

---

### 4. 工作流/实验管理层 (`qlib/workflow/`)

类 MLflow 的实验追踪体系：

```
QlibRecorder (R)           ← 全局单例，用户接口
    │
ExpManager                 ← 实验管理器
    └── MLflowExpManager   ← MLflow 后端实现
          │
    Experiment             ← 单次实验
        └── MLflowExperiment
              │
        Recorder            ← 单次运行记录
            └── MLflowRecorder
                  ├── log_params / log_metrics
                  ├── save_objects / load_object
                  └── log_artifact
```

#### RecordTemp — 标准化记录模板

| 类 | 职责 |
|----|------|
| `SignalRecord` | 保存模型预测信号 |
| `SigAnaRecord` | 信号分析（IC / IR / 多空收益） |
| `PortAnaRecord` | 组合回测分析报告 |
| `HFSignalRecord` | 高频信号记录 |

#### `task/` 子目录 — 分布式任务管理

| 文件 | 职责 |
|------|------|
| `gen.py` | 任务生成（参数网格搜索/滚动窗口切分） |
| `manage.py` | MongoDB 任务队列管理（多机并行） |
| `collect.py` | 实验结果收集与聚合 |
| `utils.py` | 任务辅助工具 |

#### `online/` 子目录

支持在线服务部署与模型滚动更新（生产环境中的实时预测更新）。

---

### 5. 回测层 (`qlib/backtest/`)

分层事件驱动回测引擎，支持**嵌套多频率回测**（如日级策略 + 分钟级执行）：

```
backtest() / collect_data()
        │
  NestedExecutor (外层：日频)
        │── BaseStrategy.generate_trade_decision()
        │── TradeCalendarManager
        │
  SimulatorExecutor (内层：分钟频)
        │── Exchange.deal_order()
        │── Account.update_order()
        │       └── Position
        │
  Report: PortfolioMetrics + Indicator (成交质量分析)
```

#### 核心类说明

| 类/文件 | 职责 |
|---------|------|
| `Exchange` | 交易所模拟：涨跌停/停牌/交易成本/成交量约束 |
| `Position` / `InfPosition` | 持仓管理（有限资金 / 无限资金） |
| `Account` | 账户（持仓 + 现金 + 绩效指标统计） |
| `BaseTradeDecision` / `TradeDecisionWO` | 策略决策对象（携带 Order 列表） |
| `Order` | 订单对象（股票/方向/数量/时间） |
| `NestedExecutor` | 支持多层级嵌套执行的执行器 |
| `TradeCalendarManager` | 交易日历管理，追踪当前交易步骤 |
| `PortfolioMetrics` | 组合绩效指标（收益/成本/换手率） |
| `Indicator` | 成交质量分析（成交率/价格优势/正向率） |
| `profit_attribution.py` | Brinson 业绩归因分析 |

---

### 6. 策略层 (`qlib/strategy/`)

```
BaseStrategy
    ├── generate_trade_decision()       ← 核心方法，输出 TradeDecision
    ├── reset_level_infra()             ← 注入 TradeCalendar/Exchange/Position
    ├── update_trade_decision()         ← 更新决策（子级执行反馈）
    ├── alter_outer_trade_decision()    ← 修改上级决策
    ├── RLStrategy                      ← RL 策略基类
    └── RLIntStrategy                   ← 整数动作 RL 策略
```

`contrib/strategy/` 中的具体策略实现：

| 策略 | 说明 |
|------|------|
| `TopkDropoutStrategy` | 持仓 TopK 信号股票，随机 Dropout 防过拟合 |
| `EnhancedIndexingStrategy` | 增强指数策略（跟踪误差约束） |
| `CostControlStrategy` | 成本控制策略 |
| `rule_strategy.py` | 规则策略（TWAPStrategy/SBBStrategyBase 等） |
| `signal_strategy.py` | 基于信号的策略基类 |

---

### 7. 强化学习层 (`qlib/rl/`)

通用 RL 框架接口，与 backtest 层解耦，支持 Gym 兼容：

```
Simulator                    ← 环境模拟器 (step / get_state / done)
    │
StateInterpreter             ← 模拟器状态 → Gym 观测空间 (ObsType)
ActionInterpreter            ← 策略动作 → 实际执行动作 (ActType)
Reward                       ← 奖励函数
    └── RewardCombination    ← 多奖励加权组合
AuxiliaryInfoCollector       ← 辅助信息收集（用于日志/分析）
```

`order_execution/` — 订单执行 RL 场景的完整实现（包含 TWAP/VWAP 等基准策略对比）

---

### 8. 贡献库 (`qlib/contrib/`)

包含上述所有抽象层的具体实现，是社区贡献和算法研究的集合：

| 子目录 | 内容 |
|--------|------|
| `model/` | 30+ 个模型：LSTM / GRU / Transformer / GATs / TRA / HIST / ADARNN / LightGBM / XGBoost / CatBoost 等 |
| `strategy/` | TopkDropout / EnhancedIndexing / CostControl / TWAP 等策略 |
| `data/` | 高频数据处理、Alpha158/Alpha360 因子库 |
| `rolling/` | 滚动训练框架（时间序列交叉验证） |
| `meta/` | 元学习（DDG-DA 场景适应） |
| `online/` | 在线学习 / 实时模型更新 |
| `eva/` | 评估工具 |
| `report/` | 回测报告生成 |
| `tuner/` | 超参数调优 |
| `ops/` | 自定义数据算子 |

---

## 三、典型工作流

```python
# 1. 初始化
import qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 2. 构建数据集（data 层）
from qlib.contrib.data.handler import Alpha158
dataset = qlib.data.dataset.DatasetH(
    handler=Alpha158(fit_start_time="2008-01-01", fit_end_time="2014-12-31"),
    segments={"train": ("2008-01-01", "2014-12-31"), "test": ("2015-01-01", "2020-12-31")}
)

# 3. 训练模型（model 层）
from qlib.contrib.model.gbdt import LGBModel
model = LGBModel(...)
model.fit(dataset)

# 4. 记录实验（workflow 层）
from qlib.workflow import R
with R.start(experiment_name="my_exp"):
    model.fit(dataset)
    signal = model.predict(dataset)
    R.save_objects(signal=signal)
    R.log_metrics(ic=0.05)

# 5. 回测（backtest + strategy 层）
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest, create_account_instance
strategy = TopkDropoutStrategy(model=model, dataset=dataset, topk=50)
report = backtest(strategy=strategy, ...)
```

---

## 四、设计模式总结

| 设计模式 | 应用位置 | 说明 |
|----------|----------|------|
| **Provider 模式** | `qlib/data/` | 数据访问统一通过 Provider 接口，支持 Local/Client/Server 无缝切换 |
| **表达式引擎** | `qlib/data/base.py`, `ops.py` | `Expression` 基类 + 运算符重载，用字符串描述 Alpha 因子（如 `"Ref($close,1)/$close-1"`） |
| **全局配置单例** | `qlib/config.py` | `C (QlibConfig)` 统一管理所有组件配置，`qlib.init()` 时注入 |
| **嵌套执行器** | `qlib/backtest/executor.py` | `NestedExecutor` 实现多时间频率的层次化回测（日级策略 + 分钟级执行） |
| **InstConf 实例化约定** | 全局 | 所有组件通过 `{'class': ..., 'module_path': ..., 'kwargs': ...}` 字典配置并动态实例化，高度可扩展 |
| **抽象基类 + contrib 实现** | `model/` `strategy/` `rl/` | 核心包只定义接口，具体算法放在 `contrib/`，便于社区贡献和替换 |
| **Observer/事件驱动** | `qlib/backtest/` | 回测引擎以时间步为驱动，策略/执行器响应每个交易步骤 |
| **装饰器/缓存** | `qlib/data/cache.py` | ExpressionCache / DatasetCache 对 Provider 做透明缓存包装 |

---

## 五、学习路径（资深 Python + ML + 金融背景）

### 先 review：原计划哪里好，哪里需要补强

原计划的主线是正确的：**先打通数据和端到端 workflow，再进入因子、策略、滚动训练**。这很适合有 Python、ML、金融基础的人，因为你不需要先从语法或金融概念补课，而是要尽快建立 Qlib 的工程心智模型。

需要补强的地方主要有四点：

| 缺口 | 为什么重要 | 如何补 |
|------|------------|--------|
| 缺少可执行产物 | 只看懂架构不等于会用 Qlib 做研究 | 每阶段都产出 notebook / 脚本 / 指标表 |
| 缺少验收标准 | 容易陷入“跑过一次但不知道是否正确” | 每阶段定义最小通过条件 |
| 缺少金融 ML 风控视角 | Qlib 很容易跑出漂亮但泄露/过拟合的结果 | 单独加入泄露、滚动、交易成本、稳定性检查 |
| 缺少最终项目 | 学习路径需要收束成可复用研究模板 | 最后做一个可重复的 Alpha 研究小项目 |

下面是更落地的版本。执行时建议维护一个 `learning_log.md`，每天记录：今天跑了什么、关键配置、结果指标、踩坑、明天动作。

---

### 阶段一：数据环境和表达式引擎打通（1-2 天）

**目标**：确认本地数据、交易日历、股票池、表达式引擎都能正常工作。

**关键任务：**
1. 下载 A 股数据到本地（`qlib_data/cn_data`），用官方脚本 `scripts/get_data.py`。
2. 验证 `D.calendar()`、`D.instruments()`、`D.features()` 都能返回合理结果。
3. 理解表达式引擎的时间方向：`Ref($close, 1)` 是过去，`Ref($close, -1)` 是未来，标签里可以用未来，特征里不能用未来。

```python
import qlib
from qlib.data import D
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
df = D.features(["sh600000"], ["$close", "Ref($close,1)/$close-1"],
                start_time="2020-01-01", end_time="2023-12-31")
```

**产物**：
- 一个最小脚本或 notebook：初始化 Qlib，拉取 1 只股票和 1 个股票池的 OHLCV + 自定义表达式。
- 一张表：`$close`、`Ref($close, 1)`、简单收益表达式，手工抽 3 行核对计算方向。

**验收标准**：
- 能解释 Qlib 的 MultiIndex 数据结构：`instrument` × `datetime`。
- 能说清楚 `Ref($close, 1)` 和 `Ref($close, -1)` 的区别。
- 能用 `D.features()` 拿到带自定义因子的数据。

---

### 阶段二：跑通端到端实验并拆解配置（2-3 天）

**目标**：跑通 Qlib 的标准研究链路，并知道每个配置块负责什么。

先运行 `examples/workflow_by_code.py`，再对照 `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` 拆配置：

```
数据(Alpha158) → 模型(LGBModel) → 实验记录(R) → 回测(TopkDropoutStrategy) → 报告
```

Alpha158 中 158 个因子本质是动量/反转/波动率等金融信号的特征工程，有 ML 背景会很快理解每个特征含义。

**关键指标含义：**

| 指标 | 含义 | 参考值 |
|------|------|--------|
| IC | 预测值与未来收益的相关系数 | > 0.03 有效 |
| ICIR | IC 均值 / IC 标准差（稳定性） | > 0.5 较好 |
| Sharpe | 年化收益 / 波动率 | > 1.5 可用 |
| 最大回撤 | 峰值到谷值的最大跌幅 | < 30% |

**执行动作**：
1. 跑 `examples/workflow_by_code.py`，保存控制台输出和 artifacts 路径。
2. 跑一次 YAML 版本：`qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`。
3. 对照代码定位：`Alpha158`、`LGBModel`、`SignalRecord`、`SigAnaRecord`、`PortAnaRecord` 分别在哪里被实例化。

**产物**：
- 一页 workflow map：配置块 `task.model`、`task.dataset`、`record`、`port_analysis_config` 分别对应哪个 Python 类。
- 一张指标表：IC、ICIR、Rank IC、Rank ICIR、年化收益、Sharpe、最大回撤、换手率。

**验收标准**：
- 能看懂回测报告中的 IC/ICIR/年化收益/最大回撤。
- 能把一次实验从配置追到 `qlib/contrib/model/gbdt.py`、`qlib/contrib/data/handler.py`、`qlib/workflow/record_temp.py`。

---

### 阶段三：自定义 Alpha 因子和数据处理链（3-5 天）

在 `qlib/data/ops.py` 的 40+ 算子基础上，用字符串描述因子：

```python
# 20日动量反转因子
"(-1) * Corr(Rank(Ref($close, 1)), Rank($volume), 6)"

# 量价乖离（布林带位置）
"($close - Mean($close, 20)) / Std($close, 20)"

# 换手率异常
"$volume / Mean($volume, 20) - 1"
```

**学习重点：**
- `DatasetH` + 自定义 `DataHandlerLP` 的 `fields` 配置
- `Processor` 链：`RobustZScoreNorm` → `Fillna` → `CSRankNorm`（截面标准化）
- PIT 数据防未来泄露（`qlib/data/pit.py`）—— 财务数据必须用

**金融特殊性（区别于普通 ML）：**
- 金融数据截面非独立，不能用普通 K-Fold，必须用时间序列前向验证
- 关注 ICIR（IC 稳定性）而不只是 IC 均值
- IC 衰减是正常现象，关注衰减速度
- 标签定义要和交易假设一致。Qlib 的 Alpha158 常用 `Ref($close, -2)/Ref($close, -1) - 1`，原因是当天收盘后得到信号，下一交易日买入，再下一交易日卖出。

**执行动作**：
1. 阅读 `qlib/contrib/data/loader.py` 中 `Alpha158DL` 的字段生成逻辑。
2. 自己定义一个 10-20 个因子的轻量版 handler，先不要追求数量。
3. 对每个因子做单因子分析：覆盖率、缺失率、极值、IC 均值、ICIR、分年度 IC。
4. 把因子分成 4 类：动量/反转、量价、波动率、流动性。

**产物**：
- `my_alpha_fields.md`：记录每个因子的公式、金融含义、预期方向、潜在泄露点。
- 一个单因子评估表：每行一个因子，每列是 IC、ICIR、覆盖率、换手影响备注。

**验收标准**：
- 能自己写 3-5 个 Alpha 因子并完成 IC 验证。
- 不把 `Ref(..., -N)`、未来收益、未来财务数据放进特征。
- 至少能解释一个因子为什么有效、什么时候可能失效。

---

### 阶段四：自定义策略和交易假设（3-5 天）

**目标**：先学会改 `TopkDropoutStrategy` 的行为，再决定是否继承 `BaseStrategy` 从零写策略。对初学 Qlib 的资深开发者，建议顺序是：**读现成策略 → 调参数 → 小改策略 → 再写新策略**。

如果从零写，可以继承 `BaseStrategy`，实现 `generate_trade_decision()`：

```python
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import TradeDecisionWO, Order

class MyStrategy(BaseStrategy):
    def __init__(self, model, dataset, topk=20, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.dataset = dataset
        self.topk = topk

    def generate_trade_decision(self, execute_result=None):
        # 获取模型预测信号
        pred_scores = self.signal.get_signal()

        # 选股逻辑：取预测分最高的 topk 只
        buy_list = pred_scores.nlargest(self.topk).index.tolist()

        # 当前持仓
        current_pos = set(self.trade_position.get_stock_list())
        target_pos = set(buy_list)

        order_list = []
        # 买入新进股
        for stock in (target_pos - current_pos):
            order_list.append(Order(stock_id=stock, amount=10000,
                                    direction=Order.BUY))
        # 卖出调出股
        for stock in (current_pos - target_pos):
            order_list.append(Order(stock_id=stock, amount=10000,
                                    direction=Order.SELL))

        return TradeDecisionWO(order_list, self)
```

**参考实现**：`qlib/contrib/strategy/signal_strategy.py`（`TopkDropoutStrategy`）是最好的学习范本。

**关键配置项（`Exchange` 参数）：**
- `open_cost` / `close_cost`：买卖佣金（A股约 0.0003 / 0.0013）
- `min_cost`：最低成本（5元）
- `limit_threshold`：涨跌停限制（0.099）
- `deal_price`：以什么价格成交（`vwap` / `open` / `close`）

**执行动作**：
1. 先只调 `TopkDropoutStrategy`：`topk`、`n_drop`、调仓频率、交易成本。
2. 做参数敏感性分析：`topk` 不同取值下，收益、回撤、换手率如何变化。
3. 再加一个简单约束：单票权重上限、行业/指数成分过滤、换手率限制三选一。
4. 最后才实现自己的策略类。

**产物**：
- 一张策略参数对比表：`topk`、`n_drop`、成本、年化收益、Sharpe、最大回撤、换手率。
- 一个策略假设说明：成交价格、交易成本、涨跌停、停牌、调仓频率。

**验收标准**：
- 自定义策略或改造版策略能跑完回测。
- 结果不是只看 Sharpe，还能解释收益来自信号、风格暴露、交易频率还是成本假设。
- 若 Sharpe > 1.5，要进一步检查是否样本切分、交易价格或标签定义过于乐观。

---

### 阶段五：滚动训练、防过拟合和研究模板化（持续）

用滚动训练模拟真实生产环境中的模型迭代：

```python
# 典型的滚动窗口设置
# 训练窗口：过去 3 年  验证窗口：过去 6 个月  预测窗口：未来 1 个月
# 每隔 1 个月滚动一次
```

**使用工具**：
- `qlib/contrib/rolling/` — `RollingBenchmark` 滚动回测框架
- `qlib/workflow/task/gen.py` — 自动生成滚动时间切片的任务列表
- `qlib/workflow/task/manage.py` — MongoDB 任务队列（多机并行训练）

**防过拟合检查清单：**
- [ ] 样本外 IC 是否接近样本内 IC（差距 > 30% 说明过拟合）
- [ ] 策略收益是否主要来自少数几天（集中度过高）
- [ ] 换手率是否合理（过高意味着过拟合噪声）
- [ ] 在不同市场环境（牛市/熊市/震荡）下表现是否稳定
- [ ] 不同股票池（CSI300 / CSI500）下是否仍有方向一致的效果
- [ ] 加入更真实交易成本后，收益是否被完全吃掉
- [ ] 换一个模型（Linear / LightGBM / XGBoost）后，因子方向是否仍然稳定

**产物**：
- 一个可复用研究目录：`configs/`、`notebooks/`、`reports/`、`scripts/`。
- 一份滚动训练报告：每期训练区间、验证区间、测试区间、IC、组合收益、回撤。

**验收标准**：
- 能用滚动方式复现实验，而不是只做一次固定切分。
- 能判断一个结果是“值得继续研究”还是“样本内漂亮但不可交易”。

---

### 四周执行节奏建议

| 周期 | 主线 | 每周交付物 |
|------|------|------------|
| 第 1 周 | 环境、数据、端到端 workflow | 跑通 demo；完成 workflow map；记录第一版指标表 |
| 第 2 周 | Alpha158、表达式、Processor、单因子分析 | 3-5 个自定义因子；单因子评估表 |
| 第 3 周 | 策略、回测、交易成本、参数敏感性 | 策略参数对比表；交易假设说明 |
| 第 4 周 | 滚动训练、防过拟合、复盘 | 一个完整小项目报告；下一步研究 backlog |

---

### 最小毕业项目

做一个“可复现实验”即可，不追求一开始就赚钱：

1. 股票池：先用 CSI300，再扩展到 CSI500。
2. 特征：Alpha158 + 自定义 5 个因子。
3. 模型：Linear、LightGBM 两个 baseline。
4. 策略：TopK + dropout，至少比较 3 组 `topk/n_drop`。
5. 回测：固定切分 + 滚动切分都跑一次。
6. 报告：必须包含 IC/ICIR、Rank IC、年化收益、Sharpe、最大回撤、换手率、成本敏感性、分年度表现。

完成这个项目后，你对 Qlib 的掌握会从“能运行示例”进入“能组织一次量化研究”。

---

## 六、推荐阅读顺序（代码）

| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | `examples/workflow_by_code.py` | 全流程感知 |
| 2 | `qlib/contrib/data/handler.py` | 理解 Alpha158 如何构建 |
| 3 | `qlib/contrib/strategy/signal_strategy.py` | 策略实现参考 |
| 4 | `qlib/backtest/executor.py` | 回测引擎核心 |
| 5 | `qlib/contrib/model/gbdt.py` | 最简单的生产级模型 |
| 6 | `qlib/data/ops.py` | Alpha 因子算子全集 |

---

## qlib Overview of Codebases from Claude Code

### 基本规模

- **332 个 Python 文件**，约 **74,000 行代码**，43 个子包
- Python 3.8+，核心数据运算有 **Cython C++ 扩展**加速（`_libs/rolling`、`_libs/expanding`）

### 目录结构

```
qlib/
├── qlib/              # 核心库
│   ├── data/          # 数据层（Provider/表达式引擎/缓存）
│   ├── model/         # 模型抽象接口
│   ├── workflow/      # 实验管理（类 MLflow）
│   ├── backtest/      # 回测引擎
│   ├── strategy/      # 策略基类
│   ├── rl/            # 强化学习框架
│   ├── contrib/       # 社区贡献的具体实现（模型/策略/数据）
│   ├── cli/           # `qrun` 命令行入口
│   └── config.py      # 全局配置单例（18KB）
├── examples/          # 示例和基准测试（16个方向）
├── tests/             # 按模块组织的测试套件
├── scripts/           # 数据下载等工具脚本
└── docs/              # 文档
```

### contrib/ 主要内容

| 子目录 | 内容 |
|--------|------|
| `model/` | LightGBM/XGBoost/LSTM/Transformer/TRA 等 30+ 模型 |
| `strategy/` | TopkDropout/EnhancedIndexing/TWAP 等策略 |
| `data/` | Alpha158/Alpha360 因子库、高频数据工具 |
| `rolling/` | 滚动训练框架 |
| `online/` | 生产环境在线服务 |
| `meta/` | 元学习（动态场景适应） |

### 主要依赖

| 类型 | 包 |
|------|----|
| 核心 | `pandas` `numpy` `mlflow` `lightgbm` |
| RL | `torch` `tianshou` |
| 组合优化 | `cvxpy` |
| 分布式任务 | `pymongo` |
| 缓存 | `redis` |

### examples/ 覆盖场景

`workflow_by_code.py` 是最佳起点，此外还有：高频交易、RL 订单执行、在线服务、超参数调优、模型可解释性等 16 个方向的完整示例。
