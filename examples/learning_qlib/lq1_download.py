import qlib
from qlib.data import D
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
df = D.features(["sh600000"], ["$close", "Ref($close,1)/$close-1"],
                start_time="2020-01-01", end_time="2023-12-31")