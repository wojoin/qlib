#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import fire
from qlib.tests.data import GetData

if __name__ == "__main__":
    # fire.Fire(GetData)
    GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", region="cn", version="v3")
