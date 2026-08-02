"""
Convert akshare-style parquet files to qlib binary format.

Usage:
    # Single file
    python scripts/akshareToBin.py \
        --src examples/data/301217_hfq_20220127_20260430.parquet

    # Directory of parquet files
    python scripts/akshareToBin.py \
        --src qlib/data/20260506

    # Custom output directory
    python scripts/akshareToBin.py \
        --src qlib/data/20260506 \
        --qlib_dir ~/.qlib/qlib_data/cn_data
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------- #
# Column mapping: akshare Chinese names → qlib standard English names
# --------------------------------------------------------------------------- #
AKSHARE_TO_QLIB = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "流通股本": "outstanding_share",
    "换手率": "turnover",
    "股票代码": "symbol",
}

# Exchange prefix rules for A-share codes
def _exchange_prefix(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code.startswith(("60", "68", "90")):
        return "sh" + code   # Shanghai: main board, STAR, B-share
    return "sz" + code       # Shenzhen: main board, ChiNext, SME


def preprocess(src_path: Path) -> pd.DataFrame:
    """Read one akshare parquet file and return a qlib-ready DataFrame."""
    df = pd.read_parquet(src_path)

    # Rename Chinese columns to qlib names (ignore columns not in mapping)
    df = df.rename(columns=AKSHARE_TO_QLIB)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add factor = 1.0  (data is already HFQ / backward-adjusted)
    df["factor"] = 1.0

    # Calculate daily change (next-day return proxy used by some Alpha factors)
    df["change"] = df["close"].pct_change().fillna(0.0)

    # Normalise stock code to qlib format: sz301217 / sh600000
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].apply(_exchange_prefix)
    else:
        # Fall back to inferring symbol from filename: "301217_hfq_..." → "301217"
        raw_code = src_path.stem.split("_")[0]
        df["symbol"] = _exchange_prefix(raw_code)

    return df


def collect_parquet_files(src_path: Path) -> list[Path]:
    if src_path.is_dir():
        return sorted(src_path.glob("*.parquet"))
    return [src_path] if src_path.suffix == ".parquet" else []


def default_qlib_dir(src_path: Path) -> Path:
    src_dir = src_path.parent if src_path.suffix else src_path
    return src_dir / "qlib_data"


def convert(src: str, qlib_dir: str | None = None, max_workers: int = 4):
    from dump_bin import DumpDataAll

    src_path = Path(src).expanduser()
    qlib_dir_path = Path(qlib_dir).expanduser() if qlib_dir else default_qlib_dir(src_path)

    # Collect source files
    parquet_files = collect_parquet_files(src_path)

    if not parquet_files:
        print(f"No .parquet files found in {src_path}")
        return

    # Write preprocessed parquet files to a temp staging directory
    staging_dir = qlib_dir_path / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    for f in parquet_files:
        print(f"  Preprocessing {f.name} ...")
        df = preprocess(f)
        symbol = df["symbol"].iloc[0]
        out_path = staging_dir / f"{symbol}.parquet"
        df.to_parquet(out_path, index=False)
        staged.append(out_path)
        print(f"    → {out_path.name}  ({len(df)} rows, columns: {df.columns.tolist()})")

    # Run qlib's DumpDataAll on the staged files
    print(f"\nDumping {len(staged)} file(s) to qlib binary format at {qlib_dir_path} ...")
    dumper = DumpDataAll(
        data_path=str(staging_dir),
        qlib_dir=str(qlib_dir_path),
        freq="day",
        max_workers=max_workers,
        date_field_name="date",
        file_suffix=".parquet",
        symbol_field_name="symbol",
        # Exclude non-feature columns that dump_bin would otherwise write as features
        exclude_fields="symbol,date",
    )
    dumper.dump()

    # Clean up staging directory
    for f in staged:
        f.unlink()
    staging_dir.rmdir()

    print(f"\nDone. qlib data written to: {qlib_dir_path.resolve()}")
    print("To use in qlib:")
    print(f'  qlib.init(provider_uri="{qlib_dir_path.resolve()}", region="cn")')


def resolve_data_date(target_date: str | None = None) -> str:
    if target_date:
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                parsed_date = datetime.strptime(target_date, fmt).date()
                return parsed_date.strftime("%Y%m%d")
            except ValueError:
                continue
        raise ValueError(f"Invalid date: {target_date}. Expected YYYYMMDD or YYYY-MM-DD")

    today = date.today()
    if today.weekday() >= 5:
        today -= timedelta(days=today.weekday() - 4)
    return today.strftime("%Y%m%d")


def main():
    parser = argparse.ArgumentParser(description="Convert akshare parquet to qlib binary format")
    parser.add_argument("--src", required=False, help="Source parquet file or directory of parquet files")
    parser.add_argument(
        "--qlib_dir",
        default=None,
        help="Target qlib data directory. Defaults to <source directory>/qlib_data.",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument(
        "--date",
        default=None,
        help="Target date to use (YYYYMMDD or YYYY-MM-DD). Defaults to today, or the most recent Friday on weekends.",
    )
    args = parser.parse_args()

    resolved_date = resolve_data_date(args.date)
    data_dir = Path(f"examples/data/{resolved_date}").expanduser()
    src = args.src if args.src else str(data_dir)

    convert(src, args.qlib_dir, args.max_workers)


if __name__ == "__main__":
    main()
