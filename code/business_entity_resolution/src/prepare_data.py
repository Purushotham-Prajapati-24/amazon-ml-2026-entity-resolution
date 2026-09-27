"""Stage 0: convert the raw TSVs into parquet once (pandas TSV load is ~150s; parquet is seconds).

Every column is read as a string with no NA coercion, so empty addresses and literal
"NA" names stay as-is and IDs keep their exact spelling.

    python src/prepare_data.py
"""
import time

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from er.config import CACHE_DIR, DATA_DIR, SOURCES, SPLITS


def convert(src, dst):
    t0 = time.time()
    header = src.open(encoding="utf-8").readline().rstrip("\n").split("\t")
    table = pcsv.read_csv(
        src,
        read_options=pcsv.ReadOptions(block_size=64 << 20),
        # RFC-4180 quoting: ~800 test names (mostly French) are wrapped in quotes with "" escapes.
        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char='"', double_quote=True),
        convert_options=pcsv.ConvertOptions(
            column_types={c: pa.string() for c in header},
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
        ),
    )
    pq.write_table(table, dst, compression="zstd")
    print(f"{src.name:28s} -> {dst.name:34s} {table.num_rows:>10,} rows  {time.time() - t0:5.1f}s")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for k in SOURCES:
            convert(DATA_DIR / split / f"{split}_source{k}.tsv", CACHE_DIR / f"{split}_source{k}.parquet")
    convert(DATA_DIR / "train" / "train_ground_truth.tsv", CACHE_DIR / "train_ground_truth.parquet")


if __name__ == "__main__":
    main()
