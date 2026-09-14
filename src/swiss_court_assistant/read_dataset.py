import polars as pl

df = pl.read_parquet("hf://datasets/voilaj/swiss-caselaw/data/*.parquet")