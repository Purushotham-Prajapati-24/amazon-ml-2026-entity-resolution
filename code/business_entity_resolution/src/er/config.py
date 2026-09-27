"""Paths shared by every pipeline stage.

Override with environment variables when the data does not live in the repo layout:
    ER_DATA_DIR   folder containing train/ and test/ with the raw TSVs
    ER_CACHE_DIR  folder for parquet caches and intermediate artefacts
    ER_OUTPUT_DIR folder for matching_results.tsv / candidate_pairs.tsv
"""
import os
from pathlib import Path

# src/er/config.py -> src -> business_entity_resolution -> code -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", REPO_ROOT / "student_resource" / "dataset"))
CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", REPO_ROOT / "data_cache"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", REPO_ROOT / "output"))

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
