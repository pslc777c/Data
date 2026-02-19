import duckdb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DUCKDB_PATH = REPO_ROOT / "data" / "duckdb" / "lakehouse.duckdb"

def get_duckdb_connection(read_only=False):
    return duckdb.connect(
        database=str(DUCKDB_PATH),
        read_only=read_only
    )
