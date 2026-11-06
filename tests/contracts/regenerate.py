"""Rewrite the committed /query response schema. Run after an intentional API change:
python tests/contracts/regenerate.py"""

import json
from pathlib import Path

from src.models import QueryResponse

SNAPSHOT = Path(__file__).with_name("query_response.schema.json")

if __name__ == "__main__":
    SNAPSHOT.write_text(
        json.dumps(QueryResponse.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )
