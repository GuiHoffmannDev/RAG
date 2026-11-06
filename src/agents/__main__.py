"""Run the corrective graph against the configured corpus and print the cited answer."""

import argparse
import logging

from src.agents.runtime import Runtime
from src.config import Settings
from src.models import Query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    request = Query(question=args.question)
    runtime = Runtime(Settings.load())
    try:
        print(runtime.ask(request)["result"].model_dump_json(indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
