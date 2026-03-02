import argparse

from config import load_environment_variables
from elastic.elasticsearch_utils import (
    build_elasticsearch_client,
    search_elasticsearch_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query stored documents directly from Elasticsearch."
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="Text query to search for within the indexed documents.",
    )
    parser.add_argument(
        "--index",
        "-i",
        help="Optional Elasticsearch index name (defaults to ELASTICSEARCH_INDEX env var).",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=5,
        help="Number of results to fetch (default: 5).",
    )
    return parser.parse_args()


def print_results(results):
    if not results:
        print("No results found.")
        return

    for idx, result in enumerate(results, start=1):
        print(f"\nResult {idx}")
        print("-" * 40)
        print(f"Score : {result.get('score')}")
        print(f"Doc ID: {result.get('doc_id')}")
        print(f"Source: {result.get('source')} | Page: {result.get('page')}")
        snippet = result.get("content", "")
        if snippet:
            print(f"Content: {snippet[:500]}{'...' if len(snippet) > 500 else ''}")


def main():
    load_environment_variables()
    args = parse_args()

    client = None
    try:
        client = build_elasticsearch_client()
        results = search_elasticsearch_documents(
            client,
            args.query,
            index_name=args.index,
            size=args.size,
        )
        print_results(results)
    except Exception as error:
        print(f"Failed to fetch Elasticsearch results: {error}")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()

