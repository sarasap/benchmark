"""Extract content from a URL using Exa."""

import asyncio
import argparse
import httpx
from shared.searchers.exa import ExaSearcher


async def run(url: str):
    searcher = ExaSearcher(max_age_hours=0)
    try:
        results = await searcher.extract(url)
    except httpx.HTTPStatusError as e:
        print(f"Exa error {e.response.status_code}: {e.response.text}")
        return
    if not results:
        print("No results returned.")
        return
    print(results[0].text)


def main():
    parser = argparse.ArgumentParser(description="Extract content from a URL using Exa")
    parser.add_argument("url", help="URL to extract")
    args = parser.parse_args()
    asyncio.run(run(args.url))


if __name__ == "__main__":
    main()
