import asyncio
import base64
import json
import os
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from google import genai
from google.genai import types

START_FROM = "contents_127"  # Set to None to process all

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "golden_output"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
DOM_DIR = OUTPUT_DIR / "dom_output"
MARKDOWN_DIR = OUTPUT_DIR / "golden_reference"


TAGS_TO_STRIP = ["script", "style", "svg", "noscript", "iframe"]


def clean_dom(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(TAGS_TO_STRIP):
        tag.decompose()
    return str(soup)


def load_queries() -> list[dict]:
    filepath = DATA_DIR / "contents" / "code_contents.jsonl"
    queries = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


async def render_page(url: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        page = await context.new_page()

        print(f"Loading {url}...")
        await page.goto(url, wait_until="load", timeout=60000)

        # Extra wait to make sure everything has settled
        await page.wait_for_timeout(8000)

        # Scroll through entire page to trigger lazy loading
        print("Triggering lazy loading...")
        await page.evaluate("""
            async () => {
                await new Promise((resolve) => {
                    const distance = 100;
                    const timer = setInterval(() => {
                        window.scrollBy(0, distance);
                        if (window.scrollY + window.innerHeight >= document.body.scrollHeight) {
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """)

        # Wait again after scrolling for any newly loaded content
        await page.wait_for_timeout(5000)

        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(2000)

        # Take full page screenshot
        print("Taking screenshot...")
        screenshot_bytes = await page.screenshot(
            full_page=True,
            type="png"
        )

        # Get the fully rendered DOM
        print("Extracting DOM...")
        dom_content = await page.content()

        # Get page title for reference
        title = await page.title()

        await browser.close()

        return {
            "url": url,
            "title": title,
            "screenshot_bytes": screenshot_bytes,
            "dom": dom_content
        }


def convert_to_markdown(screenshot_bytes: bytes, dom: str) -> str:
    """
    Feed screenshot + DOM to a multimodal model to get clean markdown.
    Using Gemini through vertexAI.
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ["VERTEX_AI_PROJECT"],
        location="us-central1"
    )

    prompt = """Convert this webpage to clean markdown.

            KEEP:
            - All prose and explanatory text
            - Code blocks (preserve exact formatting and syntax)
            - API signatures and function definitions
            - Tables (convert to markdown table format)
            - Headings and subheadings (preserve hierarchy)
            - Parameter descriptions and type information
            - Examples and usage notes

            STRIP:
            - Navigation menus and sidebars
            - Header and footer elements
            - Cookie banners and popups
            - Advertisement sections
            - Social media links
            - Breadcrumb navigation
            - Search bars
            - "Related articles" sections
            - Author attributions, contributor program notices, and donation callouts


            GOAL: What would be maximally useful for an LLM answering a coding question from this page?

            Use the screenshot to understand the visual layout and the DOM for precise text content.
            Produce markdown that faithfully represents the substantive content a developer would need."""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(
                data=screenshot_bytes,
                mime_type="image/png"
            ),
            types.Part.from_text(text=prompt),
            types.Part.from_text(text=f"DOM content for precise text extraction:\n\n{dom}")
        ]
    )

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    print(f"Tokens — input: {input_tokens}, output: {output_tokens}, total: {usage.total_token_count}")

    return response.text, input_tokens, output_tokens


def save_results(entry_id: str, result: dict, markdown: str):
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    DOM_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    screenshot_path = SCREENSHOTS_DIR / f"{entry_id}.png"
    with open(screenshot_path, "wb") as f:
        f.write(result["screenshot_bytes"])
    print(f"Screenshot saved to {screenshot_path}")

    dom_path = DOM_DIR / f"{entry_id}.html"
    with open(dom_path, "w", encoding="utf-8") as f:
        f.write(result["dom"])
    print(f"DOM saved to {dom_path}")

    markdown_path = MARKDOWN_DIR / f"{entry_id}.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Golden reference saved to {markdown_path}")


def append_to_golden_jsonl(entry_id: str, markdown: str):
    jsonl_path = DATA_DIR / "contents" / "golden_markdown.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": entry_id, "expected_markdown": markdown}) + "\n")


def already_processed(entry_id: str) -> bool:
    return (MARKDOWN_DIR / f"{entry_id}.md").exists()


async def process_entry(entry: dict):
    entry_id = entry["id"]
    url = entry["url"]

    if already_processed(entry_id):
        print(f"Skipping {entry_id} (already processed)")
        return "skipped", None, None

    print(f"\n{'='*60}")
    print(f"Processing {entry_id}: {url}")
    print(f"{'='*60}")

    try:
        result = await render_page(url)
        print(f"Page title: {result['title']}")
        print(f"Screenshot size: {len(result['screenshot_bytes'])} bytes")
        print(f"DOM size: {len(result['dom'])} characters")

        # Save screenshot and DOM before conversion
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        DOM_DIR.mkdir(parents=True, exist_ok=True)
        screenshot_path = SCREENSHOTS_DIR / f"{entry_id}.png"
        with open(screenshot_path, "wb") as f:
            f.write(result["screenshot_bytes"])
        print(f"Screenshot saved to {screenshot_path}")
        dom_path = DOM_DIR / f"{entry_id}.html"
        with open(dom_path, "w", encoding="utf-8") as f:
            f.write(result["dom"])
        print(f"DOM saved to {dom_path}")

        if len(result["dom"]) < 1000:
            print(f"FAILED {entry_id}: DOM too short ({len(result['dom'])} characters)")
            return "failed", None, None

        cleaned_dom = clean_dom(result["dom"])
        print(f"Cleaned DOM size: {len(cleaned_dom)} characters (was {len(result['dom'])})")

        cleaned_dom_path = DOM_DIR / f"{entry_id}_cleaned.html"
        with open(cleaned_dom_path, "w", encoding="utf-8") as f:
            f.write(cleaned_dom)
        print(f"Cleaned DOM saved to {cleaned_dom_path}")

        # Convert to markdown
        markdown, input_tokens, output_tokens = convert_to_markdown(result["screenshot_bytes"], cleaned_dom)
        print(f"Generated markdown ({len(markdown)} characters)")

        if not markdown.strip():
            print(f"FAILED {entry_id}: markdown is empty")
            return "failed", None, None

        if input_tokens > 1_000_000:
            print(f"WARNING {entry_id}: input tokens ({input_tokens}) exceeded 1M context window")
        if output_tokens >= 65536:
            print(f"WARNING {entry_id}: output tokens ({output_tokens}) hit the 65,536 output limit — markdown may be truncated")

        # Save markdown file and append to JSONL
        save_results(entry_id, result, markdown)
        append_to_golden_jsonl(entry_id, markdown)

        return "ok", input_tokens, output_tokens

    except Exception as e:
        print(f"Failed to process {entry_id} ({url}): {e}")
        return "failed", None, None


async def main():
    queries = load_queries()
    print(f"Found {len(queries)} URLs to process")

    failed_ids = []
    context_exceeded_ids = []
    output_truncated_ids = []

    start_reached = START_FROM is None
    for entry in queries:
        if not start_reached:
            if entry["id"] == START_FROM:
                start_reached = True
            else:
                print(f"Skipping {entry['id']} (before START_FROM)")
                continue
        status, input_tokens, output_tokens = await process_entry(entry)
        if status == "failed":
            failed_ids.append(entry["id"])
        else:
            if input_tokens and input_tokens > 1_000_000:
                context_exceeded_ids.append((entry["id"], input_tokens))
            if output_tokens and output_tokens >= 65536:
                output_truncated_ids.append((entry["id"], output_tokens))

    if failed_ids:
        print(f"\nFailed entries ({len(failed_ids)}):")
        for fid in failed_ids:
            print(f"  - {fid}")
    else:
        print("\nAll entries processed successfully.")

    if context_exceeded_ids:
        print(f"\nEntries that exceeded 1M input context ({len(context_exceeded_ids)}):")
        for fid, tokens in context_exceeded_ids:
            print(f"  - {fid}: {tokens:,} tokens")

    if output_truncated_ids:
        print(f"\nEntries that hit the 65,536 output token limit — markdown may be truncated ({len(output_truncated_ids)}):")
        for fid, tokens in output_truncated_ids:
            print(f"  - {fid}: {tokens:,} tokens")

    print(f"\nDone. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
