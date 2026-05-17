"""
Conference Demo — Multi-Model Local AI Agent with Small Language Models
=======================================================================
Showcases four specialized small language models collaborating to answer queries:

  gemma3          → intent classification & response synthesis
  Qwen3.5-4B FT   → tool selection & argument extraction (fine-tuned)
  embeddinggemma  → semantic document retrieval

Run:  python demo.py
      python demo.py --query "What is enterprise pricing?"
      python demo.py --interactive
"""

import argparse
import asyncio
import base64
import os
import sys
import time

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.inference.client import SmallLanguageModelClient, SmallLanguageModelRole
from src.engine.inference.config import SCENARIO_CONFIG, DEMO_IMAGES_DIR, DEMO_DOCUMENTS_DIR
from src.engine.knowledge.vector_store import VectorStore
from src.engine.tools import create_default_registry
from src.engine.agent import SmallLanguageModelAgentOrchestrator, Intent

import importlib
_loader = importlib.import_module(SCENARIO_CONFIG.data_loader_module)
seed_vector_store = _loader.seed_vector_store
seed_sql_database = _loader.seed_sql_database

console = Console()

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

MODEL_COLOURS = {
    "INFERENCE":  "cyan",
    "FUNCTION":   "magenta",
    "EMBEDDING":  "green",
    "VISION":     "blue",
    "execution":  "yellow",
}

INTENT_COLOURS = {
    Intent.RAG_QUERY:     "green",
    Intent.TOOL_USE:      "yellow",
    Intent.DIRECT_ANSWER: "white",
    Intent.IMAGE_QUERY:   "blue",
}

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEMO_IMAGES_DIR)


def _load_image_b64(filename: str) -> str:
    """Load an image file and return its base64-encoded content."""
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------------------------------------------------------------------------
# Showcase queries — designed to exercise every intent type
# ---------------------------------------------------------------------------

_SHOWCASE_QUERIES_NEXTERA: list[tuple[str, str, list[str] | None]] = [
    ("What are the features included in the Enterprise plan?",    "RAG → knowledge base search",  None),
    ("What were the total sales revenue figures for 2024?",       "TOOL → sql_query",             None),
    ("If I have 50 customers paying €999/month, what is my ARR?", "TOOL → calculator",            None),
    ("Which plan should a 15-person startup choose?",             "RAG → reasoning from docs",    None),
    ("What is 23% of 84900?",                                     "TOOL → calculator",            None),
    ("How many new customers joined in Q3 and Q4 of 2024?",       "TOOL → sql_query",             None),
    ("Which product tier generates the most revenue?",              "TOOL → Qwen FT → sql",   None),
    ("How does our MRR break down by industry?",                   "TOOL → Qwen FT → sql",   None),
    ("What was our best-performing product last quarter, and what would revenue look like if we grew it by 15%?",
                                                                     "MULTI-STEP → sql_query → calculator", None),
    ("Hello! What can you help me with?",                         "DIRECT → conversational",      None),
    ("What trends do you see in this revenue chart?",             "IMAGE → revenue analysis",     ["revenue_chart.png"]),
    ("Summarize the pricing tiers shown in this table",           "IMAGE → pricing comparison",   ["pricing_table.png"]),
    ("Explain what this system diagram shows",                    "IMAGE → architecture review",  ["architecture_diagram.png"]),
]

SHOWCASE_QUERIES = _SHOWCASE_QUERIES_NEXTERA


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def print_header() -> None:
    console.print()
    console.print(Rule("[bold white]Multi-Model Local AI Agent with Small Language Models[/bold white]", style="dim"))
    console.print()

    model_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    model_table.add_column("Role",    style="bold", min_width=14)
    model_table.add_column("Model",   style="cyan")
    model_table.add_column("Purpose", style="dim")

    model_table.add_row("Thinker",   "gemma3-ft",       "Intent classification · response synthesis")
    model_table.add_row("Doer",      "Qwen3.5-4B FT",  "Tool selection · argument extraction")
    model_table.add_row("Librarian", "embeddinggemma", "Semantic search · document retrieval")
    model_table.add_row("Vision",    "gemma3-4B",     "Multimodal image understanding")

    console.print(Padding(model_table, (0, 2)))
    console.print()


def print_health(health: dict[str, bool]) -> None:
    status_parts = []
    all_ok = True
    for model_role, available in health.items():
        color = "green" if available else "red"
        icon  = "●" if available else "○"
        status_parts.append(f"[{color}]{icon} {model_role}[/{color}]")
        if not available:
            all_ok = False

    status_line = "   ".join(status_parts)
    console.print(f"  Models: {status_line}")

    if not all_ok:
        console.print()
        console.print(
            "  [yellow]Some model servers are not running. Start them with:[/yellow]\n"
            "    [dim]bash scripts/start_servers.sh --bg     # base model[/dim]\n"
            "    [dim]bash scripts/start_servers.sh --bg --ft  # fine-tuned model[/dim]"
        )
        console.print()
    else:
        console.print()


def print_step(step, index: int) -> None:
    """Render a single execution step with model badge."""
    model_name = step.model
    colour = "cyan"
    for role_name, colour_val in MODEL_COLOURS.items():
        if role_name.lower() in model_name.lower():
            colour = colour_val
            break

    action_label = step.action.replace("_", " ").title()
    model_label  = f"[{colour}]{escape(model_name)}[/{colour}]"
    timing = f"  [dim]({step.duration_ms:.0f} ms)[/dim]" if step.duration_ms > 0 else ""
    console.print(f"  [dim]{index}.[/dim] [bold]{action_label}[/bold]  via  {model_label}{timing}")

    if step.details:
        for key, value in step.details.items():
            if key == "error" and not value:
                continue
            val_str = str(value)
            if len(val_str) > 80:
                val_str = val_str[:77] + "…"
            console.print(f"      [dim]{key}:[/dim] {escape(val_str)}")


async def run_query(
    agent: SmallLanguageModelAgentOrchestrator,
    query: str,
    hint: str = "",
    images: list[str] | None = None,
) -> None:
    """Execute a query and render the full trace with Rich formatting."""
    console.print()
    console.print(Rule(style="dim"))
    console.print()

    # Query panel
    image_note = f"  [blue][{len(images)} image(s)][/blue]" if images else ""
    q_panel = Panel(
        f"[bold white]{escape(query)}[/bold white]{image_note}",
        title=f"[dim]Query[/dim]{('  · [dim italic]' + hint + '[/dim italic]') if hint else ''}",
        border_style="dim",
        expand=False,
    )
    console.print(Padding(q_panel, (0, 2)))
    console.print()

    start = time.perf_counter()

    # Show a spinner while the agent is thinking
    with console.status("[dim]Agent thinking…[/dim]", spinner="dots"):
        result = await agent.process(query, images=images)

    elapsed = (time.perf_counter() - start) * 1000

    # Intent badge
    intent_colour = INTENT_COLOURS.get(result.intent, "white")
    intent_label  = result.intent.value.replace("_", " ").upper()
    console.print(
        f"  Intent [bold {intent_colour}]{intent_label}[/bold {intent_colour}]  "
        f"[dim]({elapsed:.0f} ms)[/dim]"
    )
    console.print()

    # Execution trace
    if result.steps:
        console.print("  [bold dim]Execution trace[/bold dim]")
        for i, step in enumerate(result.steps, 1):
            print_step(step, i)
        console.print()

    # Response
    response_panel = Panel(
        escape(result.response),
        title="[bold]Response[/bold]",
        border_style=intent_colour,
        padding=(1, 2),
    )
    console.print(Padding(response_panel, (0, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Startup: check health and load data
# ---------------------------------------------------------------------------

async def startup(client: SmallLanguageModelClient, vector_store: VectorStore) -> bool:
    """Check llama-server health and load demo data. Returns True if ready."""
    console.print("  [dim]Checking model servers…[/dim]", end="")
    health = await client.check_health()
    console.print("\r", end="")
    print_health(health)

    # Seed data (no-op if already done)
    console.print("  [dim]Loading demo knowledge base…[/dim]")
    vector_store.set_client(client)
    n_new = await seed_vector_store(client, vector_store)
    n_total = await vector_store.count()
    if n_new > 0:
        console.print(f"  [green]✓[/green] Indexed {n_new} new documents ({n_total} total)")
    else:
        console.print(f"  [dim]✓ Knowledge base ready ({n_total} documents)[/dim]")

    console.print("  [dim]Loading SQLite database…[/dim]")
    await seed_sql_database()
    console.print("  [dim]✓ Business database ready[/dim]")
    console.print()

    return True


# ---------------------------------------------------------------------------
# Showcase mode
# ---------------------------------------------------------------------------

async def run_showcase(agent: SmallLanguageModelAgentOrchestrator) -> None:
    """Run all preset showcase queries in sequence."""
    console.print(
        Padding(
            "[bold]Running showcase queries…[/bold]  "
            "[dim](Press Ctrl+C to stop)[/dim]",
            (0, 2),
        )
    )

    for query, hint, image_files in SHOWCASE_QUERIES:
        images = None
        if image_files:
            try:
                images = [_load_image_b64(f) for f in image_files]
            except FileNotFoundError:
                console.print(f"  [yellow]Skipping image query (images not found). Run: python scripts/generate_sample_images.py[/yellow]")
                continue
        await run_query(agent, query, hint, images=images)
        await asyncio.sleep(0.5)   # brief pause between queries

    console.print()
    console.print(Rule("[dim]Showcase complete[/dim]", style="dim"))
    console.print()
    console.print(
        "  [dim]Interaction logs saved. Export for fine-tuning:[/dim]\n"
        "  [dim]  agent.export_training_data('./data/interactions.json')[/dim]"
    )


# ---------------------------------------------------------------------------
# Interactive REPL mode
# ---------------------------------------------------------------------------

async def run_interactive(agent: SmallLanguageModelAgentOrchestrator) -> None:
    """Interactive query REPL for live demos."""
    console.print(
        Padding(
            "[bold]Interactive mode[/bold]  [dim](type 'exit' to quit, 'showcase' for preset queries)[/dim]",
            (0, 2),
        )
    )

    while True:
        try:
            console.print()
            query = console.input("  [bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query.lower() == "showcase":
            await run_showcase(agent)
            continue

        await run_query(agent, query)

    console.print()
    console.print(Rule("[dim]Session ended[/dim]", style="dim"))


# ---------------------------------------------------------------------------
# OCR showcase
# ---------------------------------------------------------------------------

OCR_SHOWCASE_QUERIES: list[tuple[str, str]] = [
    ("What was total revenue in Q4 2024?",       "table extraction"),
    ("Which customer has the highest MRR?",       "customer table"),
    ("What is the Enterprise plan monthly price?", "pricing table"),
]
_OCR_PRIMARY_PDF = "nextera_quarterly_report.pdf"
_OCR_PRIMARY_DOC_ID = "nextera-quarterly-report"
_OCR_SECONDARY_PDF = "snowflake-fy2025-first50.pdf"
_OCR_SECONDARY_DOC_ID = "snowflake-fy2025-first50"
SECONDARY_QUERIES: list[tuple[str, str]] = [
    ("What was Snowflake's total revenue in fiscal year 2025?",   "revenue extraction"),
    ("How many customers does Snowflake serve as of FY2025?",     "customer count"),
]


async def run_ocr_showcase(client, vector_store, agent) -> None:
    """Upload demo PDFs and query their content via document chat."""
    from src.engine.knowledge.document_processor import DocumentProcessor
    from src.engine.knowledge.ocr_client import OCRClient
    from src.engine.knowledge.vector_store import VectorStore as VS

    console.print(Rule("[bold cyan]OCR Document Chat Showcase[/bold cyan]", style="cyan"))
    console.print()

    # Set up upload store
    upload_store = VS(collection_name="uploads", persist_dir=SCENARIO_CONFIG.chroma_dir)
    upload_store.set_client(client)

    # Check OCR availability
    ocr_client = None
    try:
        ocr = OCRClient()
        if await ocr.check_health():
            ocr_client = ocr
            console.print("  [green]\u2713[/green] GLM-OCR available (port 9098)")
        else:
            console.print("  [yellow]\u26A0[/yellow]  GLM-OCR not running \u2014 using pypdf fallback")
    except Exception:
        console.print("  [yellow]\u26A0[/yellow]  GLM-OCR not available")
    console.print()

    # Upload primary demo document
    primary_pdf = os.path.join(DEMO_DOCUMENTS_DIR, _OCR_PRIMARY_PDF)
    if os.path.isfile(primary_pdf):
        console.print(f"[bold]Uploading {_OCR_PRIMARY_PDF}...[/bold]")
        processor = DocumentProcessor(upload_store, ocr_client=ocr_client)
        t0 = time.perf_counter()
        async for event in processor.process_file(_OCR_PRIMARY_PDF,
                                                   open(primary_pdf, "rb").read()):
            console.print(f"  [{event.stage}] {event.message}")
        elapsed = (time.perf_counter() - t0) * 1000
        console.print(f"  [green]\u2713[/green] Done in {elapsed:.0f}ms")
        console.print()

        # Query via document chat
        doc_id = _OCR_PRIMARY_DOC_ID
        for query, hint in OCR_SHOWCASE_QUERIES:
            console.print(f"[bold cyan]Q:[/bold cyan] {query}  [dim]({hint})[/dim]")
            results = await upload_store.search(query, top_k=5,
                                                 where={"document_id": doc_id})
            if results:
                context = "\n\n".join(
                    f"[Source: {d.metadata.get('title', d.id)}]\n{d.content}"
                    for d in results[:5]
                )
                from src.engine.inference.prompts import RAG_SYNTHESIS_SYSTEM_PROMPT, RAG_SYNTHESIS_USER_TEMPLATE
                resp = await client.generate_synthesis(
                    messages=[
                        {"role": "system", "content": RAG_SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": RAG_SYNTHESIS_USER_TEMPLATE.format(
                            context=context, query=query)},
                    ],
                )
                console.print(Panel(resp.content.strip(), title="[bold]Response[/bold]",
                                     border_style="green", padding=(0, 1)))
            else:
                console.print("  [red]No results found[/red]")
            console.print()

    # Upload secondary demo document (if available)
    secondary_pdf = os.path.join(DEMO_DOCUMENTS_DIR, _OCR_SECONDARY_PDF)
    if os.path.isfile(secondary_pdf):
        console.print(f"[bold]Uploading {_OCR_SECONDARY_PDF}...[/bold]")
        processor = DocumentProcessor(upload_store, ocr_client=ocr_client)
        t0 = time.perf_counter()
        last_stage = ""
        async for event in processor.process_file(
            _OCR_SECONDARY_PDF,
            open(secondary_pdf, "rb").read(),
        ):
            if event.stage != last_stage:
                console.print(f"  [{event.stage}] {event.message}")
                last_stage = event.stage
        elapsed = (time.perf_counter() - t0) * 1000
        console.print(f"  [green]\u2713[/green] Done in {elapsed:.0f}ms")
        console.print()

        doc_id = _OCR_SECONDARY_DOC_ID
        for query, hint in SECONDARY_QUERIES:
            console.print(f"[bold cyan]Q:[/bold cyan] {query}  [dim]({hint})[/dim]")
            results = await upload_store.search(query, top_k=5,
                                                 where={"document_id": doc_id})
            if results:
                context = "\n\n".join(
                    f"[Source: {d.metadata.get('title', d.id)}]\n{d.content}"
                    for d in results[:5]
                )
                from src.engine.inference.prompts import RAG_SYNTHESIS_SYSTEM_PROMPT, RAG_SYNTHESIS_USER_TEMPLATE
                resp = await client.generate_synthesis(
                    messages=[
                        {"role": "system", "content": RAG_SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": RAG_SYNTHESIS_USER_TEMPLATE.format(
                            context=context, query=query)},
                    ],
                )
                console.print(Panel(resp.content.strip(), title="[bold]Response[/bold]",
                                     border_style="green", padding=(0, 1)))
            else:
                console.print("  [red]No results found[/red]")
            console.print()

    console.print(Rule("[dim]OCR showcase complete[/dim]", style="dim"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    print_header()

    # Initialise all components
    client       = SmallLanguageModelClient.create_with_auto_detection()
    vector_store = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)
    tools        = create_default_registry(vector_store=vector_store)
    agent        = SmallLanguageModelAgentOrchestrator(client, tools)

    # Startup checks
    await startup(client, vector_store)

    # Run mode
    if args.ocr:
        await run_ocr_showcase(client, vector_store, agent)
    elif args.query:
        images = None
        if args.image:
            with open(args.image, "rb") as f:
                images = [base64.b64encode(f.read()).decode("utf-8")]
        await run_query(agent, args.query, images=images)
    elif args.interactive:
        await run_interactive(agent)
    else:
        await run_showcase(agent)

    # Export training data if any interactions happened
    if agent.interaction_count > 0:
        n = agent.export_training_data("./data/interactions.json")
        console.print(
            f"\n  [dim]Exported {n} interactions → ./data/interactions.json[/dim]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Model Local AI Agent with Small Language Models"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="",
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to an image file (used with --query for vision queries)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL session",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run OCR showcase: upload a PDF, then query its content",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        console.print("\n  [dim]Interrupted.[/dim]")
