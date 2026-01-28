"""
Train a byte-level BPE tokenizer on the TinyStories dataset.

This script trains a BPE tokenizer with vocab_size=10,000 including the <|endoftext|> special token,
serializes the vocabulary and merges to disk, and profiles the training process.
"""

import time
import tracemalloc
import cProfile
import pstats
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cs336_basics.tokenizer.bpe import train_bpe
from cs336_basics.tokenizer.utils import serialize_vocab_and_merges, find_longest_token
from cs336_basics.utils import get_logger

console = Console()
logger = get_logger(__name__)


def main():
    # Configuration
    input_path = "/root/assignment1-data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    enable_tracemalloc = True
    enable_profiling = True
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.json"

    # Print configuration
    config_table = Table(title="Configuration", show_header=False, border_style="blue")
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", str(input_path))
    config_table.add_row("Vocabulary size", str(vocab_size))
    config_table.add_row("Special tokens", str(special_tokens))
    config_table.add_row("Output directory", str(output_dir))
    config_table.add_row("Memory tracking", "Enabled" if enable_tracemalloc else "Disabled")
    config_table.add_row("Profiling", "Enabled" if enable_profiling else "Disabled")

    console.print(Panel("[bold blue]Training BPE Tokenizer on TinyStories Dataset[/]", expand=False))
    console.print(config_table)
    console.print()

    # Start memory tracking
    if enable_tracemalloc:
        tracemalloc.start()

    # Profile the training process
    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    # Time the training
    start_time = time.time()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    end_time = time.time()
    if profiler is not None:
        profiler.disable()

    # Get memory usage
    if enable_tracemalloc:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        current = 0
        peak = 0

    training_time = end_time - start_time

    # Serialize to disk
    logger.info("Serializing vocabulary and merges to disk...")
    serialize_vocab_and_merges(vocab, merges, vocab_path, merges_path)
    logger.info(f"Vocabulary saved to: {vocab_path}")
    logger.info(f"Merges saved to: {merges_path}")

    # Find longest token
    longest_id, longest_token, longest_len = find_longest_token(vocab)

    # Print statistics
    stats_table = Table(title="Training Statistics", show_header=False, border_style="green")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    stats_table.add_row("Training time", f"{training_time:.2f}s ({training_time/60:.2f}min)")
    if enable_tracemalloc:
        stats_table.add_row("Peak memory", f"{peak / 1024**3:.2f} GB")
    stats_table.add_row("Vocabulary size", str(len(vocab)))
    stats_table.add_row("Number of merges", str(len(merges)))
    stats_table.add_row("Longest token ID", str(longest_id))
    stats_table.add_row("Longest token length", f"{longest_len} bytes")
    try:
        decoded = longest_token.decode("utf-8")
        stats_table.add_row("Longest token", repr(decoded))
    except UnicodeDecodeError:
        stats_table.add_row("Longest token", "<unable to decode>")

    console.print()
    console.print(stats_table)

    if profiler is not None:
        console.print()
        console.print(Panel("[bold]Profiling Results - Top 20 Functions[/]", expand=False))
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(20)
        console.print(s.getvalue())

        profile_path = output_dir / "tinystories_bpe_training_profile.txt"
        with open(profile_path, "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats()
        logger.info(f"Full profiling results saved to: {profile_path}")

    # Summary
    console.print()
    if enable_tracemalloc:
        memory_summary = f"{peak / 1024**3:.2f} GB RAM"
    else:
        memory_summary = "memory tracking disabled"
    summary = (
        f"Training completed in [green]{training_time/60:.2f} minutes[/] using {memory_summary}. "
        f"Vocabulary: [cyan]{len(vocab)}[/] tokens, [cyan]{len(merges)}[/] merges. "
        f"Longest token: [yellow]{longest_len}[/] bytes."
    )
    console.print(Panel(summary, title="[bold]Summary[/]", border_style="blue"))


if __name__ == "__main__":
    main()
