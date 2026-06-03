"""
collect_reconstructions.py

Interactively walks through all prompts in a prompts.json file, displays each
prompt, asks you to paste the model's reconstruction, and saves responses into
a mirrored results JSON ready for score_reconstructions.py.

Supports resuming an interrupted session — already-answered entries are skipped.

Usage:
    python collect_reconstructions.py \
        --prompts results/prompts.json \
        --output results/reconstructions.json \
        [--show-obfuscated]   # also print the obfuscated text above the prompt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "─" * 72


def walk_leaves(node: Any, path: list[str] | None = None):
    """Yield (path, value) for every string leaf in a nested dict."""
    if path is None:
        path = []
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_leaves(v, path + [k])
    elif isinstance(node, str):
        yield path, node


def get_nested(d: dict, path: list[str]) -> Any:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def set_nested(d: dict, path: list[str], value: Any) -> None:
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_multiline(prompt_label: str) -> str:
    """
    Read a multi-line paste from stdin.
    The user signals end-of-input with a blank line, or EOF (Ctrl+D / Ctrl+Z).
    """
    print(prompt_label)
    print(
        "  (Paste the model's response. Press Enter twice — or Ctrl+D — when done.)\n"
    )
    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                # Two consecutive blank lines = done
                break
            lines.append(line)
    except EOFError:
        pass
    # Strip trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def collect_prompt_entries(prompts: dict) -> list[dict]:
    """
    Walk the prompts dict and collect every leaf that is a dict with a
    'prompt' key (the structure produced by build_prompts_from_results).
    Also handles plain string leaves for flexibility.

    Returns a flat list of:
        { path, prompt, obfuscated (optional) }
    """
    entries = []

    def _walk(node, path):
        if isinstance(node, dict):
            if "prompt" in node:
                entries.append(
                    {
                        "path": path,
                        "prompt": node["prompt"],
                        "obfuscated": node.get("obfuscated", ""),
                    }
                )
            else:
                for k, v in node.items():
                    _walk(v, path + [k])
        elif isinstance(node, str):
            # Flat string leaf — treat the string itself as the prompt
            entries.append(
                {
                    "path": path,
                    "prompt": node,
                    "obfuscated": "",
                }
            )

    _walk(prompts, [])
    return entries


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


def collect(args):
    print(f"\nLoading prompts from: {args.prompts}")
    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)

    output_path = Path(args.output)

    # Load existing results so we can resume
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from existing results at: {output_path}")
    else:
        results = {}

    entries = collect_prompt_entries(prompts)
    total = len(entries)

    if total == 0:
        print("No prompt entries found in the file. Check the JSON structure.")
        sys.exit(1)

    # Count already answered
    already_done = sum(
        1 for e in entries if isinstance(get_nested(results, e["path"]), str)
    )
    print(
        f"Found {total} prompts — {already_done} already answered, {total - already_done} remaining.\n"
    )

    answered_this_session = 0

    for idx, entry in enumerate(entries, start=1):
        path = entry["path"]
        path_label = " › ".join(path)

        # Skip if already answered
        existing = get_nested(results, path)
        if isinstance(existing, str):
            print(f"[{idx}/{total}] Skipping (already answered): {path_label}")
            continue

        print(f"\n{SEPARATOR}")
        print(f"[{idx}/{total}]  {path_label}")
        print(SEPARATOR)

        if args.show_obfuscated and entry["obfuscated"]:
            print("\n── Obfuscated text ──")
            print(entry["obfuscated"])

        print("\n── Prompt ──")
        print(entry["prompt"])
        print()

        # Commands
        print("Commands:  [s] skip   [q] quit and save   or just paste the response.")
        print()

        response = read_multiline("── Model response ──")

        if response.lower() in ("s", "skip"):
            print("  Skipped.")
            continue

        if response.lower() in ("q", "quit"):
            print("\nQuitting — saving progress ...")
            save(results, output_path)
            print(f"Saved {answered_this_session} new response(s) to {output_path}")
            sys.exit(0)

        if not response:
            print("  Empty response — skipping.")
            continue

        set_nested(results, path, response)
        answered_this_session += 1

        # Autosave after every response
        save(results, output_path)
        print(f"  ✓ Saved.")

    print(f"\n{SEPARATOR}")
    print(f"All done! {answered_this_session} new response(s) collected this session.")
    save(results, output_path)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively collect model reconstructions for each prompt and save to JSON."
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to prompts.json produced by the obfuscation pipeline.",
    )
    parser.add_argument(
        "--output",
        default="results/reconstructions.json",
        help="Path to write (or resume) the reconstructions JSON.",
    )
    parser.add_argument(
        "--show-obfuscated",
        action="store_true",
        help="Also display the obfuscated source text above each prompt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args)
