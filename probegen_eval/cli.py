"""Command line: `generate` predictions and `evaluate` them."""

import argparse
import json
from pathlib import Path

from .data import TASKS, load_manifest
from .evaluate import evaluate
from .generate import MODELS, generate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m probegen_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate predictions through a hosted image generator")
    ev = sub.add_parser("evaluate", help="Score a folder of predictions")
    for command in (gen, ev):
        command.add_argument("--dataset", choices=TASKS, required=True)
        command.add_argument("--data-root", type=Path, default=Path("data"))
        command.add_argument("--n-samples", type=int, help="Use only the first N samples")
    gen.add_argument("--model", choices=MODELS, required=True)
    gen.add_argument("--output-dir", type=Path, help="Default: outputs/<model>/<dataset>")
    ev.add_argument("--predictions", type=Path, required=True, help="Folder of predictions")
    ev.add_argument("--output", type=Path, help="Optional JSON report with per-sample scores")
    args = parser.parse_args(argv)

    samples = load_manifest(Path("manifests") / f"{args.dataset}.json", args.n_samples)
    if args.command == "generate":
        output_dir = args.output_dir or Path("outputs") / args.model / args.dataset
        return generate(args.dataset, args.model, samples, args.data_root, output_dir)

    report = evaluate(args.dataset, samples, args.data_root, args.predictions)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("dataset", "selected", "scored", "metrics")}, indent=2))
    if report["missing"] or report["failed"]:
        print(f"Not scored: {len(report['missing'])} missing, {len(report['failed'])} failed"
              + (f" (see {args.output})" if args.output else ""))
    return 0
