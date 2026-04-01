#!/usr/bin/env python3
"""
wandb_cleanup.py — Delete model/checkpoint artifacts by run group or run name prefix.
Metrics (wandb-history, wandb-events, metadata, run_table) are never touched.

Usage:
  # Dry run first (default — prints what would be deleted)
  python wandb_cleanup.py --entity ashwin-ms-does-ai --project transformer-room-baseline --group my-group

  # Filter by run name prefix instead of group
  python wandb_cleanup.py --entity ashwin-ms-does-ai --project transformer-room-baseline --prefix mb_sweep_

  # Actually delete
  python wandb_cleanup.py ... --delete

  # Limit artifact types (default: model checkpoint)
  python wandb_cleanup.py ... --types model

  # Skip confirmation prompt
  python wandb_cleanup.py ... --delete --yes
"""

import argparse
import sys
import wandb
from wandb import Api

# Artifact types that are safe to delete (never metrics/system types)
DELETABLE_TYPES = {"model", "checkpoint"}

# Never touch these — they hold metrics and run metadata
PROTECTED_TYPES = {"wandb-history", "wandb-events", "metadata", "run_table"}


def get_runs(api: Api, entity: str, project: str, group: str | None, prefix: str | None):
    filters = {}
    if group:
        filters["group"] = group

    runs = api.runs(f"{entity}/{project}", filters=filters or None)

    if prefix:
        runs = [r for r in runs if r.name.startswith(prefix)]
    else:
        runs = list(runs)

    return runs


def collect_artifacts(api: Api, entity: str, project: str, runs: list, types: set[str]):
    """Return list of (artifact, run_name) tuples to be deleted."""
    targets = []
    for run in runs:
        try:
            for artifact in run.logged_artifacts():
                atype = artifact.type
                if atype in PROTECTED_TYPES:
                    continue
                if atype not in types:
                    continue
                targets.append((artifact, run.name))
        except Exception as e:
            print(f"  [warn] Could not fetch artifacts for run {run.name}: {e}")
    return targets


def main():
    parser = argparse.ArgumentParser(description="Clean up W&B model/checkpoint artifacts.")
    parser.add_argument("--entity",  required=True, help="W&B entity (username or team)")
    parser.add_argument("--project", required=True, help="W&B project name")

    group_filter = parser.add_mutually_exclusive_group(required=True)
    group_filter.add_argument("--group",  help="Run group name to target")
    group_filter.add_argument("--prefix", help="Run name prefix to target (e.g. 'mb_sweep_')")

    parser.add_argument(
        "--types",
        nargs="+",
        default=list(DELETABLE_TYPES),
        help=f"Artifact types to delete. Choices: {sorted(DELETABLE_TYPES)}. Default: all.",
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete (default is dry run)")
    parser.add_argument("--yes",    action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    # Validate requested types
    requested_types = set(args.types)
    bad = requested_types - DELETABLE_TYPES
    if bad:
        print(f"[error] Refusing to target protected or unknown types: {bad}")
        print(f"        Allowed deletable types: {sorted(DELETABLE_TYPES)}")
        sys.exit(1)

    api = Api()

    print(f"\nFetching runs from {args.entity}/{args.project} ...", flush=True)
    runs = get_runs(api, args.entity, args.project, args.group, args.prefix)

    if not runs:
        print("No matching runs found. Exiting.")
        sys.exit(0)

    filter_desc = f"group='{args.group}'" if args.group else f"prefix='{args.prefix}'"
    print(f"Found {len(runs)} run(s) matching {filter_desc}:")
    for r in runs:
        print(f"  {r.name}  (group={r.group or '-'})")

    print(f"\nCollecting {requested_types} artifacts ...", flush=True)
    targets = collect_artifacts(api, args.entity, args.project, runs, requested_types)

    if not targets:
        print("No matching artifacts found. Nothing to do.")
        sys.exit(0)

    print(f"\n{'[DRY RUN] ' if not args.delete else ''}Artifacts to delete ({len(targets)} total):")
    for artifact, run_name in targets:
        size_mb = artifact.size / 1e6 if artifact.size else 0
        print(f"  [{artifact.type}]  {artifact.name}  ({size_mb:.1f} MB)  — logged by run: {run_name}")

    if not args.delete:
        print("\nDry run complete. Pass --delete to actually remove these artifacts.")
        return

    # Confirm
    if not args.yes:
        answer = input(f"\nDelete {len(targets)} artifact(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # Delete
    deleted, failed = 0, 0
    for artifact, run_name in targets:
        try:
            artifact.delete(delete_aliases=True)
            print(f"  ✓ deleted  {artifact.name}")
            deleted += 1
        except Exception as e:
            print(f"  ✗ failed   {artifact.name}: {e}")
            failed += 1

    print(f"\nDone. {deleted} deleted, {failed} failed.")


if __name__ == "__main__":
    main()