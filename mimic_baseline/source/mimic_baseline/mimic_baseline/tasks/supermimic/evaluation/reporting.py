"""Original evaluation reporting functions."""

from __future__ import annotations


import csv


import json


import os


import re


from collections import defaultdict


from pathlib import Path


from typing import Any





def motion_category(name: str) -> str:
    """Derive a stable evaluation category from a motion path/name."""
    normalized = str(name).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) > 1:
        return parts[-2].lower()
    leaf = Path(parts[-1] if parts else "").stem
    token = re.split(r"[_.\-]", leaf, maxsplit=1)
    return token[0].lower() or "uncategorized"


def _write_per_clip_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "suite",
        "name",
        "category",
        "clip_id",
        "clip_start",
        "clip_end",
        "full_clip_frames",
        "steps",
        "return",
        "normalized_return",
        "terminated",
        "completed",
        "outcome",
        "termination_policy",
        "push_speed",
        "root_xy_rmse",
        "root_z_rmse",
        "root_rotation_rmse_rad",
        "joint_rmse",
        "body_position_rmse",
        "body_orientation_rmse_rad",
        "diagnostic_final_dwell_ok",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in columns})


def _write_category_csv(path: Path, grouped: dict[str, dict[str, Any]]) -> None:
    columns = [
        "suite",
        "category",
        "clips",
        "mean_normalized_return",
        "completion_rate",
        "termination_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key in sorted(grouped):
            writer.writerow({column: grouped[key][column] for column in columns})


def write_results(
    output_dir: str | os.PathLike[str],
    records: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Write per-clip JSON/CSV and category aggregates with explicit suite labels."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[record["category"]].append(record)
    grouped: dict[str, dict[str, Any]] = {}
    for category, rows in grouped_records.items():
        count = len(rows)
        grouped[category] = {
            "suite": rows[0]["suite"],
            "category": category,
            "clips": count,
            "mean_normalized_return": sum(
                float(row["normalized_return"]) for row in rows
            )
            / count,
            "completion_rate": sum(bool(row["completed"]) for row in rows) / count,
            "termination_rate": sum(bool(row["terminated"]) for row in rows) / count,
            "outcomes": {
                outcome: sum(row["outcome"] == outcome for row in rows)
                for outcome in sorted({row["outcome"] for row in rows})
            },
        }
    payload = {
        "metadata": metadata or {},
        "by_category": grouped,
        "clips": records,
    }
    (output / "perclip.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_per_clip_csv(output / "perclip.csv", records)
    _write_category_csv(output / "by_category.csv", grouped)
    return grouped
