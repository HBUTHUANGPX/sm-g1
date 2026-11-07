"""Coverage-only reports for trajectories with all termination disabled."""

import csv
import json
from collections import defaultdict
from pathlib import Path


def write_clean_results(output_dir, records, metadata=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in row.items() if k != "completed"} for row in records]
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    grouped = {
        category: {
            "category": category,
            "clips": len(items),
            "coverage_rate": sum(bool(r["covered_to_end"]) for r in items) / len(items),
            "tracking_success": None,
            "mean_normalized_return": sum(r["normalized_return"] for r in items)
            / len(items),
        }
        for category, items in categories.items()
    }
    (output / "perclip.json").write_text(
        json.dumps(
            {
                "metadata": metadata or {},
                "clips": rows,
                "by_category": grouped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, values in [("perclip", rows), ("by_category", list(grouped.values()))]:
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(values[0]) if values else []
            )
            writer.writeheader()
            writer.writerows(values)
    return grouped
