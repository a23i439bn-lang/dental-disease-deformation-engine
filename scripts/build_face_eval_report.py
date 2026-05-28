from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FACE_REQUIRED_KEYS = ("control_points", "mask_pixels", "delta_abs_mean", "delta_abs_max")
PANEL_IMAGES = (
    ("input", "input.png"),
    ("shaded", "shaded_warped_full.png"),
    ("final", "final_output.png"),
    ("delta", "delta_heatmap.png"),
    ("weight", "deformation_weight_map.png"),
    ("depth", "pseudo_depth_map.png"),
    ("highlight", "pseudo_highlight_map.png"),
    ("3d relief", "pseudo3d_relief_preview.png"),
    ("shading", "shading_map.png"),
    ("landmarks", "landmarks_debug.png"),
)
SUMMARY_FIELDS = [
    "case_id",
    "disease",
    "requested_disease",
    "severity",
    "detector_mode",
    "warp_sigma",
    "feather",
    "control_points",
    "mask_pixels",
    "weight_map_mean",
    "weight_map_max",
    "shading_map_mean",
    "shading_map_max",
    "pseudo_depth_mean",
    "pseudo_depth_max",
    "pseudo_highlight_mean",
    "pseudo_highlight_max",
    "pseudo3d_strength",
    "mediapipe_z_range",
    "mediapipe_chin_z_mean",
    "delta_abs_mean",
    "delta_abs_max",
    "sd_enabled",
    "output_image",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quantitative and visual reports for face dysmorph outputs.")
    parser.add_argument("--output-root", default="outputs", help="Root directory containing inference outputs.")
    parser.add_argument("--report-dir", default="outputs/face_eval_report", help="Directory for report artifacts.")
    parser.add_argument("--max-panels", type=int, default=200, help="Maximum number of visual panels to render.")
    parser.add_argument("--panel-width", type=int, default=260, help="Width of each panel tile.")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def slugify(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_")
    return value[:120] or "case"


def is_face_summary(summary: dict[str, Any], summary_path: Path) -> bool:
    if not all(key in summary for key in FACE_REQUIRED_KEYS):
        return False
    return (summary_path.parent / "lower_face_mask.png").exists()


def load_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not is_face_summary(summary, summary_path):
            continue

        sd_meta = summary.get("sd_refine", {})
        sd_enabled = bool(sd_meta.get("enabled", False)) if isinstance(sd_meta, dict) else False
        case_id = str(summary_path.parent.relative_to(output_root))
        records.append(
            {
                "case_id": case_id,
                "disease": summary.get("disease", ""),
                "requested_disease": summary.get("requested_disease", ""),
                "severity": summary.get("severity", ""),
                "detector_mode": summary.get("detector_mode", ""),
                "warp_sigma": summary.get("warp_sigma", ""),
                "feather": summary.get("feather", ""),
                "control_points": summary.get("control_points", ""),
                "mask_pixels": summary.get("mask_pixels", ""),
                "weight_map_mean": summary.get("weight_map_mean", ""),
                "weight_map_max": summary.get("weight_map_max", ""),
                "shading_map_mean": summary.get("shading_map_mean", ""),
                "shading_map_max": summary.get("shading_map_max", ""),
                "pseudo_depth_mean": summary.get("pseudo_depth_mean", ""),
                "pseudo_depth_max": summary.get("pseudo_depth_max", ""),
                "pseudo_highlight_mean": summary.get("pseudo_highlight_mean", ""),
                "pseudo_highlight_max": summary.get("pseudo_highlight_max", ""),
                "pseudo3d_strength": summary.get("pseudo3d_strength", ""),
                "mediapipe_z_range": summary.get("mediapipe_z_range", ""),
                "mediapipe_chin_z_mean": summary.get("mediapipe_chin_z_mean", ""),
                "delta_abs_mean": summary.get("delta_abs_mean", ""),
                "delta_abs_max": summary.get("delta_abs_max", ""),
                "sd_enabled": sd_enabled,
                "output_image": summary.get("output_image", str((summary_path.parent / "final_output.png").resolve())),
                "summary_path": str(summary_path.resolve()),
                "_case_dir": summary_path.parent,
            }
        )
    return records


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("disease", "")), str(record.get("severity", "")))].append(record)

    aggregate: list[dict[str, Any]] = []
    metric_fields = (
        "delta_abs_mean",
        "delta_abs_max",
        "mask_pixels",
        "weight_map_mean",
        "weight_map_max",
        "shading_map_mean",
        "shading_map_max",
        "pseudo_depth_mean",
        "pseudo_depth_max",
        "pseudo_highlight_mean",
        "pseudo_highlight_max",
    )
    for (disease, severity), rows in sorted(groups.items()):
        item: dict[str, Any] = {"disease": disease, "severity": severity, "num_cases": len(rows)}
        for field in metric_fields:
            values = [value for value in (to_float(row.get(field)) for row in rows) if value is not None]
            item[f"{field}_avg"] = float(np.mean(values)) if values else ""
            item[f"{field}_max"] = float(np.max(values)) if values else ""
        aggregate.append(item)
    return aggregate


def make_tile(image: np.ndarray | None, label: str, width: int) -> np.ndarray:
    tile_h = int(width * 0.78)
    if image is None:
        tile = np.full((tile_h, width, 3), 235, dtype=np.uint8)
        cv2.putText(tile, "missing", (18, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
    else:
        h, w = image.shape[:2]
        scale = min(width / max(w, 1), tile_h / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        tile = np.full((tile_h, width, 3), 245, dtype=np.uint8)
        top = (tile_h - resized.shape[0]) // 2
        left = (width - resized.shape[1]) // 2
        tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized

    label_h = 32
    labeled = np.full((tile_h + label_h, width, 3), 255, dtype=np.uint8)
    labeled[:tile_h] = tile
    cv2.putText(labeled, label, (10, tile_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return labeled


def render_panels(records: list[dict[str, Any]], report_dir: Path, panel_width: int, max_panels: int) -> list[str]:
    panel_paths: list[str] = []
    panel_dir = report_dir / "panels"
    for record in records[:max_panels]:
        case_dir = Path(record["_case_dir"])
        tiles = [make_tile(read_image(case_dir / filename), label, panel_width) for label, filename in PANEL_IMAGES]
        panel = cv2.hconcat(tiles)
        header_h = 40
        header = np.full((header_h, panel.shape[1], 3), 255, dtype=np.uint8)
        title = f"{record.get('disease', '')}  severity={record.get('severity', '')}  case={record.get('case_id', '')}"
        cv2.putText(header, title[:150], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 1, cv2.LINE_AA)
        out = cv2.vconcat([header, panel])
        panel_path = panel_dir / f"{slugify(str(record.get('case_id', 'case')))}.png"
        write_image(panel_path, out)
        panel_paths.append(str(panel_path.resolve()))
    return panel_paths


def write_markdown(report_dir: Path, records: list[dict[str, Any]], aggregate: list[dict[str, Any]], panel_paths: list[str]) -> None:
    lines = [
        "# Face Dysmorph Evaluation Report",
        "",
        f"- cases: {len(records)}",
        f"- visual_panels: {len(panel_paths)}",
        "",
        "## Aggregate",
        "",
        "| disease | severity | n | delta_mean_avg | delta_max_avg | weight_mean_avg | shading_mean_avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {disease} | {severity} | {num_cases} | {delta_abs_mean_avg:.4f} | {delta_abs_max_avg:.4f} | {weight_map_mean_avg} | {shading_map_mean_avg} |".format(
                disease=row.get("disease", ""),
                severity=row.get("severity", ""),
                num_cases=row.get("num_cases", 0),
                delta_abs_mean_avg=float(row.get("delta_abs_mean_avg") or 0.0),
                delta_abs_max_avg=float(row.get("delta_abs_max_avg") or 0.0),
                weight_map_mean_avg=(
                    f"{float(row['weight_map_mean_avg']):.4f}" if row.get("weight_map_mean_avg") != "" else ""
                ),
                shading_map_mean_avg=(
                    f"{float(row['shading_map_mean_avg']):.4f}" if row.get("shading_map_mean_avg") != "" else ""
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `summary_table.csv`",
            "- `aggregate_by_disease_severity.csv`",
            "- `panels/`",
            "",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(output_root)
    aggregate = aggregate_records(records)
    panel_paths = render_panels(records, report_dir, args.panel_width, args.max_panels)

    write_csv(report_dir / "summary_table.csv", records, SUMMARY_FIELDS)
    aggregate_fields = [
        "disease",
        "severity",
        "num_cases",
        "delta_abs_mean_avg",
        "delta_abs_mean_max",
        "delta_abs_max_avg",
        "delta_abs_max_max",
        "mask_pixels_avg",
        "mask_pixels_max",
        "weight_map_mean_avg",
        "weight_map_mean_max",
        "weight_map_max_avg",
        "weight_map_max_max",
            "shading_map_mean_avg",
            "shading_map_mean_max",
            "shading_map_max_avg",
            "shading_map_max_max",
            "pseudo_depth_mean_avg",
            "pseudo_depth_mean_max",
            "pseudo_depth_max_avg",
            "pseudo_depth_max_max",
            "pseudo_highlight_mean_avg",
            "pseudo_highlight_mean_max",
            "pseudo_highlight_max_avg",
            "pseudo_highlight_max_max",
        ]
    write_csv(report_dir / "aggregate_by_disease_severity.csv", aggregate, aggregate_fields)
    write_markdown(report_dir, records, aggregate, panel_paths)

    manifest = {
        "output_root": str(output_root.resolve()),
        "report_dir": str(report_dir.resolve()),
        "num_cases": len(records),
        "num_panels": len(panel_paths),
        "summary_table": str((report_dir / "summary_table.csv").resolve()),
        "aggregate_table": str((report_dir / "aggregate_by_disease_severity.csv").resolve()),
        "report": str((report_dir / "report.md").resolve()),
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
