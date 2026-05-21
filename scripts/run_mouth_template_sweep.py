from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_DISEASES = ("protrusion", "openbite", "spacing", "caries")
DEFAULT_SEVERITIES = ("0.2", "0.5", "0.8", "1.2")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run infer_mouth_template sweeps over a subset of normal face images.")
    parser.add_argument("--input-dir", default="data/inputs_normal/ffhq_subset")
    parser.add_argument("--output-dir", default="outputs/mouth_template_sweep")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--diseases", nargs="+", default=list(DEFAULT_DISEASES))
    parser.add_argument("--severities", nargs="+", default=list(DEFAULT_SEVERITIES))
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    records: list[dict[str, object]] = []
    for image_path in image_paths:
        for disease in args.diseases:
            case_output_dir = output_dir / image_path.stem / disease
            command = [
                args.python_exe,
                "infer_mouth_template.py",
                "--input",
                str(image_path),
                "--disease",
                disease,
                "--severity-sweep",
                ",".join(args.severities),
                "--output-dir",
                str(case_output_dir),
            ]
            print("Running:", " ".join(command))
            subprocess.run(command, check=True)

            for severity in args.severities:
                severity_dir = case_output_dir / f"severity_{float(severity):.2f}".replace(".", "_")
                summary_path = severity_dir / "summary.json"
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                records.append(
                    {
                        "image_name": image_path.name,
                        "disease": disease,
                        "severity": severity,
                        "detector_mode": summary.get("detector_mode", ""),
                        "teeth_pixels": summary.get("teeth_pixels", 0),
                        "delta_abs_mean": summary.get("delta_abs_mean", 0.0),
                        "delta_abs_max": summary.get("delta_abs_max", 0.0),
                        "texture_pixels": summary.get("texture_pixels", 0),
                        "summary_path": str(summary_path.resolve()),
                    }
                )

    csv_path = output_dir / "summary_table.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_name",
                "disease",
                "severity",
                "detector_mode",
                "teeth_pixels",
                "delta_abs_mean",
                "delta_abs_max",
                "texture_pixels",
                "summary_path",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_images": len(image_paths),
        "diseases": list(args.diseases),
        "severities": list(args.severities),
        "num_records": len(records),
        "summary_table": str(csv_path.resolve()),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
