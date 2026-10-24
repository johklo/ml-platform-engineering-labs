"""Synthetic-card object detection: explainable color baseline and optional YOLOv8."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import ColSpec, Schema
from PIL import Image

from ml_labs.common import read_json, record_context, setup_tracking, verify_dataset, write_json


def _read_box(label_path: Path) -> list[float] | None:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    fields = text.split()
    if len(fields) != 5 or fields[0] != "0":
        raise ValueError(f"Expected one normalized synthetic-card box in {label_path}.")
    class_id, x, y, width, height = map(float, fields)
    if class_id != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError(f"Invalid normalized box in {label_path}.")
    return [x, y, width, height]


def detect_color_component(image: Image.Image, min_pixels: int = 64) -> list[int] | None:
    """Detect the generated card's uniform fill and return its pixel xyxy box.

    This intentionally narrow heuristic is a baseline, not a learned detector. The
    synthetic generator uses a single uniform card fill over pixel-noise backgrounds.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    index = int(np.argmax(counts))
    if counts[index] < min_pixels:
        return None
    mask = np.all(rgb == colors[index], axis=2)
    ys, xs = np.where(mask)
    if xs.size < min_pixels:
        return None
    height, width = rgb.shape[:2]
    # The generator draws a two-pixel border over the solid card fill.
    return [
        max(0, int(xs.min()) - 2),
        max(0, int(ys.min()) - 2),
        min(width - 1, int(xs.max()) + 2),
        min(height - 1, int(ys.max()) + 2),
    ]


def _iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_card_split(data: str | Path, split: str, backend: str = "color_component_baseline",
                        model_path: str | Path | None = None) -> dict[str, float | int]:
    root = Path(data)
    ious: list[float] = []
    detected = 0
    total = 0
    negative_images = 0
    false_positives = 0
    yolo_model = None
    if backend == "yolov8":
        if model_path is None:
            raise ValueError("YOLO evaluation requires a trained best.pt model path.")
        from ultralytics import YOLO

        yolo_model = YOLO(str(model_path))
    for image_path in sorted((root / "images" / split).glob("*.png")):
        ground_truth = _read_box(root / "labels" / split / f"{image_path.stem}.txt")
        if backend == "color_component_baseline":
            with Image.open(image_path) as image:
                prediction = detect_color_component(image)
        elif backend == "yolov8":
            result = yolo_model.predict(source=str(image_path), verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            prediction = boxes[int(result.boxes.conf.argmax())].tolist() if len(boxes) else None
        else:
            raise ValueError(f"Unknown card detector backend: {backend}")
        if prediction is not None:
            detected += 1
        if ground_truth is not None:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            center_x, center_y, box_width, box_height = ground_truth
            ground_truth = [
                (center_x - box_width / 2) * image_width,
                (center_y - box_height / 2) * image_height,
                (center_x + box_width / 2) * image_width,
                (center_y + box_height / 2) * image_height,
            ]
        if ground_truth is None:
            negative_images += 1
            false_positives += int(prediction is not None)
        else:
            ious.append(_iou(prediction, ground_truth))
        total += 1
    if not ious:
        raise ValueError(f"No positive card images found in split {split!r} under {root}.")
    return {
        "images": total,
        "detection_rate": detected / total,
        "mean_iou": float(np.mean(ious)),
        "box_recall_iou_0_5": float(np.mean(np.asarray(ious) >= 0.5)),
        "negative_images": negative_images,
        "false_positives": false_positives,
    }


class CardDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.config = read_json(context.artifacts["config"])
        self.yolo = None
        if self.config["backend"] == "yolov8":
            from ultralytics import YOLO

            self.yolo = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        results = []
        for record in model_input.to_dict(orient="records"):
            path = record.get("image_path")
            encoded = record.get("image_base64")
            has_path = isinstance(path, str) and bool(path)
            has_bytes = isinstance(encoded, str) and bool(encoded)
            if has_path == has_bytes:
                raise ValueError("Provide exactly one of image_path or image_base64.")
            if has_bytes:
                if len(encoded) > 14_000_000:
                    raise ValueError("Encoded image exceeds the 10 MB input limit.")
                image_source = BytesIO(base64.b64decode(encoded, validate=True))
            else:
                if not Path(path).is_file():
                    raise ValueError(f"image_path must point to an existing image: {path!r}")
                image_source = path
            with Image.open(image_source) as source:
                image = source.convert("RGB")
            if self.config["backend"] == "color_component_baseline":
                box = detect_color_component(image)
            else:
                predictions = self.yolo.predict(source=image, verbose=False)[0]
                boxes = predictions.boxes.xyxy.cpu().numpy()
                box = boxes[int(predictions.boxes.conf.argmax())].tolist() if len(boxes) else None
            results.append({
                "image_path": path if has_path else "",
                "detected": box is not None,
                "x_min": float("nan") if box is None else float(box[0]),
                "y_min": float("nan") if box is None else float(box[1]),
                "x_max": float("nan") if box is None else float(box[2]),
                "y_max": float("nan") if box is None else float(box[3]),
                "backend": self.config["backend"],
            })
        return pd.DataFrame(results)


def train_cards(data: str | Path, output: str | Path, backend: str = "color_component_baseline",
                work: str | Path = "work", epochs: int = 5) -> Path:
    data, output = Path(data), Path(output)
    manifest = verify_dataset(data)
    output.mkdir(parents=True, exist_ok=True)
    if backend not in {"color_component_baseline", "yolov8"}:
        raise ValueError("backend must be color_component_baseline or yolov8.")
    artifacts: dict[str, str] = {}
    if backend == "color_component_baseline":
        config_path = write_json(output / "detector.json", {"backend": backend, "min_pixels": 64})
        artifacts["config"] = str(config_path)
    else:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the optional detector with `pip install -e .[vision]`.") from exc
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        import yaml

        runtime_dataset = output / "dataset.yaml"
        runtime_dataset.write_text(yaml.safe_dump({
            "path": str(data.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {0: "synthetic_id_card"},
        }), encoding="utf-8")
        trainer = YOLO("yolov8n.pt")
        trainer.train(
            data=str(runtime_dataset.resolve()), epochs=epochs, imgsz=128, device="cpu",
            project=str(output / "yolo"), name="synthetic-cards", exist_ok=True, verbose=False,
        )
        weights = Path(trainer.trainer.save_dir) / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO training did not produce {weights}.")
        config_path = write_json(output / "detector.json", {"backend": backend})
        artifacts.update({"config": str(config_path), "weights": str(weights)})
    metrics = evaluate_card_split(
        data, "val", backend, artifacts.get("weights")
    )
    setup_tracking(work)
    with mlflow.start_run() as run:
        record_context(manifest, {"backend": backend, "epochs": epochs})
        mlflow.log_metrics({f"validation_{key}": float(value) for key, value in metrics.items()})
        sample_image = next((data / "images" / "val").glob("*.png"))
        sample = pd.DataFrame({"image_path": [str(sample_image.resolve())]})
        pyfunc = CardDetector()
        fake_context = type("Context", (), {"artifacts": {
            key: str(value) for key, value in artifacts.items()
        }})()
        pyfunc.load_context(fake_context)
        mlflow.pyfunc.log_model(
            "model", python_model=CardDetector(), artifacts=artifacts,
            code_paths=[str(Path(__file__).parent)],
            signature=ModelSignature(
                inputs=Schema([
                    ColSpec("string", "image_path", required=False),
                    ColSpec("string", "image_base64", required=False),
                ]),
                outputs=infer_signature(sample, pyfunc.predict(None, sample)).outputs,
            ),
            input_example=sample,
            pip_requirements=[
                "mlflow==2.16.2", "numpy==1.26.4", "pandas==2.2.3", "pillow==11.0.0",
            ] + ([
                "ultralytics==8.3.40", "torch==2.5.1", "torchvision==0.20.1",
            ] if backend == "yolov8" else []),
        )
        candidate = {
            "task": "cards",
            "backend": backend,
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "tracking_uri": mlflow.get_tracking_uri(),
            "data_path": str(data.resolve()),
            "validation": metrics,
            "dataset": manifest,
        }
        write_json(output / "candidate.json", candidate)
    return output / "candidate.json"


def evaluate_cards(candidate_path: str | Path, data: str | Path, output: str | Path,
                   min_iou: float = 0.75) -> dict:
    candidate = read_json(candidate_path)
    verify_dataset(data, candidate["dataset"])
    if candidate["task"] != "cards":
        raise ValueError("Candidate is not a cards model.")
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    # Reload through MLflow to ensure the packaged inference artifact is usable.
    model = mlflow.pyfunc.load_model(candidate["model_uri"])
    validation_iou = float(candidate["validation"]["mean_iou"])
    if validation_iou < min_iou:
        report = {
            "task": "cards", "passed": False, "min_mean_iou": min_iou,
            "validation_mean_iou": validation_iou, "split": "validation",
            "test_policy": "Test split remains untouched unless validation passes.",
            "candidate": candidate,
        }
        write_json(output, report)
        with mlflow.start_run(run_id=candidate["run_id"]):
            mlflow.log_dict(report, "evaluation/validation.json")
            mlflow.set_tag("validation_passed", "false")
        raise ValueError(f"Detector rejected by validation IoU gate; report: {output}")
    test_images = sorted((Path(data) / "images" / "test").glob("*.png"))
    if not test_images:
        raise ValueError("The test split is empty.")
    predictions = model.predict(pd.DataFrame({"image_path": [str(p.resolve()) for p in test_images]}))
    per_image = []
    negative_images = 0
    false_positives = 0
    for row in predictions.itertuples():
        predicted = None if not row.detected else [row.x_min, row.y_min, row.x_max, row.y_max]
        actual = _read_box(Path(data) / "labels" / "test" / f"{Path(row.image_path).stem}.txt")
        with Image.open(row.image_path) as image:
            size = image.size
        actual_pixels = None if actual is None else [
            (actual[0] - actual[2] / 2) * size[0],
            (actual[1] - actual[3] / 2) * size[1],
            (actual[0] + actual[2] / 2) * size[0],
            (actual[1] + actual[3] / 2) * size[1],
        ]
        if actual_pixels is None:
            negative_images += 1
            false_positives += int(predicted is not None)
        else:
            per_image.append(_iou(predicted, actual_pixels))
    if not per_image:
        raise ValueError("The test split has no positive card images.")
    metrics = {
        "images": len(test_images),
        "mean_iou": float(np.mean(per_image)),
        "box_recall_iou_0_5": float(np.mean(np.asarray(per_image) >= 0.5)),
        "negative_images": negative_images,
        "false_positives": false_positives,
    }
    report = {
        "task": "cards", "passed": metrics["mean_iou"] >= min_iou,
        "min_mean_iou": min_iou, "validation_mean_iou": validation_iou,
        "metrics": metrics, "candidate": candidate,
        "split": "test",
        "dataset_note": "Synthetic ID-card-shaped rectangles only; no real identity documents or OCR.",
        "test_policy": "Test split opened after passing the validation IoU gate.",
    }
    write_json(output, report)
    with mlflow.start_run(run_id=candidate["run_id"]):
        mlflow.log_metrics({f"test_{key}": float(value) for key, value in metrics.items()})
        mlflow.log_dict(report, "evaluation/test.json")
        mlflow.set_tag("test_passed", str(report["passed"]).lower())
    if not report["passed"]:
        raise ValueError(f"Model gate rejected mean IoU below {min_iou}; report: {output}")
    return report
