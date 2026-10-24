import os
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from ml_labs.cards import train_cards
from ml_labs.common import read_json
from ml_labs.data import prepare_cards


@pytest.mark.vision
@pytest.mark.skipif(
    os.environ.get("ML_LABS_RUN_VISION") != "1",
    reason="Opt-in CPU training and pretrained weight download.",
)
def test_yolo_training_packages_weights_and_reloads_for_inference(tmp_path: Path):
    pytest.importorskip("ultralytics")
    data = prepare_cards(tmp_path / "cards", train=12, validation=4, test=4)
    candidate = train_cards(
        data, tmp_path / "trained", backend="yolov8", epochs=1, work=tmp_path / "tracking",
    )
    metadata = read_json(candidate)
    mlflow.set_tracking_uri(metadata["tracking_uri"])
    model = mlflow.pyfunc.load_model(metadata["model_uri"])
    image = next((data / "images" / "test").glob("*.png"))
    output = model.predict(pd.DataFrame({"image_path": [str(image)]}))
    assert len(output) == 1
    assert output.iloc[0]["backend"] == "yolov8"
