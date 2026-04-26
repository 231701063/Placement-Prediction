import json

from config import FEATURE_CONFIG, PUBLIC_DIR
from model_service import MODEL_BUNDLE, build_overview_payload


def export_bundle():
    bundle = {
        "overview": build_overview_payload(),
        "predictor": {
            "featureConfig": FEATURE_CONFIG,
            "means": MODEL_BUNDLE["scalingStats"]["means"],
            "stds": MODEL_BUNDLE["scalingStats"]["stds"],
            "weights": MODEL_BUNDLE["model"]["weights"],
            "bias": MODEL_BUNDLE["model"]["bias"],
        },
    }

    data_dir = PUBLIC_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "model-bundle.json"
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Static bundle written to {output_path}")


if __name__ == "__main__":
    export_bundle()
