from datetime import datetime, timezone

from config import FEATURE_CONFIG
from data_loader import (
    build_matrix,
    load_dataset,
    split_dataset,
    summarize_dataset,
    to_feature_value,
)
from ml_model import (
    build_feature_impact,
    compute_scaling_stats,
    evaluate_model,
    predict_probability,
    scale_matrix,
    train_logistic_regression,
)


def format_percent(value):
    return f"{value * 100:.2f}%"


def create_model_bundle():
    rows = load_dataset()
    train_rows, test_rows = split_dataset(rows)
    train_x, train_y = build_matrix(train_rows)
    test_x, test_y = build_matrix(test_rows)

    scaling_stats = compute_scaling_stats(train_x)
    scaled_train_x = scale_matrix(train_x, scaling_stats)
    scaled_test_x = scale_matrix(test_x, scaling_stats)
    model = train_logistic_regression(scaled_train_x, train_y)
    metrics = evaluate_model(scaled_test_x, test_y, model)

    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "model": model,
        "scalingStats": scaling_stats,
        "metrics": metrics,
        "datasetSummary": summarize_dataset(rows),
        "featureImpact": build_feature_impact(model),
        "trainingInfo": {
            "algorithm": "Logistic Regression",
            "trainSize": len(train_rows),
            "testSize": len(test_rows),
            "featureCount": len(FEATURE_CONFIG),
        },
    }


MODEL_BUNDLE = create_model_bundle()


def predict_placement(payload):
    scaled_row = []
    for index, feature in enumerate(FEATURE_CONFIG):
        value = payload.get(feature["key"])
        if value in (None, ""):
            raise ValueError(f"Missing field: {feature['label']}")
        numeric = to_feature_value(feature["type"], value)
        scaled_value = (
            numeric - MODEL_BUNDLE["scalingStats"]["means"][index]
        ) / MODEL_BUNDLE["scalingStats"]["stds"][index]
        scaled_row.append(scaled_value)

    probability = predict_probability(scaled_row, MODEL_BUNDLE["model"])
    confidence = abs(probability - 0.5) * 2
    return {
        "probability": probability,
        "prediction": "Placed" if probability >= 0.5 else "NotPlaced",
        "confidence": confidence,
        "probabilityLabel": format_percent(probability),
        "confidenceLabel": format_percent(confidence),
    }


def build_overview_payload():
    dataset = dict(MODEL_BUNDLE["datasetSummary"])
    dataset["placementRateLabel"] = format_percent(dataset["placementRate"])

    metrics = dict(MODEL_BUNDLE["metrics"])
    metrics["accuracyLabel"] = format_percent(metrics["accuracy"])
    metrics["precisionLabel"] = format_percent(metrics["precision"])
    metrics["recallLabel"] = format_percent(metrics["recall"])
    metrics["f1ScoreLabel"] = format_percent(metrics["f1Score"])

    return {
        "dataset": dataset,
        "trainingInfo": MODEL_BUNDLE["trainingInfo"],
        "metrics": metrics,
        "featureImpact": MODEL_BUNDLE["featureImpact"],
        "createdAt": MODEL_BUNDLE["createdAt"],
    }
