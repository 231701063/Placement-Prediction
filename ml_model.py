import math

from config import FEATURE_CONFIG


def sigmoid(value):
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def compute_scaling_stats(features):
    feature_count = len(features[0])
    means = [0.0] * feature_count
    stds = [0.0] * feature_count

    for row in features:
        for index, value in enumerate(row):
            means[index] += value

    sample_count = len(features)
    means = [value / sample_count for value in means]

    for row in features:
        for index, value in enumerate(row):
            diff = value - means[index]
            stds[index] += diff * diff

    stds = [math.sqrt(value / sample_count) or 1.0 for value in stds]
    return {"means": means, "stds": stds}


def scale_matrix(features, scaling_stats):
    scaled = []
    for row in features:
        scaled.append(
            [
                (value - scaling_stats["means"][index]) / scaling_stats["stds"][index]
                for index, value in enumerate(row)
            ]
        )
    return scaled


def train_logistic_regression(
    features, labels, learning_rate=0.08, epochs=1600, regularization=0.0008
):
    weights = [0.0] * len(features[0])
    bias = 0.0
    sample_count = len(features)

    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        bias_gradient = 0.0

        for row, actual in zip(features, labels):
            linear = bias + sum(weight * value for weight, value in zip(weights, row))
            prediction = sigmoid(linear)
            error = prediction - actual

            for index, value in enumerate(row):
                gradient[index] += error * value
            bias_gradient += error

        for index in range(len(weights)):
            average_gradient = (
                gradient[index] / sample_count + regularization * weights[index]
            )
            weights[index] -= learning_rate * average_gradient

        bias -= learning_rate * (bias_gradient / sample_count)

    return {"weights": weights, "bias": bias}


def predict_probability(scaled_row, model):
    linear = model["bias"] + sum(
        weight * value for weight, value in zip(model["weights"], scaled_row)
    )
    return sigmoid(linear)


def evaluate_model(features, labels, model):
    correct = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row, actual in zip(features, labels):
        probability = predict_probability(row, model)
        prediction = 1 if probability >= 0.5 else 0

        if prediction == actual:
            correct += 1
        if prediction == 1 and actual == 1:
            true_positive += 1
        elif prediction == 0 and actual == 0:
            true_negative += 1
        elif prediction == 1 and actual == 0:
            false_positive += 1
        else:
            false_negative += 1

    accuracy = correct / len(features)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = (2 * precision * recall) / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1Score": f1_score,
        "confusionMatrix": {
            "truePositive": true_positive,
            "trueNegative": true_negative,
            "falsePositive": false_positive,
            "falseNegative": false_negative,
        },
    }


def build_feature_impact(model):
    max_weight = max(abs(weight) for weight in model["weights"]) or 1.0
    feature_impact = []
    for index, feature in enumerate(FEATURE_CONFIG):
        weight = model["weights"][index]
        feature_impact.append(
            {
                "key": feature["key"],
                "label": feature["label"],
                "weight": weight,
                "impact": abs(weight) / max_weight,
            }
        )
    feature_impact.sort(key=lambda item: item["impact"], reverse=True)
    return feature_impact
