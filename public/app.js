let predictorBundle = null;

async function loadBundle() {
  const response = await fetch("./data/model-bundle.json");
  if (!response.ok) {
    throw new Error("Unable to load model bundle");
  }
  return response.json();
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function sigmoid(value) {
  if (value < -40) {
    return 0;
  }
  if (value > 40) {
    return 1;
  }
  return 1 / (1 + Math.exp(-value));
}

function toFeatureValue(featureType, rawValue) {
  if (featureType === "binary") {
    return String(rawValue).trim().toLowerCase() === "yes" ? 1 : 0;
  }
  return Number(rawValue);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setOverview(data) {
  const { dataset, trainingInfo, metrics, featureImpact, createdAt } = data;

  setText("hero-accuracy", metrics.accuracyLabel);
  setText("hero-dataset", dataset.totalStudents.toLocaleString());
  setText("hero-rate", dataset.placementRateLabel);
  setText("metric-accuracy", metrics.accuracyLabel);
  setText("metric-precision", metrics.precisionLabel);
  setText("metric-recall", metrics.recallLabel);
  setText("metric-f1", metrics.f1ScoreLabel);
  setText("total-students", dataset.totalStudents.toLocaleString());
  setText("placed-count", dataset.placedCount.toLocaleString());
  setText("not-placed-count", dataset.notPlacedCount.toLocaleString());
  setText("split-info", `${trainingInfo.trainSize} / ${trainingInfo.testSize}`);
  setText("tp", metrics.confusionMatrix.truePositive.toLocaleString());
  setText("fp", metrics.confusionMatrix.falsePositive.toLocaleString());
  setText("tn", metrics.confusionMatrix.trueNegative.toLocaleString());
  setText("fn", metrics.confusionMatrix.falseNegative.toLocaleString());
  setText(
    "trained-at",
    `Trained ${new Date(createdAt).toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric"
    })}`
  );

  const featureContainer = document.getElementById("feature-impact");
  featureContainer.innerHTML = "";
  featureImpact.forEach((feature) => {
    const wrapper = document.createElement("div");
    wrapper.className = "feature-row";
    wrapper.innerHTML = `
      <div class="feature-meta">
        <strong>${feature.label}</strong>
        <span>${(feature.impact * 100).toFixed(0)}%</span>
      </div>
      <div class="feature-bar">
        <span style="width:${(feature.impact * 100).toFixed(1)}%"></span>
      </div>
    `;
    featureContainer.appendChild(wrapper);
  });
}

function predictPlacement(payload) {
  const { featureConfig, means, stds, weights, bias } = predictorBundle;
  const scaledRow = featureConfig.map((feature, index) => {
    const rawValue = payload[feature.key];
    if (rawValue === undefined || rawValue === null || rawValue === "") {
      throw new Error(`Missing field: ${feature.label}`);
    }
    const numericValue = toFeatureValue(feature.type, rawValue);
    return (numericValue - means[index]) / stds[index];
  });

  const linear =
    bias + scaledRow.reduce((sum, value, index) => sum + value * weights[index], 0);
  const probability = sigmoid(linear);
  const confidence = Math.abs(probability - 0.5) * 2;

  return {
    prediction: probability >= 0.5 ? "Placed" : "NotPlaced",
    probabilityLabel: formatPercent(probability),
    confidenceLabel: formatPercent(confidence)
  };
}

function handlePrediction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  const result = predictPlacement(payload);
  setText("prediction-label", result.prediction);
  setText("probability-value", result.probabilityLabel);
  setText("confidence-value", result.confidenceLabel);
  document.getElementById("probability-bar").style.width = result.probabilityLabel;

  document.getElementById("prediction-copy").textContent =
    result.prediction === "Placed"
      ? "The model sees this profile as likely to get placed based on academic strength, training, and overall readiness."
      : "The model suggests this profile is currently at risk. Improving aptitude score, training exposure, projects, and academic consistency can help.";
}

async function initialize() {
  try {
    const bundle = await loadBundle();
    predictorBundle = bundle.predictor;
    setOverview(bundle.overview);
    document
      .getElementById("prediction-form")
      .addEventListener("submit", handlePrediction);
  } catch (error) {
    setText("prediction-label", "Unable to load dashboard");
    document.getElementById("prediction-copy").textContent = error.message;
  }
}

initialize();
