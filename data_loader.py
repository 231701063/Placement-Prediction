from config import DATASET_PATH, FEATURE_CONFIG


def parse_csv(text):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    headers = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        values = line.split(",")
        row = {header: values[index] for index, header in enumerate(headers)}
        rows.append(row)
    return rows


def load_dataset():
    csv_text = DATASET_PATH.read_text(encoding="utf-8")
    return parse_csv(csv_text)


def to_feature_value(feature_type, raw_value):
    if feature_type == "binary":
        return 1.0 if str(raw_value).strip().lower() == "yes" else 0.0
    return float(raw_value)


def split_dataset(rows):
    train_size = int(len(rows) * 0.8)
    return rows[:train_size], rows[train_size:]


def build_matrix(rows):
    features = []
    labels = []
    for row in rows:
        feature_row = [
            to_feature_value(feature["type"], row[feature["key"]])
            for feature in FEATURE_CONFIG
        ]
        features.append(feature_row)
        labels.append(1 if row["PlacementStatus"] == "Placed" else 0)
    return features, labels


def summarize_dataset(rows):
    total_students = len(rows)
    placed_count = sum(1 for row in rows if row["PlacementStatus"] == "Placed")

    def average(key):
        return sum(float(row[key]) for row in rows) / total_students

    return {
        "totalStudents": total_students,
        "placedCount": placed_count,
        "notPlacedCount": total_students - placed_count,
        "placementRate": placed_count / total_students,
        "averages": {
            "cgpa": average("CGPA"),
            "aptitude": average("AptitudeTestScore"),
            "softSkills": average("SoftSkillsRating"),
            "sscMarks": average("SSC_Marks"),
            "hscMarks": average("HSC_Marks"),
        },
    }
