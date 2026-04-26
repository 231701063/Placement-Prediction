from pathlib import Path


PORT = 3000
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "placementdata.csv"
PUBLIC_DIR = BASE_DIR / "public"

FEATURE_CONFIG = [
    {"key": "CGPA", "type": "number", "label": "CGPA"},
    {"key": "Internships", "type": "number", "label": "Internships"},
    {"key": "Projects", "type": "number", "label": "Projects"},
    {
        "key": "Workshops/Certifications",
        "type": "number",
        "label": "Workshops & Certifications",
    },
    {"key": "AptitudeTestScore", "type": "number", "label": "Aptitude Test Score"},
    {"key": "SoftSkillsRating", "type": "number", "label": "Soft Skills Rating"},
    {
        "key": "ExtracurricularActivities",
        "type": "binary",
        "label": "Extracurricular Activities",
    },
    {"key": "PlacementTraining", "type": "binary", "label": "Placement Training"},
    {"key": "SSC_Marks", "type": "number", "label": "SSC Marks"},
    {"key": "HSC_Marks", "type": "number", "label": "HSC Marks"},
]
