"""RSNA Knee Abnormality Detection — baseline package."""

# The 12 targets, exact order + spelling (verified against sample_submission.csv).
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

ID_COL = "StudyInstanceUID"

__all__ = ["TARGETS", "ID_COL"]
