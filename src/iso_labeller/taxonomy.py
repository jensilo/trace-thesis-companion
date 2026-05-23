ISO_TAXONOMY: dict[str, list[str]] = {
    "Functional suitability": [
        "Functional completeness",
        "Functional appropriateness",
        "Functional correctness",
    ],
    "Performance efficiency": [
        "Time behaviour",
        "Resource utilization",
        "Capacity",
    ],
    "Compatibility": [
        "Co-existence",
        "Interoperability",
    ],
    "Interaction capability": [
        "Appropriateness recognizability",
        "Learnability",
        "Operability",
        "User error protection",
        "User engagement",
        "Inclusivity",
        "User assistance",
        "Self descriptiveness",
    ],
    "Reliability": [
        "Faultlessness",
        "Availability",
        "Fault tolerance",
        "Recoverability",
    ],
    "Security": [
        "Confidentiality",
        "Integrity",
        "Non-repudiation",
        "Accountability",
        "Authenticity",
        "Resistance",
    ],
    "Maintainability": [
        "Modularity",
        "Reusability",
        "Analysability",
        "Modifiability",
        "Testability",
    ],
    "Flexibility": [
        "Adaptability",
        "Scalability",
        "Installability",
        "Replaceability",
    ],
    "Safety": [
        "Operational constraint",
        "Risk identification",
        "Fail safe",
        "Hazard warning",
        "Safe integration",
    ],
}

# Flat set of all valid sub-characteristic names
ALL_SUB_CHARS: set[str] = {sub for subs in ISO_TAXONOMY.values() for sub in subs}

# Reverse lookup: sub-characteristic → parent characteristic
SUB_TO_CHAR: dict[str, str] = {
    sub: char
    for char, subs in ISO_TAXONOMY.items()
    for sub in subs
}

# Pragmatic best-fit mapping: NICE label → expected ISO characteristics (at characteristic level)
# Used only for scoring — never shown to the labelling agent.
# L and OT have no principled mapping and are omitted.
NICE_TO_ISO_CHARS: dict[str, list[str]] = {
    "Availability (A)":     ["Reliability"],
    "Fault Tolerance (FT)": ["Reliability", "Safety"],
    "Look & Feel (LF)":     ["Interaction capability"],
    "Maintainability (MN)": ["Maintainability"],
    "Operability (O)":      ["Interaction capability"],
    "Performance (PE)":     ["Performance efficiency"],
    "Portability (PO)":     ["Flexibility", "Compatibility"],
    "Scalability (SC)":     ["Flexibility"],
    "Security (SE)":        ["Security"],
    "Usability (US)":       ["Interaction capability"],
}

# NICE CSV column names (binary 0/1) — in CSV order
NICE_LABEL_COLUMNS: list[str] = [
    "Availability (A)",
    "Fault Tolerance (FT)",
    "Legal (L)",
    "Look & Feel (LF)",
    "Maintainability (MN)",
    "Operability (O)",
    "Performance (PE)",
    "Portability (PO)",
    "Scalability (SC)",
    "Security (SE)",
    "Usability (US)",
    "Other (OT)",
]


def score(predicted_sub_chars: list[str], nice_labels: list[str]) -> dict:
    """Compute Jaccard and Recall at the ISO characteristic level.

    Returns a dict with keys: scoreable, jaccard, recall, predicted_chars,
    expected_chars, complete_miss.  When scoreable=False, only that key is set.
    """
    predicted_chars = {SUB_TO_CHAR[s] for s in predicted_sub_chars if s in SUB_TO_CHAR}
    expected_chars = {
        char
        for label in nice_labels
        for char in NICE_TO_ISO_CHARS.get(label, [])
    }
    if not expected_chars:
        return {"scoreable": False}

    intersection = predicted_chars & expected_chars
    union = predicted_chars | expected_chars

    jaccard = len(intersection) / len(union) if union else 0.0
    recall = len(intersection) / len(expected_chars)

    return {
        "scoreable": True,
        "jaccard": jaccard,
        "recall": recall,
        "predicted_chars": sorted(predicted_chars),
        "expected_chars": sorted(expected_chars),
        "complete_miss": jaccard == 0.0,
    }
