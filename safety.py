"""
safety.py
---------
Safety layer for the Healthcare Staff Knowledge Assistant.

Before ANY question is sent to the retrieval + Gemini pipeline, it passes
through this file. If the question looks clinical, medical, emergency
related, or asks for a decision staff are not allowed to make, the app
immediately returns an escalation message and NEVER calls Gemini.

This is deliberately simple keyword/phrase matching so it is easy to
read, easy to explain in a presentation, and easy to extend.
"""

import re

# ---------------------------------------------------------------------------
# Risky keywords and phrases, grouped by the reason they are risky.
# Grouping lets get_risk_reason() explain WHY a question was blocked.
# All matching is done in lowercase.
# ---------------------------------------------------------------------------
RISKY_CATEGORIES = {
    "medical diagnosis": [
        "diagnosis",
        "diagnose",
        "what illness",
        "what disease",
        "what condition does",
    ],
    "treatment advice": [
        "treatment",
        "treat ",          # trailing space avoids matching words like "treaty"
        "how to cure",
        "therapy for",
    ],
    "medication or dosage advice": [
        "medication",
        "medicine",
        "dosage",
        "dose",
        "prescription",
        "prescribe",
        "what drug",
        "should the patient take",
        "which pill",
        "painkiller",
    ],
    "emergency / clinical symptoms": [
        "symptoms",
        "emergency",
        "chest pain",
        "bleeding",
        "unconscious",
        "allergic reaction",
        "not breathing",
        "seizure",
        "stroke",
        "heart attack",
        "overdose",
    ],
    "patient-specific clinical decision": [
        "patient-specific advice",
        "medical advice",
        "clinical decision",
        "clinical advice",
        "is it safe for the patient",
    ],
    "insurance approval decision": [
        "insurance approval",
        "approve insurance",
        "approve the claim",
        "approve coverage",
        "deny the claim",
    ],
    "legal decision": [
        "legal advice",
        "legal decision",
        "lawsuit",
        "sue ",
    ],
}


def _normalize(question: str) -> str:
    """Lowercase the question and collapse extra whitespace.

    A trailing space is added so patterns that end with a space
    (like "treat ") can still match at the very end of the question.
    """
    text = question.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text + " "


def is_risky_question(question: str) -> bool:
    """Return True if the question touches a restricted topic.

    Restricted topics: diagnosis, treatment, medication, dosage,
    emergencies, patient-specific clinical decisions, insurance
    approvals, and legal decisions.
    """
    text = _normalize(question)
    for phrases in RISKY_CATEGORIES.values():
        for phrase in phrases:
            if phrase in text:
                return True
    return False


def get_risk_reason(question: str) -> str:
    """Return a short human-readable reason why the question is risky.

    Returns an empty string if the question is NOT risky.
    """
    text = _normalize(question)
    for reason, phrases in RISKY_CATEGORIES.items():
        for phrase in phrases:
            if phrase in text:
                return reason
    return ""


def escalation_response(reason: str = "") -> str:
    """Build the standard escalation message shown to the user.

    This is the ONLY hardcoded response in the project, and it exists
    on purpose: for safety-critical topics we never want the AI to
    improvise.
    """
    message = (
        "I'm not authorized to answer this. "
        "Please contact your supervisor or the relevant department."
    )
    if reason:
        message += f"\n\n*Reason: this question appears to involve {reason}.*"
    return message
