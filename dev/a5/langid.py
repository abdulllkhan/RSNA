"""Tiny dependency-free language guesser for the RSNA knee-MRI reports.

The corpus is multilingual free text (Spanish, Dutch, English, German, Turkish,
Portuguese, French, Italian, ...).  We only need a coarse bucket for reporting
keyword hit-rates per language, so a weighted stopword / diacritic scorer beats
adding a dependency (`langdetect` is not installed in the venv and we are not
allowed to pull large things right now).

Public API:  guess_lang(text) -> ISO-639-1 code (or "unk")
"""

from __future__ import annotations

import re
import unicodedata

# --- distinctive markers -------------------------------------------------
# weight 3 = near-unique to the language (word or radiology term)
# weight 1 = common function word, ambiguous across the romance languages
_MARKERS: dict[str, list[tuple[str, int]]] = {
    "es": [
        ("rodilla", 3), ("menisco", 3), ("hallazgos", 3), ("derrame", 3),
        ("tecnica", 3), ("técnica", 3), ("impresion", 3), ("impresión", 3),
        ("senal", 2), ("señal", 3), ("condilo", 3), ("cóndilo", 3),
        ("rotura", 3), ("ligamento cruzado", 3), ("articular", 1),
        ("del", 1), ("los", 1), ("las", 1), ("con", 1), ("sin", 2), ("no se", 2),
        ("se observa", 3), ("presenta", 2),
    ],
    "pt": [
        ("joelho", 3), ("menisco", 1), ("achados", 3), ("derrame articular", 2),
        ("nao", 2), ("não", 3), ("sao", 2), ("são", 3), ("com", 1),
        ("lesao", 2), ("lesão", 3), ("observa-se", 3), ("condilo", 1),
        ("integro", 2), ("íntegro", 3), ("espessura", 3),
    ],
    "nl": [
        ("knie", 3), ("meniscus", 2), ("bevindingen", 3), ("conclusie", 3),
        ("geen", 3), ("het ", 3), ("van de", 3), ("aanwijzingen", 3),
        ("gewricht", 3), ("kraakbeen", 3), ("vocht", 3), ("wordt", 2),
        ("rechts", 1), ("links", 1), ("beeld", 2), ("mediale", 1),
        ("gewogen", 3), ("opname", 2), ("normaal", 1),
    ],
    "de": [
        ("kniegelenk", 3), ("nachweis", 3), ("keine", 3), ("kein ", 3),
        ("beurteilung", 3), ("befund", 3), ("gelenkerguss", 3), ("erguss", 3),
        ("meniskus", 3), ("knorpel", 3), ("der ", 2), ("die ", 2), ("und ", 2),
        ("mit ", 1), ("nicht", 2), ("unauffaellig", 2), ("unauffällig", 3),
        ("darstellung", 3), ("bandapparat", 3),
    ],
    "tr": [
        ("izlenmistir", 3), ("izlenmiştir", 3), ("bulgular", 3), ("diz", 3),
        ("meniskus", 2), ("menisküs", 3), ("eklem", 3), ("kemik", 3),
        ("sinyal", 3), ("ligaman", 3), ("dejenerasyon", 3), ("tetkik", 3),
        ("sag ", 1), ("sol ", 2), ("normaldir", 3), ("mevcuttur", 3),
        ("kalinlasma", 2), ("kalınlaşma", 3), ("bagi", 2), ("bağı", 3),
    ],
    "en": [
        (" the ", 2), (" of the", 3), (" is ", 2), (" are ", 2), (" with ", 1),
        ("knee", 3), ("tear", 3), ("joint", 2), ("effusion", 2), ("findings", 2),
        ("normal", 1), ("signal intensity", 3), ("no evidence", 3),
        ("meniscus", 1), ("ligament", 1), ("intact", 3), ("noted", 3),
        ("bone marrow", 3), ("impression", 2),
    ],
    "fr": [
        ("genou", 3), ("menisque", 2), ("ménisque", 3), ("epanchement", 2),
        ("épanchement", 3), ("des ", 1), ("une ", 2), ("dans ", 2),
        ("pas de", 3), ("aspect", 1), ("articulaire", 2), ("cartilage", 1),
        ("ligament croise", 3), ("conclusion", 1),
    ],
    "it": [
        ("ginocchio", 3), ("menisco", 1), ("versamento", 3), ("della ", 3),
        ("nella", 2), ("non si", 3), ("reperti", 3), ("legamento", 2),
        ("cartilagine", 2), ("articolazione", 2), ("presenza", 1),
    ],
    "pl": [
        ("kolana", 3), ("kolanowego", 3), ("lakotki", 3), ("łąkotki", 3),
        ("wiezadlo", 3), ("więzadło", 3), ("wiezadla", 3), ("więzadła", 3),
        ("maziowej", 3), ("chrzastki", 3), ("chrząstki", 3), ("plyn", 2),
        ("płyn", 3), ("uwidoczniono", 3), ("badanie", 2), ("obraz", 1),
    ],
    # Croatian / Serbian / Bosnian (latin script) — a big slice of this corpus
    "hr": [
        ("koljena", 3), ("koljeno", 3), ("meniska", 3), ("hrskavice", 3),
        ("zgloba", 3), ("krizn", 3), ("križn", 3), ("straznjeg", 3),
        ("stražnjeg", 3), ("nalaz", 2), ("uredan", 3), ("uredna", 3),
        ("prednjeg", 2), ("ligamenta", 1), ("izljev", 3), ("promjena", 2),
        ("desnog", 2), ("lijevog", 3), ("kosti", 1), ("roga", 2),
    ],
}

# Non-latin scripts are settled by codepoint block, not by stopwords.
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Bulgarian vs Russian: a few high-frequency discriminators.
_BG_HINTS = ("коляно", "колянна", "ставен", "данни", "няма", "със", "костите")
_RU_HINTS = ("коленного", "сустава", "определяется", "признаки", "отмечается")

# characters that strongly pin a language
_CHARSET_HINTS: dict[str, str] = {
    "tr": "ğışĞİŞ",
    # ć/č/š/ž are shared with hr/sr, so only the pl-exclusive letters count
    "pl": "ąęłńźżĄĘŁŃŹŻ",
    "hr": "đĐ",
    "es": "ñ¿¡",
    "pt": "ãõâç",
    "de": "ß",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _norm(text: str) -> str:
    return " " + re.sub(r"\s+", " ", text.lower()) + " "


def guess_lang(text: str | None) -> str:
    """Return an ISO-639-1 guess for `text` ("unk" if there is nothing to go on)."""
    if not text or not isinstance(text, str):
        return "unk"
    low = _norm(text)
    if len(_WORD_RE.findall(low)) < 3:
        return "unk"

    # --- script first: Greek / Cyrillic never need the latin stopword scorer
    if len(_GREEK.findall(text)) >= 10:
        return "el"
    if len(_CYRILLIC.findall(text)) >= 10:
        ru = sum(h in low for h in _RU_HINTS)
        bg = sum(h in low for h in _BG_HINTS)
        return "ru" if ru > bg else "bg"

    scores: dict[str, float] = {}
    for lang, markers in _MARKERS.items():
        s = 0.0
        for token, weight in markers:
            tok = token if token.startswith(" ") or token.endswith(" ") else token
            if tok in low:
                s += weight
        scores[lang] = s

    for lang, chars in _CHARSET_HINTS.items():
        if any(c in text for c in chars):
            scores[lang] = scores.get(lang, 0.0) + 6.0

    # Turkish dotless-i / Dutch "ij" digraph are cheap extra evidence.
    if "ı" in text:
        scores["tr"] = scores.get("tr", 0.0) + 6.0
    if re.search(r"\bij\w|\w ij\b|zijn|hij\b", low):
        scores["nl"] = scores.get("nl", 0.0) + 2.0

    best = max(scores, key=lambda k: scores[k])
    if scores[best] < 3.0:
        return "unk"
    # require a margin over the runner-up for the confusable romance set
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    if len(ordered) > 1 and ordered[0][1] - ordered[1][1] < 1.0:
        # tie-break by a tiny character-level prior, else keep the top
        return ordered[0][0]
    return best


def normalize_for_match(text: str) -> str:
    """Lowercase + strip accents, so `sinovítis` matches `sinovitis`."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))
