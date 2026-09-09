"""A5 step 2 — the focused single-finding Synovitis prompt.

Design constraints
------------------
* ONE finding only (Synovitis).  Asking for all 12 is what the public label sets
  already do; the whole point of A5 is a deep, undistracted read of one column.
* SHORT.  4,407 reports x (prompt + report).  Reports are ~350 tok (mean 1,098
  chars); the instruction block below is ~430 tok, so ~800 tok in / ~60 tok out.
* Multilingual.  The corpus is en 39% / es 16% / tr 12% / hr 9% / el 7% / de 6% /
  bg 5% / nl 3% / fr 2% (see `extract_reports.py` output), so the YES/NO
  vocabulary is given in those languages rather than assumed.
* STRICT JSON, one line, no prose, no markdown fence, no chain-of-thought.

Public API
----------
    SYSTEM_PROMPT                 -> str
    build_messages(report)        -> list[dict]   (chat/completions `messages`)
    RETRY_NUDGE                   -> str          (appended on a parse failure)
    approx_prompt_tokens(report)  -> int          (rough budgeting helper)
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You read ONE knee-MRI radiology report (any language) and judge ONE question only:
does the report ASSERT SYNOVITIS (inflammation / thickening / proliferation of the
synovial membrane)?

COUNT AS YES
- The word itself in any language: synovitis, sinovitis, synovite, Synovialitis,
  Synovitis, sinovit, sinovyal, synovitis/synoviitis, sinovitis, синовит,
  υμενίτιδα, zapalenie błony maziowej, sinovitis.
- Synovial thickening / proliferation / hypertrophy / enhancement / inflamed or
  hyperplastic synovium; pannus; villonodular synovial disease (PVNS).
  e.g. "engrosamiento sinovial", "proliferación sinovial", "synoviale verdikking",
  "Synovialisverdickung", "sinovyal kalınlaşma", "épaississement synovial",
  "zadebljanje sinovije", "удебелена синовия", "πάχυνση του αρθρικού υμένα".
- Effusion explicitly described together with synovial thickening/inflammation
  ("joint effusion with synovitis", "derrame con sinovitis").

COUNT AS NO — these are the traps; the word "synovial" appearing is NOT enough
- Joint effusion / fluid ALONE, however large: effusion, derrame, Erguss, vocht,
  epanchement, efüzyon, versamento, izljev, излив, συλλογή υγρού, płyn/wysięk.
  Fluid is NOT synovitis. "Synovial fluid" / "Synovia" used to mean the fluid is NOT
  synovitis either.
- SYNOVIAL PLICA (medial patellar plica, plica synovialis, sinovyal plika,
  plica sinovial) — an anatomical fold. Even "thickened plica" is NOT synovitis.
- Synovial cyst, ganglion cyst, Baker's/popliteal cyst, bursitis, tendinitis,
  fat-pad (Hoffa) oedema, synovial osteochondromatosis — none of these alone.
- Osteoarthritis / degenerative change / meniscal tear alone.
- Explicit negation or normal synovium: "no synovitis", "sin signos de sinovitis",
  "geen synovitis", "kein Nachweis einer Synovialitis", "Synovia unauffällig",
  "sinovit izlenmemistir", "normal synovium", "uredna sinovija".
- The report never mentions the synovium at all (SILENT) -> low probability.
Judge each mention inside ITS OWN sentence: "derrame articular con sinovitis difusa.
Quiste poplíteo..." is synovitis (the cyst is a separate sentence), while "synovial
cyst" in one phrase is not.

OUTPUT
Reply with STRICT JSON on ONE line and nothing else - no markdown, no explanation:
{"synovitis": <number 0..1>, "evidence": "<verbatim span copied from the report, or empty>", "language": "<ISO 639-1>"}

CALIBRATION for "synovitis"
  0.95  the finding is named or unambiguously described
  0.80  clearly implied (synovial thickening/enhancement without the word itself)
  0.55  hedged or equivocal ("possible synovitis", "mild synovial irregularity")
  0.30  only indirect hints (large effusion + marked inflammatory context)
  0.10  the report is SILENT on the synovium
  0.02  the report explicitly denies synovitis
"evidence" must be COPIED verbatim from the report (max ~15 words); empty string
if the report is silent.

EXAMPLES
report: "Tecnica: RMN de la rodilla. Resultados: Meniscopatia interna. Condromalacia rotuliana. Sinovitis. Bursitis prepatelar."
{"synovitis": 0.96, "evidence": "Sinovitis", "language": "es"}

report: "Moderate joint effusion is present. Medial meniscus posterior horn tear. Intact cruciate ligaments. No fracture."
{"synovitis": 0.08, "evidence": "", "language": "en"}

report: "Maessiger Gelenkerguss mit deutlicher Synovialisverdickung im Recessus suprapatellaris. Kein Nachweis einer Meniskuslaesion."
{"synovitis": 0.88, "evidence": "deutlicher Synovialisverdickung", "language": "de"}

report: "Eklem mesafesinde efuzyon izlenmistir. Sinovyal kalinlasma izlenmemistir. Medial meniskus arka boynuzunda grade II dejenerasyon."
{"synovitis": 0.03, "evidence": "Sinovyal kalinlasma izlenmemistir", "language": "tr"}

report: "Er is fors gewrichtsvocht met uitgesproken synoviale proliferatie in de suprapatellaire recessus. Meniscus intact."
{"synovitis": 0.92, "evidence": "uitgesproken synoviale proliferatie", "language": "nl"}

report: "MP данни за ставен излив. Хрущялът на фемура е изтънен медиално. Няма МР данни за руптура на предната кръстна връзка."
{"synovitis": 0.09, "evidence": "", "language": "bg"}

report: "Diz eklemi ici sivi miktari hafif derecede artmis. Medyal patellar sinovyal plika kalinligi normaldir. On capraz bag normal."
{"synovitis": 0.06, "evidence": "", "language": "tr"}
"""

USER_TEMPLATE = 'report: "{report}"'

RETRY_NUDGE = (
    "Your previous reply was not valid JSON. Reply with ONE line of strict JSON only: "
    '{"synovitis": <number 0..1>, "evidence": "<verbatim span>", "language": "<code>"}'
)

# Reports run to 4,743 chars; the launch line uses a 4k context, so cap the body
# well inside it (chars, not tokens — a safe 3.5 chars/token assumption).
MAX_REPORT_CHARS = 6000


def _clean(report: str) -> str:
    if not report:
        return ""
    r = " ".join(str(report).split())
    # keep the quoting in USER_TEMPLATE from being broken by the report itself
    r = r.replace('"', "'")
    if len(r) > MAX_REPORT_CHARS:
        r = r[:MAX_REPORT_CHARS] + " ...[truncated]"
    return r


def build_messages(report: str, use_system: bool = True) -> list[dict]:
    """Chat `messages` for one report.

    `use_system=False` folds the instructions into the single user turn — needed
    for chat templates that reject a `system` role (some Gemma-3 GGUF templates
    do).  Keeping the instruction block first either way preserves the shared
    prefix that llama-server's `--cache-reuse` reuses across requests.
    """
    user = USER_TEMPLATE.format(report=_clean(report))
    if use_system:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
    return [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}]


def build_retry_messages(report: str, bad_reply: str, use_system: bool = True) -> list[dict]:
    msgs = build_messages(report, use_system)
    msgs.append({"role": "assistant", "content": (bad_reply or "")[:400]})
    msgs.append({"role": "user", "content": RETRY_NUDGE})
    return msgs


def approx_prompt_tokens(report: str) -> int:
    """Very rough token estimate (3.5 chars/token) for runtime budgeting."""
    return int((len(SYSTEM_PROMPT) + len(_clean(report)) + 40) / 3.5)


if __name__ == "__main__":  # quick budget check
    import sys
    from pathlib import Path

    import pandas as pd

    p = Path(__file__).resolve().parents[2] / "data" / "a5" / "reports.parquet"
    if not p.exists():
        sys.exit("run dev/a5/extract_reports.py first")
    d = pd.read_parquet(p)
    est = d["Report"].map(approx_prompt_tokens)
    print(f"system prompt: {len(SYSTEM_PROMPT)} chars ~ {len(SYSTEM_PROMPT)/3.5:.0f} tok")
    print(
        f"per-request prompt tokens: mean {est.mean():.0f}  median {est.median():.0f}  "
        f"p90 {est.quantile(0.9):.0f}  max {est.max():.0f}"
    )
    print(f"total prompt tokens for {len(d)} reports: {est.sum()/1e6:.2f} M")
