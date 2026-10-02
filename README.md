# Resume Review Assistant

An explainable, human-in-the-loop Streamlit application that helps recruiters
prioritize resumes against explicitly configured job criteria. It combines
semantic similarity with required skills, preferred skills, and experience while
showing the evidence and score breakdown behind every result.

> This application is a decision-support prototype. It does not make hiring
> decisions and should never be used as the sole basis for an employment action.

## Features

- Lazy, cached Sentence Transformer loading with visible progress
- Weighted and fully visible score components
- Required and preferred skill criteria with alias-aware matching
- Resume evidence snippets for matched skills
- PDF and DOCX support with signature, size, and document limits
- Detection of scanned or text-empty PDFs
- Per-file error isolation for batch processing
- Blind-review mode for common direct identifiers
- Candidate filtering and side-by-side comparison
- CSV, Excel, and JSON audit exports
- Optional Gemini-generated candidate summaries and personalized resume suggestions
- Optional Gemini-generated interview questions and requirement evidence explanations
- Explicit methodology, safeguards, and limitations
- Automated unit and Streamlit rendering tests

## Scoring

The default score is composed of:

| Component | Default weight | Meaning |
|---|---:|---|
| Semantic similarity | 40% | Overall relationship between the job description and resume |
| Required skills | 35% | Percentage of configured required skills found |
| Preferred skills | 10% | Percentage of configured preferred skills found |
| Experience | 15% | Explicitly stated years relative to the configured minimum |

Weights are configurable and normalized automatically. If preferred skills or a
minimum-experience requirement are not configured, their weights are removed and
the remaining components are renormalized. Skill mentions demonstrate textual
evidence, not proficiency; a human reviewer must inspect the supporting context.

## Local installation

Python 3.10–3.14 is recommended.

```bash
cd /path/to/ai-resume-ranker
python3 -m venv ../ai-resume-ranker-venv
source ../ai-resume-ranker-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`. The language model is downloaded when the first
analysis starts and cached for later use; the initial download requires internet
access.

## Optional Gemini enrichment

The app works without Gemini and retains its deterministic scoring and rule-based
summary behavior. To enable the four optional GenAI features, create a Gemini API
key in [Google AI Studio](https://aistudio.google.com/apikey), copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, and replace the
placeholder value. Alternatively, set the `GEMINI_API_KEY` environment variable.

The default model is `gemini-3.1-flash-lite`. Override it with the `GEMINI_MODEL`
environment variable if needed. Model availability, free-tier limits, and pricing
can change; check Google's current Gemini API documentation before a demonstration.

One Gemini request per candidate produces a contextual summary, personalized
improvement suggestions, three interview questions, and an evidence explanation
for every required skill. Results are cached in process using a hash of the
anonymized input and job configuration, so Streamlit reruns do not repeat calls.
Gemini never changes deterministic scores, ranks, experience values, or review
priorities. If the key is missing, the API is unavailable, or its response is
invalid, the existing rule-based summary and suggestions remain available.

Only anonymized text is sent when blind-review mode is enabled. Inputs longer than
30,000 resume characters or 12,000 job-description characters are bounded for the
request by keeping the beginning and end. Free-tier data may be used by Google to
improve its products. Use fictional or appropriately anonymized resumes for demos,
and verify all generated content against the original source.

## Testing

The core suite uses Python's built-in test runner:

```bash
python -m unittest discover -s tests -v
```

For a development environment with pytest:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Privacy and security behavior

- Files are processed in memory and are not persisted by this application.
- Valid Gemini enrichments are cached only in application memory; raw resume text is not written to disk.
- Supported uploads are PDF and DOCX, limited to 10 MB.
- PDF signatures, page counts, DOCX structure, and expanded DOCX size are checked.
- Blind-review mode redacts common emails, phone numbers, URLs, and likely header names.
- Blind review cannot guarantee removal of every identity or demographic proxy.
- A production deployment should add authentication, encrypted persistence only if
  necessary, retention controls, malware scanning, access logs, and an incident process.

## Known limitations

- Scanned PDFs are detected but OCR is not bundled because reliable OCR requires an
  additional system runtime or managed service.
- Experience extraction intentionally uses explicit phrases such as “5 years of
  experience”; it does not infer duration from employment dates.
- The included skill taxonomy is deterministic and extensible, but not exhaustive.
- Semantic embedding models can reproduce biases from their training data.
- The app has not been validated for autonomous or high-stakes employment decisions.
- Gemini output can be incomplete or incorrect and always requires human verification.

## Project structure

```text
app.py                       Streamlit workflow and presentation
utils/matcher.py             Lazy model loading and semantic similarity
utils/scoring.py             Weighted, explainable scoring
utils/skills_extractor.py    Skill aliases, matching, and evidence
utils/pdf_parser.py          Secure PDF/DOCX validation and parsing
utils/privacy.py             Blind-review redaction
utils/candidate_analyzer.py  Experience and review-quality heuristics
utils/genai_analyzer.py      Optional Gemini prompting, validation, and memory cache
tests/                       Core and application smoke tests
```
