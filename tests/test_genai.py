import copy
import json
import unittest

from utils.genai_analyzer import (
    GenAIResponseError,
    clear_enrichment_cache,
    enrich_candidate,
    validate_enrichment,
)


RESUME_TEXT = (
    "Built Python automation services for reporting. "
    "Created REST APIs and improved response time by 30%."
)
JOB = {
    "description": "Build Python services and deploy them to AWS.",
    "required_skills": ["Python", "AWS"],
    "preferred_skills": ["REST APIs"],
}
ANALYSIS = {
    "matched_required": ["Python"],
    "missing_required": ["AWS"],
    "matched_preferred": ["REST APIs"],
    "missing_preferred": [],
    "years": 0,
    "experience_level": "Entry-level",
    "components": {"semantic": 70.0, "required_skills": 50.0},
    "score": 61.0,
    "recommendation": "Potential match",
}


def valid_payload():
    return {
        "candidate_summary": "The resume shows Python service and REST API experience.",
        "improvement_suggestions": [
            "If accurate, add evidence of cloud deployment experience."
        ],
        "interview_questions": [
            {"question": "How did you design the Python services?", "reason": "Verify Python depth."},
            {"question": "How did you measure API performance?", "reason": "Verify the stated impact."},
            {"question": "What AWS deployment work have you done?", "reason": "Clarify a missing requirement."},
        ],
        "requirement_evidence": [
            {
                "requirement": "Python", "status": "supported",
                "evidence": "Built Python automation services for reporting.",
                "explanation": "The resume directly describes Python service work.",
            },
            {
                "requirement": "AWS", "status": "unsupported", "evidence": "",
                "explanation": "No AWS evidence is stated.",
            },
        ],
    }


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return FakeResponse(json.dumps(self.payload))


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.models = FakeModels(payload, error)


class GenAIValidationTests(unittest.TestCase):
    def setUp(self):
        clear_enrichment_cache()

    def test_valid_structured_output_and_exactly_three_questions(self):
        result = validate_enrichment(valid_payload(), RESUME_TEXT, JOB["required_skills"])
        self.assertEqual(len(result["interview_questions"]), 3)
        self.assertEqual(result["requirement_evidence"][0]["status"], "supported")

    def test_incomplete_payload_is_rejected(self):
        payload = valid_payload()
        del payload["candidate_summary"]
        with self.assertRaises(GenAIResponseError):
            validate_enrichment(payload, RESUME_TEXT, JOB["required_skills"])

    def test_invalid_question_count_is_rejected(self):
        payload = valid_payload()
        payload["interview_questions"].pop()
        with self.assertRaises(GenAIResponseError):
            validate_enrichment(payload, RESUME_TEXT, JOB["required_skills"])

    def test_sensitive_interview_question_is_rejected(self):
        payload = valid_payload()
        payload["interview_questions"][0]["question"] = "What is your marital status?"
        with self.assertRaises(GenAIResponseError):
            validate_enrichment(payload, RESUME_TEXT, JOB["required_skills"])

    def test_invalid_evidence_status_is_rejected(self):
        payload = valid_payload()
        payload["requirement_evidence"][0]["status"] = "probably"
        with self.assertRaises(GenAIResponseError):
            validate_enrichment(payload, RESUME_TEXT, JOB["required_skills"])

    def test_fabricated_evidence_is_not_displayed_as_a_quote(self):
        payload = valid_payload()
        payload["requirement_evidence"][0]["evidence"] = "Led a Python team of 20."
        result = validate_enrichment(payload, RESUME_TEXT, JOB["required_skills"])
        python_evidence = result["requirement_evidence"][0]
        self.assertEqual(python_evidence["evidence"], "")
        self.assertEqual(python_evidence["status"], "unclear")

    def test_unknown_or_missing_requirement_is_rejected(self):
        payload = valid_payload()
        payload["requirement_evidence"][1]["requirement"] = "Docker"
        with self.assertRaises(GenAIResponseError):
            validate_enrichment(payload, RESUME_TEXT, JOB["required_skills"])


class GenAIClientTests(unittest.TestCase):
    def setUp(self):
        clear_enrichment_cache()

    def test_missing_api_key_does_not_call_provider(self):
        client = FakeClient(valid_payload())
        with self.assertRaises(RuntimeError):
            enrich_candidate(RESUME_TEXT, JOB, ANALYSIS, "", client=client)
        self.assertEqual(client.models.calls, 0)

    def test_invalid_json_is_rejected(self):
        client = FakeClient(valid_payload())
        client.models.generate_content = lambda **kwargs: FakeResponse("not-json")
        with self.assertRaises(json.JSONDecodeError):
            enrich_candidate(RESUME_TEXT, JOB, ANALYSIS, "test-key", client=client)

    def test_timeout_is_not_retried(self):
        client = FakeClient(error=TimeoutError("timed out"))
        with self.assertRaises(TimeoutError):
            enrich_candidate(RESUME_TEXT, JOB, ANALYSIS, "test-key", client=client)
        self.assertEqual(client.models.calls, 1)

    def test_rate_limit_is_not_retried(self):
        client = FakeClient(error=RuntimeError("429 rate limit"))
        with self.assertRaises(RuntimeError):
            enrich_candidate(RESUME_TEXT, JOB, ANALYSIS, "test-key", client=client)
        self.assertEqual(client.models.calls, 1)

    def test_success_is_cached_and_deterministic_analysis_is_unchanged(self):
        client = FakeClient(valid_payload())
        original_analysis = copy.deepcopy(ANALYSIS)
        first = enrich_candidate(
            RESUME_TEXT, JOB, ANALYSIS, "test-key", model_name="test-model", client=client
        )
        second = enrich_candidate(
            RESUME_TEXT, JOB, ANALYSIS, "test-key", model_name="test-model", client=client
        )
        self.assertEqual(first, second)
        self.assertEqual(client.models.calls, 1)
        self.assertEqual(ANALYSIS, original_analysis)
        self.assertNotIn("score", first)
        self.assertNotIn("recommendation", first)


if __name__ == "__main__":
    unittest.main()
