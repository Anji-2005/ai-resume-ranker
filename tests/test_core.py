from io import BytesIO
import unittest

import pymupdf
import pandas as pd
from docx import Document

from utils.candidate_analyzer import estimate_years_experience
from utils.pdf_parser import ResumeParseError, extract_text_from_bytes, validate_resume_file
from utils.privacy import anonymize_resume_text, safe_candidate_label
from utils.scoring import calculate_score_breakdown, normalize_weights, review_recommendation
from utils.skills_extractor import extract_skills, find_skill_evidence, parse_skill_list


class SkillExtractionTests(unittest.TestCase):
    def test_aliases_are_canonicalized(self):
        self.assertEqual(parse_skill_list("python; powerbi, sklearn"),
                         ["Python", "Power BI", "scikit-learn"])

    def test_word_boundaries_avoid_nosql_false_positive(self):
        self.assertNotIn("SQL", extract_skills("Built a NoSQL document store"))

    def test_evidence_is_returned_from_source_text(self):
        evidence = find_skill_evidence("Built APIs with Python. Used SQL daily.", ["Python", "SQL"])
        self.assertIn("Python", evidence["Python"][0])
        self.assertIn("SQL", evidence["SQL"][0])


class ScoringTests(unittest.TestCase):
    def test_weights_are_normalized(self):
        weights = normalize_weights({"semantic": 40, "required_skills": 40,
                                     "preferred_skills": 10, "experience": 10})
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_unconfigured_optional_criteria_have_zero_weight(self):
        result = calculate_score_breakdown(
            50, "Python developer", ["Python", "SQL"], [], 0, 0,
            {"semantic": 40, "required_skills": 35, "preferred_skills": 10, "experience": 15},
        )
        self.assertEqual(result["weights"]["preferred_skills"], 0)
        self.assertEqual(result["weights"]["experience"], 0)
        self.assertAlmostEqual(result["final_score"], 50.0)

    def test_review_label_is_not_hiring_decision(self):
        self.assertEqual(review_recommendation(85, 80), "Priority review")
        self.assertNotIn(review_recommendation(40, 20), {"Hire", "Reject"})


class ParsingAndPrivacyTests(unittest.TestCase):
    def test_pdf_signature_is_checked(self):
        with self.assertRaises(ResumeParseError):
            validate_resume_file("resume.pdf", b"not a PDF")

    def test_text_pdf_is_extracted(self):
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Python developer with five years of experience")
        data = document.tobytes()
        document.close()
        self.assertIn("Python developer", extract_text_from_bytes("resume.pdf", data))

    def test_docx_is_extracted(self):
        output = BytesIO()
        document = Document()
        document.add_paragraph("SQL analyst with 3 years of experience")
        document.save(output)
        self.assertIn("SQL analyst", extract_text_from_bytes("resume.docx", output.getvalue()))

    def test_excel_engine_is_available(self):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame([{"candidate": "Candidate 01", "score": 80}]).to_excel(
                writer, index=False
            )
        self.assertTrue(output.getvalue().startswith(b"PK"))

    def test_direct_identifiers_are_redacted(self):
        redacted = anonymize_resume_text("Jane Doe\njane@example.com\n+91 98765 43210")
        self.assertNotIn("Jane Doe", redacted)
        self.assertNotIn("jane@example.com", redacted)
        self.assertNotIn("98765", redacted)
        self.assertEqual(safe_candidate_label("jane.pdf", 2, True), "Candidate 02")

    def test_experience_requires_explicit_claim(self):
        self.assertEqual(estimate_years_experience("5 years of experience in analytics"), 5)
        self.assertEqual(estimate_years_experience("Graduated in 2020"), 0)


if __name__ == "__main__":
    unittest.main()
