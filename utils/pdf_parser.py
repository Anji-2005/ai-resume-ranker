"""Secure, bounded resume file validation and text extraction."""

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pymupdf


MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_DOCX_UNCOMPRESSED_SIZE = 25 * 1024 * 1024


class ResumeParseError(ValueError):
    pass


def validate_resume_file(filename, data):
    """Validate type, signature, and size without trusting browser metadata."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ResumeParseError("Only PDF and DOCX files are supported.")
    if not data:
        raise ResumeParseError("The uploaded file is empty.")
    if len(data) > MAX_FILE_SIZE:
        raise ResumeParseError("The file exceeds the 10 MB size limit.")
    if suffix == ".pdf" and not data.startswith(b"%PDF-"):
        raise ResumeParseError("The file extension is PDF, but its signature is invalid.")
    if suffix == ".docx" and not data.startswith(b"PK"):
        raise ResumeParseError("The file extension is DOCX, but its signature is invalid.")
    return suffix


def extract_text_from_bytes(filename, data):
    suffix = validate_resume_file(filename, data)
    return _extract_pdf(data) if suffix == ".pdf" else _extract_docx(data)


def _extract_pdf(data):
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ResumeParseError("Password-protected PDFs are not supported.")
            if doc.page_count > MAX_PDF_PAGES:
                raise ResumeParseError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
            return "\n".join(page.get_text("text") for page in doc).strip()
    except ResumeParseError:
        raise
    except Exception as exc:
        raise ResumeParseError("The PDF is corrupted or could not be read.") from exc


def _extract_docx(data):
    try:
        from docx import Document
    except ImportError as exc:
        raise ResumeParseError("DOCX support requires the python-docx package.") from exc
    try:
        with ZipFile(BytesIO(data)) as archive:
            if "word/document.xml" not in archive.namelist():
                raise ResumeParseError("The uploaded file is not a valid DOCX document.")
            if sum(item.file_size for item in archive.infolist()) > MAX_DOCX_UNCOMPRESSED_SIZE:
                raise ResumeParseError("The DOCX expands beyond the safe processing limit.")
        document = Document(BytesIO(data))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            paragraphs.extend(cell.text for row in table.rows for cell in row.cells)
        return "\n".join(paragraphs).strip()
    except ResumeParseError:
        raise
    except (BadZipFile, OSError, ValueError, KeyError) as exc:
        raise ResumeParseError("The DOCX file is corrupted or could not be read.") from exc


def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF file.
    """
    return extract_text_from_bytes(getattr(pdf_file, "name", "resume.pdf"), pdf_file.read())
