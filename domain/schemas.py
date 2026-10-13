"""Pydantic contracts for provenance-bearing recruitment documents.

Offsets use Python string offsets into ``ParsedDocument.text``.  ``page_number``
is populated only when the underlying source provides a page boundary (PDF).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentSection(BaseModel):
    """A contiguous source region, such as a PDF page or DOCX heading block."""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(min_length=1)
    kind: Literal["page", "heading", "body", "table"]
    title: str | None = None
    text: str
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_offsets(self):
        if self.source_end < self.source_start:
            raise ValueError("source_end must be greater than or equal to source_start")
        if self.source_end - self.source_start != len(self.text):
            raise ValueError("section offsets must span exactly its text")
        return self


class DocumentChunk(BaseModel):
    """A retrieval-ready text unit that never crosses a source section."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_offsets(self):
        if self.source_end < self.source_start:
            raise ValueError("source_end must be greater than or equal to source_start")
        if self.source_end - self.source_start != len(self.text):
            raise ValueError("chunk offsets must span exactly its text")
        return self


class SkillEvidence(BaseModel):
    """A deterministic skill match tied to a source range and optional page."""

    model_config = ConfigDict(frozen=True)

    skill: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    section_id: str | None = None
    chunk_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_offsets(self):
        if self.source_end < self.source_start:
            raise ValueError("source_end must be greater than or equal to source_start")
        if self.source_end - self.source_start != len(self.excerpt):
            raise ValueError("evidence offsets must span exactly its excerpt")
        return self


class ParsedDocument(BaseModel):
    """Versioned parser result for a single uploaded resume."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["parsed-document-v1"] = "parsed-document-v1"
    filename: str = Field(min_length=1)
    media_type: Literal["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    text: str
    sections: list[DocumentSection]
    chunks: list[DocumentChunk]

    @model_validator(mode="after")
    def _validate_provenance(self):
        section_ids = {section.section_id for section in self.sections}
        for section in self.sections:
            if self.text[section.source_start:section.source_end] != section.text:
                raise ValueError("section text does not match document text")
        for chunk in self.chunks:
            if chunk.section_id not in section_ids:
                raise ValueError("chunk references an unknown section")
            if self.text[chunk.source_start:chunk.source_end] != chunk.text:
                raise ValueError("chunk text does not match document text")
        return self
