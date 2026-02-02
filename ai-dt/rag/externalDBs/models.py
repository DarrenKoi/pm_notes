"""Glossary entry data model."""

from pydantic import BaseModel, Field


class GlossaryEntry(BaseModel):
    term: str = Field(description="Canonical term name (English full name)")
    aliases: list[str] = Field(default_factory=list, description="Abbreviations, Korean names, variants")
    definition: str = Field(default="", description="One-sentence definition")
    category: str = Field(default="", description="Classification keyword")
    source_ids: list[str] = Field(default_factory=list, description="Source snippet IDs")
