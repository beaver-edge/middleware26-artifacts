"""Small prompt-template helper used by the processors.

This intentionally mirrors the tiny subset of LangChain's PromptTemplate API
used in this project: `PromptTemplate.from_template(...).format(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any


class _PreserveMissing(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class PromptTemplate:
    template: str

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        return cls(template=template)

    @property
    def input_variables(self) -> list[str]:
        return [
            field_name
            for _, field_name, _, _ in Formatter().parse(self.template)
            if field_name
        ]

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    def format_partial(self, **kwargs: Any) -> "PromptTemplate":
        return PromptTemplate(self.template.format_map(_PreserveMissing(**kwargs)))

    def __str__(self) -> str:
        return self.template
