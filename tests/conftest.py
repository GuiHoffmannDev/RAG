from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.agents.graph import Claim, Draft, Grade, Rewrite, Verdict
from src.models import ScoredPassage

PASSAGE = ScoredPassage(
    score=0.83, source="errors.md", chunk=1, text="ARC-17 signals a checksum mismatch."
)


class FakeChat:
    """A fake chat model. `with_structured_output` returns the responder for the requested schema.
    `grades` holds one list of (passage_id, relevant) pairs per grading call; `prompts` records
    every grader input."""

    def __init__(self, grades):
        self.prompts = []
        self._grades = iter(grades)
        self.model = Mock(spec=ChatOpenAI)
        self.model.with_structured_output.side_effect = self._responders().__getitem__

    def _grade(self, prompt):
        self.prompts.append(prompt)
        return Grade(
            verdicts=[Verdict(passage_id=i, relevant=yes) for i, yes in next(self._grades)]
        )

    def _responders(self):
        draft = Draft(claims=[Claim(text="The checksum differs.", passage_ids=[1])])
        return {
            Grade: RunnableLambda(self._grade),
            Rewrite: RunnableLambda(lambda _: Rewrite(query="archive checksum")),
            Draft: RunnableLambda(lambda _: draft),
        }


@pytest.fixture
def fake_chat():
    return FakeChat
