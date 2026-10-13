"""Bundled data: the canonical schema, reviewed synthetic examples, and the measured LLM prompts."""
import json
from importlib import resources


def text(name):
    """A bundled data file's text."""
    return resources.files(__name__).joinpath(name).read_text()


def jsonl(name):
    """A bundled JSONL file's rows."""
    return [json.loads(line) for line in text(name).splitlines() if line.strip()]


def prompt(name):
    """A bundled LLM prompt."""
    return resources.files(f"{__name__}.prompts").joinpath(name).read_text().strip()
