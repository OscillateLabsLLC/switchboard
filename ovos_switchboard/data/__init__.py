"""Bundled data: the canonical schema, reviewed synthetic examples, and the measured LLM prompts."""
import json
from importlib import resources


def text(name):
    return resources.files(__name__).joinpath(name).read_text()


def jsonl(name):
    return [json.loads(line) for line in text(name).splitlines() if line.strip()]


def prompt(name):
    return resources.files(f"{__name__}.prompts").joinpath(name).read_text().strip()
