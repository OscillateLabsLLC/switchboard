"""Stage 2: deterministic template match, BEFORE anything trained.

A phrase matching templates of exactly one owner routes to that owner, even an unmapped skill: it is a declared
phrase. A collision between 2+ owners returns None so the classifier decides. No stochastic matcher here, ever.
"""
import re

from ovos_switchboard.core.templates import literal_words, template_regex


class ExactLayer:
    def __init__(self, device, schema, min_literal=1):
        entries = list(device.template_entries())
        self.schema = schema
        # Wildcard-heavy templates ('play {query}') still train the classifier but don't claim the exact layer:
        # they're catch-alls, not magic phrases.
        self.templates = [(label, template_regex(line)) for label, line in entries if literal_words(line) >= min_literal]
        self.prefix_templates = []
        self._entries = entries

    def route(self, utterance):
        return self._unique_owner(self.templates, utterance)

    def enable_prefix(self, min_literal):
        """Trailing speech: someone keeps talking before VAD cuts off ("make me laugh hey dad ...").
        Slot-free templates of >= min_literal words also match as a PREFIX followed by more speech.
        Slot templates are excluded: a trailing {slot} already absorbs extra words."""
        self.prefix_templates = [
            (label, re.compile(template_regex(line).pattern.removesuffix(r"\s*[?.!]*$") + r"[\s?.!,]+\S"))
            for label, line in self._entries if "{" not in line and literal_words(line) >= min_literal]

    def prefix_route(self, utterance):
        return self._unique_owner(self.prefix_templates, utterance)

    def _unique_owner(self, templates, utterance):
        owners = {self.schema.canon_or_unmapped(label) for label, rx in templates if rx.match(utterance)}
        return owners.pop() if len(owners) == 1 else None
