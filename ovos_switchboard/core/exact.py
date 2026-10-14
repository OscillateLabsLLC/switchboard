"""Stage 2: deterministic template match, BEFORE anything trained.

A phrase matching templates of exactly one owner routes to that owner, even an unmapped skill: it is a declared
phrase. A collision between 2+ owners returns None so the classifier decides. No stochastic matcher here, ever.
"""
import re

from ovos_switchboard.core.templates import is_catch_all, literal_words, template_regex


class ExactLayer:
    """Utterances that match declared template phrases, routed only when one intent owns them."""
    def __init__(self, device, schema, min_literal=1):
        entries = list(device.template_entries())
        self.schema = schema
        # Templates with no literal word ('{query}') still train the classifier but don't claim the exact layer.
        # One-word catch-alls ('play {query}') do claim it, but ExactStage can let a confident classifier overrule.
        self.templates = [(label, template_regex(line)) for label, line in entries
                          if literal_words(line) >= min_literal]
        self.catch_all = {template_regex(line).pattern for _, line in entries if is_catch_all(line)}
        self.prefix_templates = []
        self._entries = entries

    def route(self, utterance):
        """The canonical intent whose templates alone match the utterance, else None."""
        return self.match(utterance)[0]

    def match(self, utterance):
        """(owner, catch_all_only): the unique owner as in route(), and whether every matching template was a
        catch-all ('play {query}'), the only kind the audit found claiming speech nobody meant as a command."""
        hits = [(label, rx) for label, rx in self.templates if rx.match(utterance)]
        owners = {self.schema.canon_or_unmapped(label) for label, _ in hits}
        if len(owners) != 1:
            return None, False
        return owners.pop(), all(rx.pattern in self.catch_all for _, rx in hits)

    def enable_prefix(self, min_literal):
        """Trailing speech: someone keeps talking before VAD cuts off ("make me laugh hey dad ...").
        Slot-free templates of >= min_literal words also match as a PREFIX followed by more speech.
        Slot templates are excluded: a trailing {slot} already absorbs extra words."""
        self.prefix_templates = [
            (label, re.compile(template_regex(line).pattern.removesuffix(r"\s*[?.!]*$") + r"[\s?.!,]+\S"))
            for label, line in self._entries if "{" not in line and literal_words(line) >= min_literal]

    def prefix_route(self, utterance):
        """Like route, but for trailing speech after a slot-free template (enable_prefix)."""
        return self._unique_owner(self.prefix_templates, utterance)

    def _unique_owner(self, templates, utterance):
        owners = {self.schema.canon_or_unmapped(label) for label, rx in templates if rx.match(utterance)}
        return owners.pop() if len(owners) == 1 else None
