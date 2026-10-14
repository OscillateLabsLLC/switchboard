"""Stage 2: deterministic template match, BEFORE anything trained.

A phrase matching templates of exactly one owner routes to that owner, even an unmapped skill: it is a declared
phrase. A collision between 2+ owners returns None so the classifier decides, unless ExactOptions.skill_priority
lists one of the colliding skills or collision_priority names a group (core or user skills) with exactly one owner
among them: a rule, not a guess. No stochastic matcher here, ever.
"""
import re

from ovos_switchboard.core.options import ExactOptions
from ovos_switchboard.core.templates import is_catch_all, literal_words, template_regex


class ExactLayer:
    """Utterances that match declared template phrases, routed only when one intent owns them."""
    def __init__(self, device, schema, min_literal=1, options=ExactOptions()):
        entries = list(device.template_entries())
        self.schema, self.options = schema, options
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
        hits = [(self.schema.canon_or_unmapped(label), label, rx)
                for label, rx in self.templates if rx.match(utterance)]
        owners = {owner for owner, _, _ in hits}
        owner = owners.pop() if len(owners) == 1 else self._prioritized(hits)
        if owner is None:
            return None, False
        return owner, all(rx.pattern in self.catch_all for o, _, rx in hits if o == owner)

    def _prioritized(self, hits):
        """The colliding owner skill_priority or collision_priority picks, else None."""
        skills = {}
        for owner, label, _ in hits:
            skills.setdefault(owner, set()).add(label.split(":", 1)[0])
        return self._listed(skills) or self._grouped(skills)

    def _listed(self, skills):
        """The one owner holding the earliest skill_priority skill among the collision, else None."""
        ranks = {}
        for owner, ids in skills.items():
            listed = [r for r in map(self.options.priority_rank, ids) if r is not None]
            if listed:
                ranks[owner] = min(listed)
        winners = [o for o, r in ranks.items() if r == min(ranks.values())]
        return winners[0] if len(winners) == 1 else None

    def _grouped(self, skills):
        """The one owner whose skills are all in the collision_priority group, else None."""
        if self.options.collision_priority == "none":
            return None
        want_core = self.options.collision_priority == "core"
        preferred = [o for o, ids in skills.items() if all(self.options.is_core(k) == want_core for k in ids)]
        return preferred[0] if len(preferred) == 1 else None

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
