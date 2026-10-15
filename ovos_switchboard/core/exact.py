"""Stage 2: deterministic template match, BEFORE anything trained.

A phrase matching templates of exactly one owner routes to that owner, even an unmapped skill: it is a declared
phrase. A collision between 2+ owners returns None so the classifier decides, unless ExactOptions.skill_priority
lists one of the colliding skills, collision_priority names a group (core or user skills) with exactly one owner
among them, or collision_resolution "specific" (the default) finds one owner with the most specific matching
template: rules, not guesses. No stochastic matcher here, ever.
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
        # the same patterns with each slot captured (slots are the only "(?:.+?)": template text has "." escaped)
        self.slot_regex = {rx.pattern: re.compile(rx.pattern.replace("(?:.+?)", "(.+?)")) for _, rx in self.templates}
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
        owner = owners.pop() if len(owners) == 1 else self._prioritized(hits) or self._most_specific(hits, utterance)
        if owner is None:
            return None, False
        return owner, all(rx.pattern in self.catch_all for o, _, rx in hits if o == owner)

    def _prioritized(self, hits):
        """The colliding owner skill_priority or collision_priority picks, else None."""
        skills = {}
        for owner, label, _ in hits:
            skills.setdefault(owner, set()).add(label.split(":", 1)[0])
        return self._listed(skills) or self._grouped(skills)

    def _most_specific(self, hits, utterance):
        """With collision_resolution "specific": the one owner whose matching template covers the most words of the
        utterance with literal text rather than slots."""
        if self.options.collision_resolution != "specific":
            return None
        best = {}
        for owner, _, rx in hits:
            best[owner] = max(best.get(owner, 0), self._literal_coverage(rx, utterance))
        winners = [o for o, n in best.items() if n == max(best.values())]
        return winners[0] if len(winners) == 1 else None

    def _literal_coverage(self, rx, utterance):
        """Words of the utterance a template matched literally: all of them minus those its slots captured."""
        m = self.slot_regex[rx.pattern].match(utterance)
        slot_words = sum(len(g.split()) for g in m.groups() if g)
        return len(utterance.split()) - slot_words

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
