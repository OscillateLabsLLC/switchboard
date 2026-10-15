"""Find the .dialog lines installed skills and pipeline plugins can speak: the assistant's own voice, which its
microphone can pick back up. These become decline negatives (device.harvest_dialog_negatives)."""
import glob
import importlib.util
import logging
import os
from importlib.metadata import entry_points


LOG = logging.getLogger(__name__)
ENTRY_POINT_GROUPS = ("ovos.plugin.skill", "opm.pipeline")
# Some skills ship data as dialog files (days-in-history: 136,639 lines, one per historical event). Every line becomes
# a training row, and at that size the classifier's fit overflows SciPy's L-BFGS-B and crashes. The largest ordinary
# skill measured had 1,643 lines (dad jokes), so the cap leaves those as they were.
MAX_LINES_PER_PACKAGE = 2000


def package_dirs(exclude=()):
    """Package directories of every installed skill and pipeline plugin, except the skill ids in `exclude`."""
    dirs = set()
    for group in ENTRY_POINT_GROUPS:
        for ep in entry_points(group=group):
            if ep.name in exclude:
                continue
            top = ep.value.split(":", 1)[0].split(".", 1)[0]
            try:
                spec = importlib.util.find_spec(top)
            except (ImportError, ValueError):
                continue
            if spec and spec.submodule_search_locations:
                dirs.update(spec.submodule_search_locations)
    return sorted(dirs)


def dialog_files(root, lang):
    """The .dialog files for `lang` under `root`, in either locale layout."""
    patterns = [f"**/locale/{lang}/**/*.dialog", f"**/dialog/{lang}/**/*.dialog"]
    return sorted({f for p in patterns for f in glob.glob(os.path.join(root, p), recursive=True)})


def installed_dialog_lines(lang="en-us", roots=None, max_per_package=MAX_LINES_PER_PACKAGE, exclude=()):
    """Every line of every installed .dialog file for `lang`, at most `max_per_package` spoken lines per package,
    skipping the skill ids in `exclude`."""
    lines = []
    for root in roots if roots is not None else package_dirs(exclude):
        lines += _capped(root, _package_lines(root, lang), max_per_package)
    return lines


def _package_lines(root, lang):
    lines = []
    for path in dialog_files(root, lang):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines += f.read().splitlines()
        except OSError as e:
            LOG.warning("skipping unreadable dialog file %s: %s", path, e)
    return lines


def _capped(root, lines, limit):
    """The package's lines, or an evenly spaced, deterministic sample of its spoken lines when it has too many."""
    spoken = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if len(spoken) <= limit:
        return lines
    LOG.info("%s has %d dialog lines; using %d of them as decline negatives", root, len(spoken), limit)
    return [spoken[i * len(spoken) // limit] for i in range(limit)]
