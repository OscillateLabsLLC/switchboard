"""Find the .dialog lines installed skills and pipeline plugins can speak: the assistant's own voice, which its
microphone can pick back up. These become decline negatives (device.harvest_dialog_negatives)."""
import glob
import importlib.util
import logging
import os
from importlib.metadata import entry_points


LOG = logging.getLogger(__name__)
ENTRY_POINT_GROUPS = ("ovos.plugin.skill", "opm.pipeline")


def package_dirs():
    """Package directories of every installed skill and pipeline plugin."""
    dirs = set()
    for group in ENTRY_POINT_GROUPS:
        for ep in entry_points(group=group):
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


def installed_dialog_lines(lang="en-us", roots=None):
    """Every line of every installed .dialog file for `lang`."""
    lines = []
    for root in roots if roots is not None else package_dirs():
        for path in dialog_files(root, lang):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    lines += f.read().splitlines()
            except OSError as e:
                LOG.warning("skipping unreadable dialog file %s: %s", path, e)
    return lines
