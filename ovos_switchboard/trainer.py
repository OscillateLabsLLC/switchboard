"""Rebuilds the router from the live manifest whenever registrations change.

Skills register in bursts at startup, so a change schedules a rebuild after `debounce_s` of quiet. Training runs on a
worker thread; the new router replaces the old one atomically, so matching never sees a half-built router and keeps
using the previous one while the next trains.
"""
import logging
import threading
import time

from ovos_switchboard.device import Device, bundled_jsonl, harvest_dialog_negatives
from ovos_switchboard.dialogs import installed_dialog_lines
from ovos_switchboard.router import Router
from ovos_switchboard.tools import build_tools

LOG = logging.getLogger(__name__)
SYNTHETIC_FILE = "synthetic_examples_slotted.jsonl"


class RouterTrainer:
    def __init__(self, registry, schema, router_config, llm=None, lang="en-us", debounce_s=5.0, dialog_negatives=True,
                 dialog_lines=installed_dialog_lines):
        self.registry, self.schema, self.router_config, self.llm = registry, schema, router_config, llm
        self.lang, self.debounce_s, self.dialog_negatives = lang, debounce_s, dialog_negatives
        self.dialog_lines = dialog_lines
        self.router = None
        self.trained = threading.Event()  # set after every completed build; tests and shadow tooling wait on it
        self.timer = None
        self.lock = threading.Lock()
        self.synthetic = bundled_jsonl(SYNTHETIC_FILE)

    def schedule(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(self.debounce_s, self.rebuild)
            self.timer.daemon = True
            self.timer.start()

    def rebuild(self):
        t0 = time.perf_counter()
        definitions = self.registry.definitions()
        if not definitions:
            LOG.info("no intents registered yet; router not built")
            return None
        try:
            synthetic = list(self.synthetic)
            if self.dialog_negatives:
                synthetic += harvest_dialog_negatives(self.dialog_lines(self.lang), definitions, self.lang)
            device = Device.from_definitions(definitions, synthetic, self.lang)
            router = Router(device, self.schema, build_tools(definitions, self.lang), self.llm, self.router_config)
        except Exception:
            LOG.exception("router rebuild failed; keeping the previous router")
            return None
        self.router = router
        self.trained.set()
        labels = sum(len(v) for v in definitions.values())
        LOG.info(f"router rebuilt: {labels} intents from {len(definitions)} skills, "
                 f"{len(router.classifier.classes)} classes in {time.perf_counter() - t0:.1f}s")
        return router

    def shutdown(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
