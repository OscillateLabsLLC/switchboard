from ovos_switchboard.adapters.dialog_files import installed_dialog_lines, package_dirs
from ovos_switchboard.core.device import harvest_dialog_negatives


def test_dialog_lines_from_skill_locale_folders(tmp_path):
    (tmp_path / "skill_a" / "locale" / "en-us" / "dialog").mkdir(parents=True)
    (tmp_path / "skill_a" / "locale" / "en-us" / "dialog" / "done.dialog").write_text("All done.\n# comment\n")
    (tmp_path / "skill_b" / "dialog" / "en-us").mkdir(parents=True)
    (tmp_path / "skill_b" / "dialog" / "en-us" / "off.dialog").write_text("(The|Your) {device} is off\n")
    (tmp_path / "skill_b" / "dialog" / "de-de").mkdir(parents=True)
    (tmp_path / "skill_b" / "dialog" / "de-de" / "off.dialog").write_text("Ausgeschaltet\n")
    lines = installed_dialog_lines("en-us", roots=[str(tmp_path)])
    assert sorted(lines) == ["# comment", "(The|Your) {device} is off", "All done."]


def test_a_package_with_too_many_lines_is_sampled_evenly(tmp_path):
    folder = tmp_path / "skill_history" / "locale" / "en-us"
    folder.mkdir(parents=True)
    (folder / "events.dialog").write_text("# events\n" + "".join(f"Event number {i}.\n" for i in range(100)))
    lines = installed_dialog_lines("en-us", roots=[str(tmp_path / "skill_history")], max_per_package=10)
    assert lines == [f"Event number {i}." for i in range(0, 100, 10)]


def test_a_package_under_the_cap_keeps_every_line(tmp_path):
    folder = tmp_path / "skill_small" / "locale" / "en-us"
    folder.mkdir(parents=True)
    (folder / "done.dialog").write_text("# comment\nAll done.\nFinished.\n")
    assert installed_dialog_lines("en-us", roots=[str(tmp_path / "skill_small")], max_per_package=2) == [
        "# comment", "All done.", "Finished."]


def test_negatives_skip_comments_short_lines_and_real_requests():
    definitions = {"s": [{"skill_id": "s", "intent_name": "time", "lang": "en-us", "method": "template",
                          "definition": {"samples": ["what time is it"]}}]}
    lines = ["# a comment", "Okay.", "(The|Your) {device} is off", "What time is it?"]
    got = [r["example"] for r in harvest_dialog_negatives(lines, definitions, "en-us")]
    assert got == ["the is off", "your is off"]


def test_package_dirs_are_importable_locations():
    assert all(isinstance(d, str) for d in package_dirs())
