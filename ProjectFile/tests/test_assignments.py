from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.assignments import GMT8, upcoming_assignments, schedule_reply


def schedule(folder, rows):
    folder.mkdir(exist_ok=True)
    (folder / "schedule.md").write_text(
        "# Test schedule\n\n| Assignment | Due (GMT+8) | Submission instructions |\n"
        "| --- | --- | --- |\n" + rows, encoding="utf-8")


def test_gmt8_boundary_and_earliest_first(tmp_path):
    schedule(tmp_path,
             "| Later | 2026-11-05 12:00 | Portal |\n"
             "| Expired | 2026-11-03 23:59 | Portal |\n"
             "| Exact cutoff | 2026-11-04 00:00 | Portal |\n"
             "| Next | 2026-11-04 00:01 | Portal |\n")
    # UTC Nov 3 at 16:00 is Nov 4 at midnight in GMT+8.
    items = upcoming_assignments(tmp_path, datetime(2026, 11, 3, 16, 0, tzinfo=timezone.utc))
    assert [item.name for item in items] == ["Next", "Later"]
    assert items[0].due.utcoffset().total_seconds() == 8 * 3600
    assert "00:01" in schedule_reply(items)
    assert "Expired" not in schedule_reply(items)


def test_course_code_and_omitted_reminder_match_host_format(tmp_path):
    (tmp_path / "schedule.md").write_text(
        "| Course Code | Assignment | Due (GMT+8) | Reminder |\n"
        "| --- | --- | --- | --- |\n"
        "| COMP 4620SEF | lab9 Web Service | 2026-11-09 23:59 |\n"
        "| COMP 4330SEF | Tutorial 9 | 2026-11-11 23:59 | |\n"
        "| COMP 4330SEF | Project Slides (initial) | 2026-11-15 23:59 | Bring slides |\n")
    items = upcoming_assignments(tmp_path, datetime(2026, 11, 3, tzinfo=GMT8))
    assert len(items) == 3
    assert items[0].course_code == "COMP 4620SEF"
    assert items[0].instructions == items[1].instructions == ""
    reply = schedule_reply(items)
    assert "COMP 4620SEF — lab9 Web Service" in reply
    assert "Bring slides" in reply


def test_columns_can_be_reordered_and_required_date_cannot_be_omitted(tmp_path):
    path = tmp_path / "schedule.md"
    path.write_text("| Due (GMT+8) | Assignment |\n| --- | --- |\n| 2026-11-10 12:00 | Task |\n")
    assert len(upcoming_assignments(tmp_path, datetime(2026, 11, 3, tzinfo=GMT8))) == 1
    path.write_text("| Assignment | Due (GMT+8) |\n| --- | --- |\n| Task |\n")
    with pytest.raises(ValueError, match="due date"):
        upcoming_assignments(tmp_path)


def test_multiple_files_and_fresh_edits(tmp_path):
    schedule(tmp_path, "| A | 2026-11-10 23:59 | Portal |\n")
    second = tmp_path / "second.md"
    second.write_text((tmp_path / "schedule.md").read_text().replace("| A |", "| B |"))
    now = datetime(2026, 11, 3, tzinfo=GMT8)
    assert len(upcoming_assignments(tmp_path, now)) == 2
    second.write_text(second.read_text().replace("2026-11-10", "2026-10-10"))
    assert [a.name for a in upcoming_assignments(tmp_path, now)] == ["A"]


@pytest.mark.parametrize("date", ["2026-02-30 12:00", "2026-11-04", "tomorrow", "2026-11-04 25:00"])
def test_invalid_deadline_is_reported_instead_of_silently_omitted(tmp_path, date):
    schedule(tmp_path, f"| A | {date} | Portal |\n")
    with pytest.raises(ValueError, match="schedule.md, line"):
        upcoming_assignments(tmp_path)


def test_missing_and_expired_schedules(tmp_path):
    with pytest.raises(ValueError, match="not added"):
        upcoming_assignments(tmp_path)
    schedule(tmp_path, "| Old | 2020-01-01 12:00 | Portal |\n")
    items = upcoming_assignments(tmp_path, datetime(2026, 11, 3, tzinfo=GMT8))
    assert items == []
    assert "no upcoming" in schedule_reply(items)


def test_assignment_chat_has_one_button_and_does_not_read_pdfs(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ASSIGNMENTS_DIR", str(tmp_path / "assignments"))
    schedule(tmp_path / "assignments", "| Example future task | 2099-01-01 12:00 | Submit on portal |\n")
    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py")).run()
    def forbidden(*args, **kwargs):
        raise AssertionError("Assignment chat must not read or index PDFs")
    monkeypatch.setattr("rag.index.Corpus.refresh_if_needed", forbidden)
    app.radio[0].set_value("Assignment deadlines").run()
    assert not app.exception
    assert len(app.button) == 1
    assert app.button[0].label == "Tell me when to submit"
    assert not app.chat_input
    app.button[0].click().run()
    assert not app.exception
    assert any("Example future task" in m.value for m in app.markdown)
    schedule(tmp_path / "assignments", "| Example expired task | 2020-01-01 12:00 | Portal |\n")
    app.button[0].click().run()
    assert not app.exception
    assert any("no upcoming" in m.value for m in app.markdown)
