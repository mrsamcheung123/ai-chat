"""Read exact deadlines from Markdown, independently of PDF retrieval and models."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

GMT8 = timezone(timedelta(hours=8), name="GMT+8")
REQUIRED_COLUMNS = {"assignment", "due (gmt+8)"}


@dataclass(frozen=True)
class Assignment:
    name: str
    due: datetime
    instructions: str
    source: str
    course_code: str = ""


def read_assignments(folder: Path):
    files = sorted(folder.glob("*.md"))
    if not files:
        raise ValueError("The host has not added a Markdown assignment schedule yet.")
    assignments = []
    for path in files:
        active = False
        found = False
        columns = []
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line.startswith("|"):
                active = False
                continue
            # Remove only the outer pipes, preserving genuinely empty cells.
            body = line[1:]
            if body.endswith("|"):
                body = body[:-1]
            cells = [cell.strip() for cell in body.split("|")]
            headings = [re.sub(r"\s+", " ", cell).casefold() for cell in cells]
            if REQUIRED_COLUMNS.issubset(headings):
                if len(set(headings)) != len(headings):
                    raise ValueError(f"{path.name}, line {number}: column headings must be unique.")
                columns = headings
                active = found = True
                continue
            if not active:
                continue
            if len(cells) == len(columns) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            try:
                if len(cells) > len(columns):
                    raise ValueError("The row has more cells than the table headings.")
                # Markdown permits omitted trailing cells; optional reminders can be blank.
                row = dict(zip(columns, cells + [""] * (len(columns) - len(cells))))
                name, date = row["assignment"], row["due (gmt+8)"]
                if not name:
                    raise ValueError("Each row needs an assignment name.")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", date):
                    raise ValueError("Use YYYY-MM-DD HH:MM for the due date.")
                due = datetime.strptime(date, "%Y-%m-%d %H:%M").replace(tzinfo=GMT8)
                notes = "\n".join(row.get(key, "") for key in ("submission instructions", "reminder")
                                  if row.get(key))
                assignments.append(Assignment(name, due, notes, path.name, row.get("course code", "")))
            except ValueError as exc:
                raise ValueError(f"{path.name}, line {number}: {exc}") from exc
        if not found:
            raise ValueError(f"{path.name}: the table needs Assignment and Due (GMT+8) columns.")
    return assignments


def upcoming_assignments(folder, now=None):
    now = now if now is not None else datetime.now(GMT8)
    if now.tzinfo is None:
        raise ValueError("The current time must include a timezone.")
    return sorted((item for item in read_assignments(folder) if item.due > now),
                  key=lambda item: (item.due, item.name.casefold()))


def schedule_reply(items):
    if not items:
        return "There are no upcoming assignment deadlines in the host’s schedule."
    lines = ["Here are your upcoming submission deadlines (GMT+8), earliest first:"]
    for item in items:
        title = f"{item.course_code} — {item.name}" if item.course_code else item.name
        lines.append(f"\n{title}\nSubmit by {item.due.strftime('%A, %d %B %Y at %H:%M')} (GMT+8).")
        if item.instructions:
            lines.append(item.instructions)
        lines.append(f"Source: {item.source}")
    return "\n".join(lines)
