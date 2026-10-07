"""Small HTML tables for notebook display (``_repr_html_``)."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, Optional, Sequence

MAX_ROWS = 20   # longer tables show the first rows and a count of the rest


def number(x: Any) -> str:
    if isinstance(x, (bool, int)) or (hasattr(x, "dtype") and x.dtype.kind in "biu"):
        return f"{int(x):,}"
    x = float(x)
    return "–" if x != x else f"{x:.10g}"   # NaN: not set


def table(rows: Sequence[Sequence[Any]], header: Optional[Sequence[str]] = None,
          title: Optional[str] = None, total: Optional[int] = None) -> str:
    """An HTML table of already formatted cells (escaped here). ``total``:
    the full number of rows, when ``rows`` is the first part only."""
    out = ["<table>"]
    if title:
        out.append(f"<caption style='text-align:left'><b>{escape(title)}</b></caption>")
    if header:
        out.append("<tr>" + "".join(f"<th>{escape(h)}</th>" for h in header) + "</tr>")
    for row in rows[:MAX_ROWS]:
        out.append("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>")
    more = (total if total is not None else len(rows)) - min(len(rows), MAX_ROWS)
    if more > 0:
        span = len(header) if header else max((len(r) for r in rows), default=1)
        out.append(f"<tr><td colspan='{span}'>… and {more:,} more</td></tr>")
    out.append("</table>")
    return "".join(out)


def fields(title: str, pairs: Iterable[Sequence[Any]]) -> str:
    """A two-column name/value table."""
    return table([(name, value) for name, value in pairs], title=title)
