"""Agent tools over the full-corpus index: statute articles and court decisions.

The filters are the columns the index really has, and their values are codes, not prose: courts are
"bger"/"bvger", cantons are two letters, branches are German words, and a statute's abbreviation
follows the language it is published in (OR in German, CO in French). Two things keep the agent from
guessing: `list_values` prints the values actually present with their counts, and every filter
accepts a loose value, resolves it against the index and says in the result which value it used -
or, when nothing matches, which values exist. A filter that matches nothing returns no results
rather than quietly dropping the filter.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

from .library import Hit, Library

log = logging.getLogger(__name__)

READ_WINDOW = 8000


def _failed(e: Exception) -> tuple[str, dict]:
    log.exception("library tool failed")
    return f"The tool failed: {e}", {"summary": "failed", "error": True}


def _block(i: int, h: Hit) -> str:
    where = " · ".join(x for x in (h.court, h.canton, h.branch, h.language, h.date) if x)
    erw = f" · E. {', '.join(h.erwaegungen)}" if h.erwaegungen else ""
    ident = f"law_id={h.source_id}" if h.kind == "law" else f"decision_id={h.source_id}"
    return (f"Result {i}: {ident} chunk_id={h.chunk_id}\n{h.label} · {where}{erw}\n{h.text}")


def _results(hits: list[Hit], notes: list[str], what: str) -> tuple[str, dict]:
    head = ("\n".join(notes) + "\n\n") if notes else ""
    if not hits:
        return head + f"No {what} found.", {"summary": "nothing found", "notes": notes}
    body = "\n\n".join(_block(i, h) for i, h in enumerate(hits, 1))
    ids = sorted({h.source_id for h in hits if h.kind == "decision"})
    return head + body, {"summary": f"{len(hits)} passages", "decisions": ids, "notes": notes}


def make_library_tools(lib: Library) -> list:
    """The tools, bound to one Library. Every call hops to a thread: SQLite blocks the loop."""

    async def run(fn, *a, **kw):
        return await asyncio.to_thread(fn, *a, **kw)

    @tool(response_format="content_and_artifact")
    async def list_values(table: str, field: str) -> tuple[str, dict]:
        """The exact values a filter can take, most common first, with how many items have each.
        Call this before filtering on something you have not seen printed, and use the values
        verbatim. table is "decisions" or "laws".
        decisions: court, canton, branch, language, jurisdiction, chamber.
        laws: canton, language, abbreviation, category, sr_number."""
        try:
            rows = await run(lib.vocabulary, table, field)
        except Exception as e:
            return _failed(e)
        listed = ", ".join(f"{v} ({n:,})" for v, n in rows)
        return (f"{table}.{field}, {len(rows)} values shown, most common first:\n{listed}",
                {"summary": f"{len(rows)} values of {table}.{field}"})

    @tool(response_format="content_and_artifact")
    async def search_laws(query: str, abbreviation: str | None = None, canton: str | None = None,
                          language: str | None = None) -> tuple[str, dict]:
        """Find statute articles by their wording. Put an exact phrase in double quotes; all other
        words must appear. Filters: abbreviation (OR, CO, ZGB, CC, StGB ... - language-specific),
        canton ("CH" for federal law, else two letters like ZH), language (de, fr, it).
        To read one known article in full, use read_law instead."""
        try:
            hits, notes = await run(lib.keyword_search, query, "law", 8,
                                    abbreviation=abbreviation, canton=canton, language=language)
        except Exception as e:
            return _failed(e)
        return _results(hits, notes, "statute articles")

    @tool(response_format="content_and_artifact")
    async def read_law(abbreviation: str | None = None, article: str | None = None,
                       sr_number: str | None = None, language: str | None = None,
                       canton: str | None = None) -> tuple[str, dict]:
        """The verbatim text of one statute article. Give the abbreviation and the article number,
        e.g. abbreviation="OR", article="271". The abbreviation follows the language you want:
        OR/CO for the Code of Obligations, ZGB/CC for the Civil Code. Add language to pick one
        version, sr_number instead of the abbreviation if you know it (220 = OR), and canton for
        cantonal law ("CH" is federal)."""
        try:
            out = await run(lib.read_law, abbreviation, article, sr_number, language, canton)
        except Exception as e:
            return _failed(e)
        notes, arts = out["notes"], out["articles"]
        head = ("\n".join(notes) + "\n\n") if notes else ""
        if not arts:
            return head + "No such article in the index.", {"summary": "not found", "notes": notes}
        body = "\n\n".join(
            f"law_id={a['law_id']}\n{a['label']}" + (f" - {a['law_title']}" if a.get("law_title") else "")
            + f" [{a['language']}]\n{a['text']}" for a in arts)
        return head + body, {"summary": f"{len(arts)} article{'s' * (len(arts) > 1)}"}

    @tool(response_format="content_and_artifact")
    async def search_decisions(query: str, court: str | None = None, canton: str | None = None,
                               branch: str | None = None, language: str | None = None) -> tuple[str, dict]:
        """Find passages of court decisions by their wording. An exact phrase goes in double quotes;
        all other words must appear. Filters, all optional: court (a code such as bger, bvger, bge,
        bstger - call list_values("decisions", "court") for the full list), canton ("CH" for federal
        courts, else two letters such as ZH, BE, TI), branch (oeffentlich = public/administrative,
        zivil = civil, straf = criminal, sozialversicherung, unknown), language (de, fr, it)."""
        try:
            hits, notes = await run(lib.keyword_search, query, "decision", 8,
                                    court=court, canton=canton, branch=branch, language=language)
        except Exception as e:
            return _failed(e)
        return _results(hits, notes, "decisions")

    @tool(response_format="content_and_artifact")
    async def read_decision(decision_id: str, offset: int = 0) -> tuple[str, dict]:
        """Read one decision: its metadata and full text, 8,000 characters per call. Takes a
        decision_id from a search result, or a docket number. Pass offset to continue reading."""
        try:
            out = await run(lib.read_decision, decision_id, offset, READ_WINDOW)
        except Exception as e:
            return _failed(e)
        if out is None:
            return (f"No decision {decision_id!r} in the index.",
                    {"summary": "not found", "error": True})
        d, n = out["decision"], out["length"]
        head = [f"decision_id: {d['decision_id']}", out["label"],
                " · ".join(str(x) for x in (d.get("canton"), d.get("branch"), d.get("language")) if x)]
        if d.get("title"):
            head.append(f"Title: {d['title']}")
        if d.get("regeste") and out["start"] == 0:
            head.append(f"Regeste: {d['regeste']}")
        more = (f" Call read_decision(decision_id={d['decision_id']!r}, offset={out['end']}) to continue."
                if out["end"] < n else "")
        head.append(f"Full text, characters {out['start']}-{out['end']} of {n:,}.{more}")
        return ("\n".join(head) + "\n\n" + out["text"],
                {"summary": f"{out['label']}, characters {out['start']:,}–{out['end']:,} of {n:,}",
                 "decisions": [d["decision_id"]]})

    @tool(response_format="content_and_artifact")
    async def count_decisions(group_by: str, court: str | None = None, canton: str | None = None,
                              branch: str | None = None, language: str | None = None) -> tuple[str, dict]:
        """How many decisions there are per value of group_by, with the filters applied - for
        questions about coverage ("which courts decide social-insurance cases?", "how much is there
        in Italian?"). group_by and the filters are the same fields as search_decisions."""
        try:
            rows, notes = await run(lib.count, "decisions", group_by, 25,
                                    court=court, canton=canton, branch=branch, language=language)
        except Exception as e:
            return _failed(e)
        head = ("\n".join(notes) + "\n\n") if notes else ""
        if not rows:
            return head + "Nothing matches those filters.", {"summary": "0", "notes": notes}
        total = sum(n for _, n in rows)
        body = "\n".join(f"  {v}: {n:,}" for v, n in rows)
        return (head + f"decisions by {group_by} ({total:,} in total):\n{body}",
                {"summary": f"{len(rows)} groups, {total:,} decisions"})

    return [list_values, search_laws, read_law, search_decisions, read_decision, count_decisions]
