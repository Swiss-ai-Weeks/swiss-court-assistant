"""Which decisions cite which statute articles: a lookup index over the corpus statute references.

    uv run python -m swiss_court_assistant.statute_links build
    uv run python -m swiss_court_assistant.statute_links show "Art. 41 OR"

`data/raw/graph/statute_references.parquet` holds 12.8M (decision, law code, article) mentions, with
each language's own abbreviation (OR/CO, ZGB/CC). The index folds them onto the German form, keeps
only decisions in the served corpus, and stores each decision's language and citation count, so search
can pull the most-cited decisions on an article the query names and score candidates by shared articles.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import threading
import time
from pathlib import Path

import polars as pl

from swiss_court_assistant.server.mentions import _MENTION
from swiss_court_assistant.statutes import GLOSSARY

REFERENCES = Path("data/raw/graph/statute_references.parquet")
DEFAULT_CORPUS = Path("data/vectordb/corpus.sqlite")
DEFAULT_GRAPH = Path("data/graph/corpus.citations.sqlite")
DEFAULT_INDEX = Path("data/graph/corpus.statutes.sqlite")

_CANON = {f.rstrip(".").upper(): row["de"][0].upper() for row in GLOSSARY for forms in row.values() for f in forms}
# Codes outside the glossary that the reference extractor emits in French/Italian form.
_CANON.update({"CEDH": "EMRK", "CEDU": "EMRK", "LTAF": "VGG", "OJ": "OG", "AUG": "AIG", "LETR": "AIG",
               "LSTR": "AIG", "LSTRI": "AIG", "LEI": "AIG", "LEF": "SCHKG", "LAINF": "UVG", "CST": "BV",
               "COST": "BV"})


def canon(code: str) -> str:
    c = code.rstrip(".").upper()
    return _CANON.get(c, c)


def query_articles(text: str) -> set[tuple[str, str]]:
    """Articles named in a query ("Art. 41 OR", "art. 336c al. 1 CO"), in the index's canonical form."""
    return {(canon(m.group("code")), m.group("num").replace(" ", "")) for m in _MENTION.finditer(text)}


def build(corpus: Path, graph: Path, out: Path) -> None:
    t = time.time()
    refs = (pl.read_parquet(REFERENCES, columns=["decision_id", "law_code", "article", "mention_count"])
            .with_columns(pl.col("law_code").map_elements(canon, return_dtype=pl.String).alias("law"))
            .group_by(["decision_id", "law", "article"]).agg(pl.col("mention_count").sum().alias("mentions")))
    with sqlite3.connect(f"file:{corpus}?mode=ro", uri=True) as con:
        decisions = pl.DataFrame(con.execute("SELECT decision_id, language, court FROM decisions").fetchall(),
                                 schema=["decision_id", "language", "court"], orient="row")
    with sqlite3.connect(f"file:{graph}?mode=ro", uri=True) as con:
        counts = pl.DataFrame(con.execute("SELECT decision_id, cited_by FROM counts").fetchall(),
                              schema=["decision_id", "cited_by"], orient="row")
    links = (refs.join(decisions, on="decision_id", how="inner")
             .join(counts, on="decision_id", how="left").with_columns(pl.col("cited_by").fill_null(0)))
    print(f"{len(refs):,} folded references, {len(links):,} in the corpus ({time.time() - t:.0f}s)", flush=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript("""
        CREATE TABLE links (decision_id TEXT, law TEXT, article TEXT, mentions INTEGER,
                            language TEXT, court TEXT, cited_by INTEGER);
        CREATE TABLE articles (law TEXT, article TEXT, decisions INTEGER, PRIMARY KEY (law, article));
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    con.executemany("INSERT INTO links VALUES (?, ?, ?, ?, ?, ?, ?)",
                    links.select(["decision_id", "law", "article", "mentions", "language", "court", "cited_by"]).iter_rows())
    con.executemany("INSERT INTO articles VALUES (?, ?, ?)",
                    links.group_by(["law", "article"]).agg(pl.len()).iter_rows())
    con.execute("INSERT INTO meta VALUES ('decisions', ?)", [str(links["decision_id"].n_unique())])
    con.executescript("""
        CREATE INDEX links_decision ON links(decision_id);
        CREATE INDEX links_article ON links(law, article, language, cited_by DESC);
    """)
    con.commit()
    con.close()
    tmp.replace(out)
    print(f"wrote {out} ({time.time() - t:.0f}s)")


class StatuteLinks:
    """Read-only statute-reference index (see module docstring)."""

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        self.decisions = int(self._con().execute("SELECT value FROM meta WHERE key = 'decisions'").fetchone()[0])

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = self._local.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        return con

    def articles_of(self, decision_ids: list[str]) -> dict[str, set[tuple[str, str]]]:
        out: dict[str, set[tuple[str, str]]] = {d: set() for d in decision_ids}
        ids = list(out)
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            for did, law, art in self._con().execute(
                    f"SELECT decision_id, law, article FROM links WHERE decision_id IN ({','.join('?' * len(batch))})", batch):
                out[did].add((law, art))
        return out

    def idf(self, articles: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
        out = {}
        for law, art in articles:
            row = self._con().execute("SELECT decisions FROM articles WHERE law = ? AND article = ?", (law, art)).fetchone()
            out[(law, art)] = math.log(self.decisions / (1 + (row[0] if row else 0)))
        return out

    def most_cited(self, articles: set[tuple[str, str]], language: str | None, limit: int = 40) -> list[str]:
        """Decisions citing any of `articles`, most-cited first (optionally in one language)."""
        found: dict[str, int] = {}
        for law, art in articles:
            sql = "SELECT decision_id, cited_by FROM links WHERE law = ? AND article = ?"
            params: list = [law, art]
            if language:
                sql += " AND language = ?"
                params.append(language)
            for did, cited in self._con().execute(sql + " ORDER BY cited_by DESC LIMIT ?", params + [limit]):
                found[did] = max(found.get(did, 0), cited)
        return sorted(found, key=lambda d: (-found[d], d))[:limit]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    b.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    b.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    s = sub.add_parser("show")
    s.add_argument("query")
    s.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = p.parse_args()
    if args.command == "build":
        build(args.corpus, args.graph, args.out)
    else:
        links = StatuteLinks(args.index)
        arts = query_articles(args.query)
        print(arts, links.idf(arts))
        for lang in ("de", "fr", "it"):
            print(lang, links.most_cited(arts, lang, 10))


if __name__ == "__main__":
    main()
