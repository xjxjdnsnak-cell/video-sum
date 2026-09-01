import sqlite3
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import settings
from ..db import get_db


@dataclass
class SearchResult:
    video_id: int
    title: str
    source_type: str
    source_table: str
    start: float
    end: float
    text: str
    score: float
    source: str


@dataclass
class Evidence:
    start: float
    end: float
    text: str
    score: float
    source: str


def check_fts5_support() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE test USING fts5(content)")
        conn.close()
        return True
    except Exception:
        return False


def init_fts_tables():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts USING fts5(
                video_id,
                segment_id,
                start,
                "end",
                text
            )
        """)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_summaries USING fts5(
                video_id,
                chunk_id,
                start,
                "end",
                text
            )
        """)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_final USING fts5(
                video_id,
                summary_id,
                text
            )
        """)
        conn.commit()


def rebuild_transcript_index():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fts_transcripts")
        cursor.execute("""
            INSERT INTO fts_transcripts(video_id, segment_id, start, "end", text)
            SELECT video_id, id, start, "end", text
            FROM transcript_segments
        """)
        conn.commit()


def rebuild_summary_index():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fts_summaries")
        cursor.execute("""
            INSERT INTO fts_summaries(video_id, chunk_id, start, "end", text)
            SELECT video_id, id, start, "end", summary
            FROM summary_chunks
        """)
        conn.commit()


def rebuild_final_index():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fts_final")
        cursor.execute("""
            INSERT INTO fts_final(video_id, summary_id, text)
            SELECT video_id, id, 
                   COALESCE(one_sentence_summary, '') || ' ' || 
                   COALESCE(detailed_summary, '') || ' ' ||
                   COALESCE(key_points, '') || ' ' ||
                   COALESCE(questions, '')
            FROM final_summaries
        """)
        conn.commit()


def rebuild_all_indexes():
    init_fts_tables()
    rebuild_transcript_index()
    rebuild_summary_index()
    rebuild_final_index()


def _prepare_query(query: str) -> str:
    terms = re.findall(r'\w+', query.lower())
    if not terms:
        return query
    return ' '.join(f'"{t}"' for t in terms)


def _bm25_score(text: str, terms: List[str]) -> float:
    if not terms:
        return 0.0
    text_lower = text.lower()
    count = sum(1 for term in terms if term in text_lower)
    return count / len(terms)


def search_fts(
    query: str,
    video_id: Optional[int] = None,
    limit: int = 20
) -> List[SearchResult]:
    if not check_fts5_support():
        return search_like(query, video_id, limit)
    
    results = []
    prepared_query = _prepare_query(query)
    terms = re.findall(r'\w+', query.lower())
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        if video_id:
            fts_query = f"fts_transcripts MATCH ? AND video_id = ?"
            cursor.execute(f"""
                SELECT video_id, segment_id, start, "end", text, bm25(fts_transcripts) as rank
                FROM fts_transcripts
                WHERE {fts_query}
                LIMIT ?
            """, (prepared_query, video_id, limit))
            for row in cursor.fetchall():
                results.append(SearchResult(
                    video_id=row[0],
                    title=_get_video_title(conn, row[0]),
                    source_type="local",
                    source_table="transcript_segments",
                    start=row[2],
                    end=row[3],
                    text=row[4],
                    score=-row[5],
                    source="transcript"
                ))
            
            fts_query = f"fts_summaries MATCH ? AND video_id = ?"
            cursor.execute(f"""
                SELECT video_id, chunk_id, start, "end", text, bm25(fts_summaries) as rank
                FROM fts_summaries
                WHERE {fts_query}
                LIMIT ?
            """, (prepared_query, video_id, limit))
            for row in cursor.fetchall():
                results.append(SearchResult(
                    video_id=row[0],
                    title=_get_video_title(conn, row[0]),
                    source_type="local",
                    source_table="summary_chunks",
                    start=row[2],
                    end=row[3],
                    text=row[4],
                    score=-row[5],
                    source="chunk_summary"
                ))
        else:
            cursor.execute(f"""
                SELECT video_id, segment_id, start, "end", text, bm25(fts_transcripts) as rank
                FROM fts_transcripts
                WHERE fts_transcripts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (prepared_query, limit))
            for row in cursor.fetchall():
                results.append(SearchResult(
                    video_id=row[0],
                    title=_get_video_title(conn, row[0]),
                    source_type="local",
                    source_table="transcript_segments",
                    start=row[2],
                    end=row[3],
                    text=row[4],
                    score=-row[5],
                    source="transcript"
                ))
            
            cursor.execute(f"""
                SELECT video_id, chunk_id, start, "end", text, bm25(fts_summaries) as rank
                FROM fts_summaries
                WHERE fts_summaries MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (prepared_query, limit))
            for row in cursor.fetchall():
                results.append(SearchResult(
                    video_id=row[0],
                    title=_get_video_title(conn, row[0]),
                    source_type="local",
                    source_table="summary_chunks",
                    start=row[2],
                    end=row[3],
                    text=row[4],
                    score=-row[5],
                    source="chunk_summary"
                ))
            
            cursor.execute(f"""
                SELECT video_id, summary_id, text, bm25(fts_final) as rank
                FROM fts_final
                WHERE fts_final MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (prepared_query, limit))
            for row in cursor.fetchall():
                results.append(SearchResult(
                    video_id=row[0],
                    title=_get_video_title(conn, row[0]),
                    source_type="local",
                    source_table="final_summaries",
                    start=0.0,
                    end=0.0,
                    text=row[2],
                    score=-row[3],
                    source="final_summary"
                ))
    
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


def search_like(
    query: str,
    video_id: Optional[int] = None,
    limit: int = 20
) -> List[SearchResult]:
    results = []
    like_pattern = f"%{query}%"
    terms = re.findall(r'\w+', query.lower())
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        if video_id:
            cursor.execute("""
                SELECT video_id, id, start, "end", text
                FROM transcript_segments
                WHERE video_id = ? AND text LIKE ?
            """, (video_id, like_pattern))
        else:
            cursor.execute("""
                SELECT video_id, id, start, "end", text
                FROM transcript_segments
                WHERE text LIKE ?
                LIMIT ?
            """, (like_pattern, limit))
        
        for row in cursor.fetchall():
            score = _bm25_score(row[4], terms)
            results.append(SearchResult(
                video_id=row[0],
                title=_get_video_title(conn, row[0]),
                source_type="local",
                source_table="transcript_segments",
                start=row[2],
                end=row[3],
                text=row[4],
                score=score,
                source="transcript"
            ))
        
        if video_id:
            cursor.execute("""
                SELECT video_id, id, start, "end", summary
                FROM summary_chunks
                WHERE video_id = ? AND summary LIKE ?
            """, (video_id, like_pattern))
        else:
            cursor.execute("""
                SELECT video_id, id, start, "end", summary
                FROM summary_chunks
                WHERE summary LIKE ?
                LIMIT ?
            """, (like_pattern, limit))
        
        for row in cursor.fetchall():
            score = _bm25_score(row[4], terms)
            results.append(SearchResult(
                video_id=row[0],
                title=_get_video_title(conn, row[0]),
                source_type="local",
                source_table="summary_chunks",
                start=row[2],
                end=row[3],
                text=row[4],
                score=score,
                source="chunk_summary"
            ))
    
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


def _get_video_title(conn: sqlite3.Connection, video_id: int) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM videos WHERE id = ?", (video_id,))
    row = cursor.fetchone()
    return row[0] if row else "Unknown"


def get_evidence(
    video_id: int,
    question: str,
    top_k: int = 5
) -> List[Evidence]:
    terms = re.findall(r'\w+', question.lower())
    if not terms:
        return []
    
    results = search_fts(question, video_id, top_k * 2)
    evidence_list = []
    
    seen_texts = set()
    for result in results:
        if result.text in seen_texts:
            continue
        seen_texts.add(result.text)
        
        evidence_list.append(Evidence(
            start=result.start,
            end=result.end,
            text=result.text,
            score=result.score,
            source=result.source
        ))
        
        if len(evidence_list) >= top_k:
            break
    
    return evidence_list


def get_transcript_for_qa(video_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT start, "end", text
            FROM transcript_segments
            WHERE video_id = ?
            ORDER BY start
        """, (video_id,))
        return [{"start": row[0], "end": row[1], "text": row[2]} for row in cursor.fetchall()]


def get_summary_for_qa(video_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT start, "end", summary
            FROM summary_chunks
            WHERE video_id = ?
            ORDER BY start
        """, (video_id,))
        return [{"start": row[0], "end": row[1], "text": row[2]} for row in cursor.fetchall()]


def get_final_summary_for_qa(video_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT one_sentence_summary, detailed_summary, key_points, questions
            FROM final_summaries
            WHERE video_id = ?
        """, (video_id,))
        row = cursor.fetchone()
        if row:
            return {
                "one_sentence_summary": row[0],
                "detailed_summary": row[1],
                "key_points": row[2],
                "questions": row[3]
            }
        return None
