"""
MVP2V2 FastAPI Backend — Museum Semantic Search (v5 Hybrid)
============================================================

Start with:
    cd MVP2V2 && uvicorn server:app --reload --port 8000

Search flow:
  0. Claude query analysis → intent, keyword_filters, conceptual_query
  1. Keyword filtering on artist/title/culture/classification
  2. Cosine similarity using conceptual_query (not raw query)
  3. Merge exact matches + cosine top 300 → re-ranking with raw query
  4. Return unified results sorted by rerank_score
  5. /search/more: progressive re-ranking of next N candidates
"""

import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

COMBINED_PATH = BASE / "data" / "combined_collection.json"
TAGS_PATH = BASE / "tags" / "all_artwork_tags.json"
TRANSLATIONS_PATH = BASE / "tags" / "louvre_translations.json"
EMBEDDINGS_PATH = BASE / "embeddings" / "embeddings.npz"
INDEX_PATH = BASE / "embeddings" / "embeddings_index.json"

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Re-ranking parameters (from Experiment_C_2/run_reranking.py)
# ---------------------------------------------------------------------------
RERANK_MODEL = "claude-sonnet-4-20250514"
RERANK_TEMPERATURE = 0
RERANK_MAX_TOKENS = 8192
RERANK_BATCH_SIZE = 75
RERANK_CONCURRENCY = 5
RERANK_MAX_RETRIES = 3
RERANK_TOP_N = 300

EMBEDDING_MODEL = "text-embedding-3-large"

RERANK_PROMPT = '''You are an art curator evaluating how well each artwork matches a visitor's search intent.

The visitor searched for: "{query}"

For each artwork below, rate how well it matches the COMPLETE intent of the search query on a scale of 0-100. Consider not just individual keywords but the RELATIONSHIPS between concepts in the query. For example, "the quiet tension between humans and nature" means artworks showing humans AND nature TOGETHER with a sense of quiet tension — not just nature alone, not just humans alone. "feminine beauty" means artworks depicting or evoking feminine beauty — not just any artwork by a female artist.

Score guide:
* 90-100: Directly and powerfully embodies the full query intent
* 70-89: Strongly related, captures most of the query's meaning
* 50-69: Partially relevant, matches some aspects
* 30-49: Tangentially related
* 0-29: Not relevant

Artworks:
{artworks_text}

Respond with ONLY a JSON array, no other text: [{{"id": "artwork_id", "score": 85, "reason": "one sentence explanation"}}]'''


# ---------------------------------------------------------------------------
# Data (loaded at startup)
# ---------------------------------------------------------------------------
artworks = {}          # id -> full record
tags_dict = {}         # id -> [40 tags]
louvre_trans = {}       # louvre_id -> {title, medium, classification, description}
embeddings = None      # np.ndarray (14000, 3072)
index_list = []        # [{row, id, title, ...}, ...]
id_to_row = {}         # id -> row index

openai_client = None
anthropic_client = None

# LRU cache for cosine-sorted results (query -> [(art_id, cosine_score), ...])
cosine_cache = OrderedDict()
MAX_CACHE = 10


def get_or_compute_cosine(query, query_vec):
    """Return full cosine-sorted list, using cache if available."""
    if query in cosine_cache:
        cosine_cache.move_to_end(query)
        return cosine_cache[query]

    scores = cosine_scores(query_vec)
    ranked_indices = np.argsort(-scores)
    all_ranked = []
    for idx in ranked_indices:
        idx = int(idx)
        all_ranked.append((index_list[idx]["id"], float(scores[idx])))

    cosine_cache[query] = all_ranked
    if len(cosine_cache) > MAX_CACHE:
        cosine_cache.popitem(last=False)
    return all_ranked


def load_data():
    global artworks, tags_dict, louvre_trans, embeddings, index_list, id_to_row
    global openai_client, anthropic_client

    print("Loading data...", flush=True)

    with open(COMBINED_PATH, "r", encoding="utf-8") as f:
        combined = json.load(f)
    for rec in combined["records"]:
        artworks[rec["id"]] = rec
    print(f"  Artworks: {len(artworks)}", flush=True)

    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        tags_dict.update(json.load(f))
    print(f"  Tags: {len(tags_dict)}", flush=True)

    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        louvre_trans.update(json.load(f))
    print(f"  Louvre translations: {len(louvre_trans)}", flush=True)

    embeddings = np.load(str(EMBEDDINGS_PATH))['embeddings'].astype(np.float32)
    print(f"  Embeddings: {embeddings.shape}", flush=True)

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index_list.extend(json.load(f))
    for entry in index_list:
        id_to_row[entry["id"]] = entry["row"]
    print(f"  Index: {len(index_list)} entries", flush=True)

    # Normalize embeddings for fast cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings[:] = embeddings / norms

    openai_client = OpenAI(api_key=OPENAI_KEY)
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    print("Data loaded.", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_scores(query_vec):
    """Compute cosine similarity between query and all embeddings."""
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    return embeddings @ q_norm


def build_artwork_summary(rank, art_id):
    """Build summary string for re-ranking (from run_reranking.py)."""
    rec = artworks.get(art_id, {})
    llm_tags = tags_dict.get(art_id, [])

    # For Louvre, use English translations
    trans = louvre_trans.get(art_id)
    if trans:
        title = trans.get("title", "") or rec.get("title", "")
        medium = trans.get("medium", "") or rec.get("medium", "")
        classification = trans.get("classification", "") or rec.get("classification", "")
    else:
        title = rec.get("title", "")
        medium = rec.get("medium", "")
        classification = rec.get("classification", "")

    artist = rec.get("artist", "")
    date = rec.get("date", "")
    dept = rec.get("department", "")
    culture = rec.get("culture", "")

    parts = [f'{rank}. [ID: {art_id}] {title}']
    if artist and artist != "Unknown":
        parts.append(f"by {artist}")
    if date:
        parts.append(str(date))
    if dept:
        parts.append(f"Department: {dept}")
    if culture:
        parts.append(f"Culture: {culture}")
    if medium:
        parts.append(f"Medium: {medium}")
    if classification:
        parts.append(f"Classification: {classification}")
    if llm_tags:
        tag_str = ", ".join(llm_tags[:10])
        parts.append(f"Tags: {tag_str}")

    return ". ".join(parts)


def build_result(art_id, cosine_score, rerank_score=None, rerank_reason=None):
    """Build a result dict for the API response."""
    rec = artworks.get(art_id, {})
    return {
        "id": art_id,
        "title": rec.get("title", ""),
        "artist": rec.get("artist", ""),
        "culture": rec.get("culture", ""),
        "date": rec.get("date", ""),
        "classification": rec.get("classification", ""),
        "medium": rec.get("medium", ""),
        "department": rec.get("department", ""),
        "image_url": rec.get("image_url", ""),
        "source_url": rec.get("source_url", ""),
        "source_museum": rec.get("source_museum", ""),
        "source_museum_code": rec.get("source_museum_code", ""),
        "cosine_score": round(float(cosine_score), 6),
        "rerank_score": rerank_score,
        "rerank_reason": rerank_reason,
    }


# ---------------------------------------------------------------------------
# Query Analysis (Step 0)
# ---------------------------------------------------------------------------

QUERY_ANALYSIS_PROMPT = '''User searched: "{query}"

Analyze the search intent and return JSON only, no other text:
{{
  "intent": "specific" or "conceptual" or "hybrid",
  "keyword_filters": {{
    "artist": "..." or null,
    "title": "..." or null,
    "culture": "..." or null,
    "classification": "..." or null
  }},
  "conceptual_query": "...",
  "explanation": "..."
}}

Rules:
- "specific": user wants a specific artist, artwork, or object (e.g., "Monet", "Mona Lisa", "Bauhaus chair")
- "conceptual": user wants thematic/emotional/conceptual results (e.g., "motherhood", "loneliness")
- "hybrid": user wants both (e.g., "Japanese ceramics about nature")
- keyword_filters: extract specific names, cultures, or object types for exact matching. null if none needed.
- conceptual_query: rewrite the query to maximize semantic search effectiveness. Expand artist names into their style/themes/period. Expand object types into visual/cultural characteristics. For conceptual queries, enhance with related concepts and cross-cultural terms.
- explanation: one sentence describing how you interpreted the query, will be shown to the user.

Examples:
- "Monet" -> {{"intent":"specific","keyword_filters":{{"artist":"Monet","title":null,"culture":null,"classification":null}},"conceptual_query":"impressionist painting, light, color, water, plein-air, atmosphere, French landscape","explanation":"Searching for works by Claude Monet and exploring impressionist connections across collections"}}
- "motherhood" -> {{"intent":"conceptual","keyword_filters":{{"artist":null,"title":null,"culture":null,"classification":null}},"conceptual_query":"motherhood, maternal love, mother and child, nurturing, family bond, fertility, protective care, divine motherhood","explanation":"Exploring artworks depicting maternal love and the mother-child bond across cultures"}}
- "Japanese ceramics" -> {{"intent":"hybrid","keyword_filters":{{"artist":null,"title":null,"culture":"Japan","classification":"ceramics"}},"conceptual_query":"Japanese ceramic art, pottery, stoneware, raku, glaze, tea ceremony, wabi-sabi aesthetic","explanation":"Finding Japanese ceramic works and exploring related pottery traditions across cultures"}}'''


async def analyze_query(query: str) -> dict:
    """Use Claude to analyze search intent and rewrite query."""
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    try:
        response = await asyncio.to_thread(
            anthropic_client.messages.create,
            model=RERANK_MODEL,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
            if m:
                raw = m.group(1).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  Query analysis failed: {e}", flush=True)
        return {
            "intent": "conceptual",
            "keyword_filters": {"artist": None, "title": None, "culture": None, "classification": None},
            "conceptual_query": query,
            "explanation": f"Exploring artworks related to \"{query}\" across cultures",
        }


# ---------------------------------------------------------------------------
# Keyword Filtering (Step 1)
# ---------------------------------------------------------------------------

KEYWORD_WEIGHTS = {"artist": 100, "title": 80, "culture": 60, "classification": 40}


def keyword_filter(filters):
    """Exact-match filter on artist/title/culture/classification. Returns [(art_id, keyword_score)]."""
    if not filters or all(v is None for v in filters.values()):
        return []

    matches = []
    for art_id, rec in artworks.items():
        score = 0
        for field, value in filters.items():
            if value is None:
                continue
            val_lower = value.lower()

            field_val = (rec.get(field) or "").lower()
            # Also check Louvre translations
            trans_val = ""
            if art_id in louvre_trans:
                trans_val = (louvre_trans[art_id].get(field) or "").lower()

            if val_lower in field_val or val_lower in trans_val:
                score += KEYWORD_WEIGHTS.get(field, 20)

        if score > 0:
            matches.append((art_id, score))

    matches.sort(key=lambda x: -x[1])
    return matches[:20]


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

async def rerank_batch(query, candidates, batch_start, semaphore):
    """Re-rank a single batch via Claude."""
    summaries = []
    id_list = []
    for i, (art_id, cos_score) in enumerate(candidates):
        rank = batch_start + i + 1
        summary = build_artwork_summary(rank, art_id)
        summaries.append(summary)
        id_list.append(art_id)

    artworks_text = "\n".join(summaries)
    prompt = RERANK_PROMPT.format(query=query, artworks_text=artworks_text)

    async with semaphore:
        for attempt in range(RERANK_MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    anthropic_client.messages.create,
                    model=RERANK_MODEL,
                    max_tokens=RERANK_MAX_TOKENS,
                    temperature=RERANK_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()

                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3].strip()

                scores = json.loads(text)
                score_lookup = {}
                for item in scores:
                    score_lookup[item["id"]] = {
                        "score": item["score"],
                        "reason": item.get("reason", ""),
                    }

                results = []
                for art_id, cos_score in candidates:
                    if art_id in score_lookup:
                        results.append((
                            art_id, cos_score,
                            score_lookup[art_id]["score"],
                            score_lookup[art_id]["reason"],
                        ))
                    else:
                        results.append((art_id, cos_score, 0, "Missing from response"))

                return results

            except Exception as e:
                if attempt < RERANK_MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rerank retry {attempt+1}: {e}. Waiting {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    print(f"  Rerank FAILED: {e}", flush=True)
                    return [
                        (art_id, cos_score, 0, f"API failure: {e}")
                        for art_id, cos_score in candidates
                    ]


async def rerank_candidates(query, top_candidates):
    """Re-rank top N candidates in batches with concurrency."""
    semaphore = asyncio.Semaphore(RERANK_CONCURRENCY)

    batches = []
    for b_start in range(0, len(top_candidates), RERANK_BATCH_SIZE):
        batch = top_candidates[b_start:b_start + RERANK_BATCH_SIZE]
        batches.append((b_start, batch))

    tasks = [
        rerank_batch(query, batch, b_start, semaphore)
        for b_start, batch in batches
    ]
    batch_results = await asyncio.gather(*tasks)

    merged = []
    for br in batch_results:
        merged.extend(br)

    # Sort by rerank score descending
    merged.sort(key=lambda x: -x[2])
    return merged


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MVP2V2 Museum Semantic Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class SearchMoreRequest(BaseModel):
    query: str
    offset: int = 300
    count: int = 100


@app.on_event("startup")
async def startup():
    load_data()


@app.post("/search")
async def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        return {"query": "", "reranked_count": 0, "total_count": 0, "results": []}

    t0 = time.time()

    # Step 0: Claude query analysis
    analysis = await analyze_query(query)
    intent = analysis.get("intent", "conceptual")
    kw_filters = analysis.get("keyword_filters", {})
    conceptual_query = analysis.get("conceptual_query", query)
    explanation = analysis.get("explanation", "")

    print(f"  Intent: {intent}, conceptual_query: '{conceptual_query[:60]}...'", flush=True)

    # Step 1: Keyword filtering
    exact_matches = keyword_filter(kw_filters)
    exact_match_ids = set(aid for aid, _ in exact_matches)
    if exact_matches:
        print(f"  Exact matches: {len(exact_matches)}", flush=True)

    # Step 2: Cosine similarity using conceptual_query (not raw query)
    embed_resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[conceptual_query],
    )
    query_vec = np.array(embed_resp.data[0].embedding, dtype=np.float32)

    # Cache under conceptual_query (for /search/more)
    all_ranked = get_or_compute_cosine(conceptual_query, query_vec)
    # Also cache a mapping from raw query to conceptual_query
    cosine_cache[f"__cq__{query}"] = conceptual_query

    # Take cosine top N, excluding exact matches (they're already in the pool)
    cosine_candidates = []
    for art_id, cos_score in all_ranked:
        if art_id not in exact_match_ids:
            cosine_candidates.append((art_id, cos_score))
            if len(cosine_candidates) >= RERANK_TOP_N:
                break

    # Merge: exact matches (with their cosine scores) + cosine candidates
    # Look up cosine scores for exact matches
    cosine_lookup = {aid: cs for aid, cs in all_ranked[:RERANK_TOP_N * 2]}
    merged_candidates = []
    for aid, kw_score in exact_matches:
        cos = cosine_lookup.get(aid, 0.0)
        merged_candidates.append((aid, cos))
    merged_candidates.extend(cosine_candidates)

    # Step 3: Re-rank using original query (not conceptual_query)
    reranked = await rerank_candidates(query, merged_candidates)

    # Step 4: Build results
    results = []
    for art_id, cos_score, rerank_score, rerank_reason in reranked:
        results.append(build_result(art_id, cos_score, rerank_score, rerank_reason))

    # Cosine zone (remainder, for completeness)
    reranked_ids = set(r["id"] for r in results)
    cosine_zone_start = RERANK_TOP_N + len(exact_matches)
    for art_id, cos_score in all_ranked[cosine_zone_start:]:
        if art_id not in reranked_ids:
            results.append(build_result(art_id, cos_score))

    elapsed = time.time() - t0
    print(f"Search '{query[:50]}' done in {elapsed:.1f}s "
          f"(intent={intent}, exact={len(exact_matches)}, "
          f"reranked={len(reranked)})", flush=True)

    return {
        "query": query,
        "intent": intent,
        "explanation": explanation,
        "exact_match_count": len(exact_matches),
        "reranked_count": len(reranked),
        "total_count": len(results),
        "has_more": True,
        "results": results,
    }


@app.post("/search/more")
async def search_more(req: SearchMoreRequest):
    query = req.query.strip()
    offset = req.offset
    count = req.count
    if not query:
        return {"query": "", "results": [], "has_more": False}

    t0 = time.time()

    # Resolve conceptual_query from cache (set during /search)
    cq_key = f"__cq__{query}"
    conceptual_query = cosine_cache.get(cq_key, query)

    # Get cached cosine results (keyed by conceptual_query), or recompute
    if conceptual_query in cosine_cache:
        cosine_cache.move_to_end(conceptual_query)
        all_ranked = cosine_cache[conceptual_query]
    else:
        embed_resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=[conceptual_query],
        )
        query_vec = np.array(embed_resp.data[0].embedding, dtype=np.float32)
        all_ranked = get_or_compute_cosine(conceptual_query, query_vec)

    # Slice the requested range
    end = min(offset + count, len(all_ranked))
    candidates = all_ranked[offset:end]

    if not candidates:
        return {"query": query, "results": [], "has_more": False}

    # Re-rank using original query
    reranked = await rerank_candidates(query, candidates)

    results = []
    for art_id, cos_score, rerank_score, rerank_reason in reranked:
        results.append(build_result(art_id, cos_score, rerank_score, rerank_reason))

    elapsed = time.time() - t0
    print(f"SearchMore '{query[:50]}' offset={offset} count={count} "
          f"done in {elapsed:.1f}s ({len(results)} reranked)", flush=True)

    return {
        "query": query,
        "results": results,
        "has_more": end < len(all_ranked),
        "next_offset": end,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "artworks": len(artworks),
        "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
    }


@app.get("/api/artwork/{artwork_id}")
async def get_artwork(artwork_id: str):
    rec = artworks.get(artwork_id)
    if rec is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "Not found"})
    result = dict(rec)
    result["original_tags"] = rec.get("tags", [])
    result["llm_tags"] = tags_dict.get(artwork_id, [])
    result["louvre_translation"] = louvre_trans.get(artwork_id, None)
    return result


@app.get("/api/artworks")
async def get_artworks():
    results = []
    for rec in artworks.values():
        results.append({
            "id": rec["id"],
            "title": rec.get("title", ""),
            "artist": rec.get("artist", ""),
            "date": rec.get("date", ""),
            "culture": rec.get("culture", ""),
            "medium": rec.get("medium", ""),
            "classification": rec.get("classification", ""),
            "image_url": rec.get("image_url", ""),
            "source_url": rec.get("source_url", ""),
            "source_museum": rec.get("source_museum", ""),
            "source_museum_code": rec.get("source_museum_code", ""),
        })
    return results


@app.get("/")
async def index():
    return FileResponse(str(BASE / "museum-search.html"))
