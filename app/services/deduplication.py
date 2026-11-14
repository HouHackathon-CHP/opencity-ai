# SPDX-License-Identifier: Apache-2.0
"""Spatiotemporal Deduplication Engine: Semantic Embeddings + Haversine Distance + Time Decay."""

import math
from collections import Counter
from datetime import datetime

from app.core.config import DEVICE, TOKEN_PATTERN
from app.schemas.analysis import Candidate
from app.services.model_loader import get_ai_engine


def tokenize(text: str) -> list[str]:
    """Tokenize words for fallback similarity."""
    return [word for word in TOKEN_PATTERN.findall(text.casefold()) if len(word) > 1]


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS coordinates in meters."""
    r = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def parse_iso_time(ts_str: str | None) -> float | None:
    """Parse ISO 8601 timestamp string into epoch seconds."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def calculate_semantic_scores(text: str, candidates: list[Candidate]) -> list[float]:
    """Calculate cosine similarity using PhoBERT sentence embeddings (Head 3) or TF-IDF."""
    model, tokenizer, _, is_multitask = get_ai_engine()

    if is_multitask and model is not None and tokenizer is not None:
        try:
            import torch

            all_texts = [text] + [c.text for c in candidates]
            enc = tokenizer(all_texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(enc["input_ids"], enc["attention_mask"])
                embs = out["sentence_embeddings"]
                q_emb = embs[0:1]
                cand_embs = embs[1:]
                cos_sims = torch.mm(cand_embs, q_emb.t()).squeeze(-1).cpu().numpy()
                return [max(0.0, float(s)) for s in cos_sims]
        except Exception:
            pass

    # Fallback to TF-IDF cosine similarity
    documents = [set(tokenize(text)), *(set(tokenize(item.text)) for item in candidates)]
    frequencies = Counter(word for document in documents for word in document)
    count = len(documents)

    def vector(document: set[str]) -> dict[str, float]:
        return {word: math.log((count + 1) / (frequencies[word] + 1)) + 1 for word in document}

    query = vector(documents[0])
    query_norm = math.sqrt(sum(value * value for value in query.values()))
    scores = []
    for candidate, document in zip(candidates, documents[1:]):
        other = vector(document)
        other_norm = math.sqrt(sum(value * value for value in other.values()))
        score = sum(query.get(word, 0) * other.get(word, 0) for word in document)
        score = score / (query_norm * other_norm) if query_norm and other_norm else 0.0
        scores.append(float(score))
    return scores


def find_similar_reports(
    text: str,
    candidates: list[Candidate],
    query_lat: float | None = None,
    query_lon: float | None = None,
    query_time: str | None = None,
    threshold: float = 0.35,
    max_results: int = 3,
) -> list[dict]:
    """Find duplicate / similar civic reports using spatiotemporal fusion."""
    if not candidates:
        return []

    semantic_scores = calculate_semantic_scores(text, candidates)
    q_time_sec = parse_iso_time(query_time)
    matches = []

    for idx, candidate in enumerate(candidates):
        s_sem = semantic_scores[idx] if idx < len(semantic_scores) else 0.0

        # 1. Geographic factor
        has_geo = (
            query_lat is not None
            and query_lon is not None
            and candidate.latitude is not None
            and candidate.longitude is not None
        )
        s_geo = 1.0
        geo_mismatch = False
        if has_geo:
            dist_m = haversine_distance_meters(query_lat, query_lon, candidate.latitude, candidate.longitude)
            if dist_m <= 50.0:
                s_geo = 1.0
            elif dist_m <= 1000.0:
                s_geo = max(0.0, 1.0 - (dist_m - 50.0) / 950.0)
            else:
                s_geo = 0.0

            if dist_m > 2000.0:
                geo_mismatch = True

        # 2. Temporal factor
        cand_time_sec = parse_iso_time(candidate.created_at)
        has_time = q_time_sec is not None and cand_time_sec is not None
        s_time = 1.0
        if has_time:
            delta_hours = abs(q_time_sec - cand_time_sec) / 3600.0
            if delta_hours <= 24.0:
                s_time = 1.0
            elif delta_hours <= 168.0:
                s_time = max(0.0, 1.0 - (delta_hours - 24.0) / 144.0)
            else:
                s_time = 0.0

        # 3. Composite Fusion Score
        if geo_mismatch:
            final_score = s_sem * 0.20
        elif has_geo and has_time:
            final_score = 0.50 * s_sem + 0.35 * s_geo + 0.15 * s_time
        elif has_geo:
            final_score = 0.60 * s_sem + 0.40 * s_geo
        elif has_time:
            final_score = 0.80 * s_sem + 0.20 * s_time
        else:
            final_score = s_sem

        if final_score >= threshold:
            matches.append({"id": candidate.id, "similarity": round(final_score, 4)})

    return sorted(matches, key=lambda item: (-item["similarity"], item["id"]))[:max_results]
