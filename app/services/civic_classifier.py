# SPDX-License-Identifier: Apache-2.0
"""Civic classification and urgency prediction service."""

import math
from collections import Counter
from typing import Any

from app.core.config import DEVICE, TOKEN_PATTERN
from app.schemas.analysis import Example
from app.services.model_loader import get_ai_engine


def tokenize(text: str) -> list[str]:
    """Extract word tokens from text."""
    return [word for word in TOKEN_PATTERN.findall(text.casefold()) if len(word) > 1]


def naive_bayes_classify(text: str, labels: list[str], examples: list[Example]) -> tuple[list[dict], list[str]]:
    """Dynamic Naive Bayes fallback for custom labels or when neural weights are absent."""
    vocabulary = set()
    word_counts = {label: Counter() for label in labels}
    document_counts = Counter()
    for example in examples:
        words = tokenize(example.text)
        word_counts[example.label].update(words)
        document_counts[example.label] += 1
        vocabulary.update(words)

    query = Counter(word for word in tokenize(text) if word in vocabulary)
    if not query or not vocabulary:
        return ([{"label": label, "score": round(1.0 / len(labels), 4)} for label in labels], [])

    total_docs = sum(document_counts.values())
    log_scores = {}
    for label in labels:
        counts = word_counts[label]
        denominator = sum(counts.values()) + len(vocabulary)
        score = math.log((document_counts[label] + 1) / (total_docs + len(labels)))
        for word, frequency in query.items():
            score += frequency * math.log((counts[word] + 1) / denominator)
        log_scores[label] = score

    offset = max(log_scores.values())
    weights = {label: math.exp(value - offset) for label, value in log_scores.items()}
    total = sum(weights.values())
    ranked = sorted(
        [{"label": label, "score": round(weights[label] / total, 4)} for label in labels],
        key=lambda item: (-item["score"], item["label"]),
    )

    winner = ranked[0]["label"]
    signals = sorted(
        (word for word in query if word_counts[winner][word] > 0),
        key=lambda word: (-word_counts[winner][word], word),
    )[:6]
    return ranked, signals


def classify_text(text: str, labels: list[str], examples: list[Example]) -> tuple[list[dict], list[str]]:
    """Classify text into requested labels using PhoBERT Multi-Task (with Naive Bayes fallback)."""
    model, tokenizer, meta, is_multitask = get_ai_engine()

    civic_units = {"infrastructure", "environment", "transport", "intake"}
    label12_to_unit4 = meta.get("label12_to_unit4", {})
    id2cat = {int(k): v for k, v in meta.get("id2cat", {}).items()}
    id2urg = {int(k): v for k, v in meta.get("id2urg", {}).items()}
    classes_12 = meta.get("classes_12", [])

    is_civic = all(l in civic_units or l in label12_to_unit4 for l in labels)
    is_urgency = set(labels) == {"high", "normal", "low"}

    ranked = None

    # Task Head 2: Urgency Classification
    if is_urgency and is_multitask and model is not None and tokenizer is not None:
        try:
            import torch

            enc = tokenizer(text, truncation=True, max_length=96, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(enc["input_ids"], enc["attention_mask"])
                urg_probs = torch.softmax(out["urgency_logits"], dim=-1).cpu().numpy()[0]

            scores = {u: float(urg_probs[idx]) for idx, u in id2urg.items()}
            total_score = sum(scores.get(l, 0.0) for l in labels) or 1.0
            ranked = sorted(
                [{"label": l, "score": round(scores.get(l, 0.0) / total_score, 4)} for l in labels],
                key=lambda item: (-item["score"], item["label"]),
            )
        except Exception:
            ranked = None

    # Task Head 1: Civic Category & Municipal Unit Routing
    elif is_civic and is_multitask and model is not None and tokenizer is not None:
        try:
            import torch

            enc = tokenizer(text, truncation=True, max_length=96, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(enc["input_ids"], enc["attention_mask"])
                cat_probs = torch.softmax(out["category_logits"], dim=-1).cpu().numpy()[0]

            scores = {}
            for l in labels:
                if l in civic_units:
                    scores[l] = sum(
                        float(cat_probs[idx]) for idx, c12 in id2cat.items() if label12_to_unit4.get(c12) == l
                    )
                elif l in label12_to_unit4:
                    idx = classes_12.index(l)
                    scores[l] = float(cat_probs[idx])
                else:
                    scores[l] = 0.0

            total_score = sum(scores.values()) or 1.0
            ranked = sorted(
                [{"label": l, "score": round(scores[l] / total_score, 4)} for l in labels],
                key=lambda item: (-item["score"], item["label"]),
            )
        except Exception:
            ranked = None

    # Fallback to dynamic Naive Bayes
    if ranked is None:
        return naive_bayes_classify(text, labels, examples)

    tokens_query = Counter(tokenize(text))
    signals = [w for w, _ in tokens_query.most_common(5) if len(w) > 2]
    return ranked, signals
