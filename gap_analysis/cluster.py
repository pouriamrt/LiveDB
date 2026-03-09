"""Phase 3: Embedding-based clustering + LLM theme labeling."""

from __future__ import annotations

import asyncio
import json

import numpy as np
from hdbscan import HDBSCAN
from loguru import logger as log
from openai import AsyncOpenAI

from Config import config
from gap_analysis.models import PaperExtraction, ThemeCluster
from gap_analysis.prompts import CLUSTER_LABEL_SYSTEM, CLUSTER_LABEL_USER

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


async def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API."""
    resp = await _client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([d.embedding for d in resp.data])


async def _label_cluster(papers: list[PaperExtraction], model: str) -> dict[str, str]:
    """Ask LLM to name a cluster theme."""
    papers_text = "\n".join(
        f"- {p.paper.title}: {', '.join(p.claims[:3])}" for p in papers[:15]
    )
    try:
        resp = await _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLUSTER_LABEL_SYSTEM},
                {
                    "role": "user",
                    "content": CLUSTER_LABEL_USER.format(papers=papers_text),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.warning(f"Cluster labeling failed: {e}")
        return {
            "label": "Unlabeled cluster",
            "description": "Could not generate label",
        }


async def cluster_papers(
    extractions: list[PaperExtraction],
    min_cluster_size: int = 3,
    model: str | None = None,
) -> list[ThemeCluster]:
    """Cluster papers by embedding similarity and label each cluster."""
    model = model or config.MODEL_NAME

    if len(extractions) < min_cluster_size:
        # Too few papers -- put everything in one cluster
        label_info = await _label_cluster(extractions, model)
        return [
            ThemeCluster(
                cluster_id=0,
                label=label_info.get("label", "All papers"),
                description=label_info.get("description", ""),
                papers=extractions,
            )
        ]

    # Embed: title + top claims for each paper
    texts = [f"{e.paper.title}. {' '.join(e.claims[:3])}" for e in extractions]
    embeddings = await _embed_texts(texts)

    # Cluster
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    unique_labels = set(labels)

    log.info(
        f"HDBSCAN found {len(unique_labels - {-1})} clusters, "
        f"{(labels == -1).sum()} noise points"
    )

    # Group papers by cluster
    clusters_dict: dict[int, list[PaperExtraction]] = {}
    noise_papers: list[PaperExtraction] = []
    for ext, label in zip(extractions, labels):
        if label == -1:
            noise_papers.append(ext)
        else:
            clusters_dict.setdefault(label, []).append(ext)

    # Assign noise papers to nearest cluster
    if noise_papers and clusters_dict:
        centroids = {}
        for cid, papers in clusters_dict.items():
            indices = [i for i, lbl in enumerate(labels) if lbl == cid]
            centroids[cid] = embeddings[indices].mean(axis=0)

        for paper in noise_papers:
            idx = extractions.index(paper)
            emb = embeddings[idx]
            nearest = min(centroids, key=lambda c: np.linalg.norm(emb - centroids[c]))
            clusters_dict[nearest].append(paper)

    # Label each cluster via LLM
    label_tasks = [_label_cluster(papers, model) for papers in clusters_dict.values()]
    label_results = await asyncio.gather(*label_tasks)

    theme_clusters = []
    for (cid, papers), label_info in zip(clusters_dict.items(), label_results):
        indices = [i for i, lbl in enumerate(labels) if lbl == cid]
        centroid = embeddings[indices].mean(axis=0).tolist() if indices else []
        theme_clusters.append(
            ThemeCluster(
                cluster_id=cid,
                label=label_info.get("label", f"Cluster {cid}"),
                description=label_info.get("description", ""),
                papers=papers,
                centroid_embedding=centroid,
            )
        )

    log.info(f"Created {len(theme_clusters)} theme clusters")
    return theme_clusters
