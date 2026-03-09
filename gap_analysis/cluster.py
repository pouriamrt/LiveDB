"""Phase 3: Embedding-based clustering + LLM theme labeling."""

from __future__ import annotations

import asyncio
import json

import numpy as np
from hdbscan import HDBSCAN
from loguru import logger as log
from umap import UMAP

from Config import config
from gap_analysis import openai_client as _client
from gap_analysis.models import PaperExtraction, ThemeCluster
from gap_analysis.prompts import CLUSTER_LABEL_SYSTEM, CLUSTER_LABEL_USER


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

    # Reduce dimensionality with UMAP before clustering (critical for small datasets
    # in high-dimensional embedding space — HDBSCAN density estimates fail otherwise)
    n_samples = len(extractions)
    n_components = min(10, n_samples - 2)  # UMAP needs n_components < n_samples
    n_neighbors = min(15, n_samples - 1)  # UMAP default is 15
    reduced = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    # Cluster
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})

    log.info(
        f"HDBSCAN found {n_clusters} clusters, {(labels == -1).sum()} noise points"
    )

    # Fallback: if HDBSCAN finds no clusters, put all papers in one group
    if n_clusters == 0:
        log.warning("No clusters found — grouping all papers into a single theme")
        label_info = await _label_cluster(extractions, model)
        return [
            ThemeCluster(
                cluster_id=0,
                label=label_info.get("label", "All papers"),
                description=label_info.get("description", ""),
                papers=extractions,
            )
        ]

    # Group papers by cluster
    clusters_dict: dict[int, list[PaperExtraction]] = {}
    noise_papers: list[PaperExtraction] = []
    for ext, label in zip(extractions, labels):
        if label == -1:
            noise_papers.append(ext)
        else:
            clusters_dict.setdefault(label, []).append(ext)

    # Assign noise papers to nearest cluster (using reduced embeddings)
    if noise_papers and clusters_dict:
        # Pre-build lookup: id(extraction) -> index for O(1) access
        ext_index = {id(ext): i for i, ext in enumerate(extractions)}

        centroids = {}
        for cid, papers in clusters_dict.items():
            indices = [i for i, lbl in enumerate(labels) if lbl == cid]
            centroids[cid] = reduced[indices].mean(axis=0)

        for paper in noise_papers:
            idx = ext_index[id(paper)]
            emb = reduced[idx]
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
