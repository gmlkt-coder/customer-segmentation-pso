from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

try:
    from .config import MAX_ITER, N_CLUSTERS, N_PARTICLES, RANDOM_STATE
    from .evaluation import compute_sse, evaluate_clustering
except ImportError:  # pragma: no cover
    from config import MAX_ITER, N_CLUSTERS, N_PARTICLES, RANDOM_STATE
    from evaluation import compute_sse, evaluate_clustering


@dataclass
class FitResult:
    labels: np.ndarray
    centers: np.ndarray
    history: List[float]


@dataclass
class ArchiveMember:
    position: np.ndarray
    sse: float
    silhouette: float


def pairwise_distances(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    if Y is None:
        Y = X
    X2 = np.sum(X * X, axis=1)[:, None]
    Y2 = np.sum(Y * Y, axis=1)[None, :]
    dist2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return np.sqrt(np.maximum(dist2, 0.0))


def assign_labels(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(data, centers)
    return np.argmin(distances, axis=1)


def sample_silhouette(data: np.ndarray, labels: np.ndarray, rng: np.random.Generator, max_points: int = 200) -> float:
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return -1.0
    if len(data) <= max_points:
        return float(silhouette_score(data, labels))
    idx = rng.choice(len(data), size=max_points, replace=False)
    sampled_labels = labels[idx]
    if len(np.unique(sampled_labels)) < 2:
        return -1.0
    return float(silhouette_score(data[idx], sampled_labels))


def init_centers_random(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(data), size=k, replace=False)
    return data[idx].copy()


def init_centers_kmeanspp(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [data[rng.integers(len(data))].copy()]
    for _ in range(1, k):
        dist = pairwise_distances(data, np.vstack(centers))
        closest = np.min(dist**2, axis=1)
        probs = closest / (closest.sum() + 1e-12)
        next_idx = rng.choice(len(data), p=probs)
        centers.append(data[next_idx].copy())
    return np.vstack(centers)


def refine_kmeans(
    data: np.ndarray,
    init_centers: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    km = KMeans(
        n_clusters=init_centers.shape[0],
        init=init_centers,
        n_init=1,
        max_iter=max_iter,
        tol=tol,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    labels = km.fit_predict(data)
    return labels, km.cluster_centers_, float(km.inertia_), [float(km.inertia_)]


class BaseClusterer:
    def __init__(self, n_clusters: int = N_CLUSTERS, max_iter: int = MAX_ITER, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)


class KMeansRandomClusterer(BaseClusterer):
    def fit_predict(self, data: np.ndarray) -> FitResult:
        init_centers = init_centers_random(data, self.n_clusters, self.rng)
        labels, centers, _, history = refine_kmeans(data, init_centers, self.rng)
        return FitResult(labels=labels, centers=centers, history=history)


class KMeansPlusPlusClusterer(BaseClusterer):
    def fit_predict(self, data: np.ndarray) -> FitResult:
        init_centers = init_centers_kmeanspp(data, self.n_clusters, self.rng)
        labels, centers, _, history = refine_kmeans(data, init_centers, self.rng)
        return FitResult(labels=labels, centers=centers, history=history)


class PSOKMeansClusterer(BaseClusterer):
    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        max_iter: int = MAX_ITER,
        n_particles: int = N_PARTICLES,
        random_state: int = RANDOM_STATE,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        self.n_particles = n_particles
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def _escape_boundary(self, particle: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        mask = (particle < low) | (particle > high)
        if mask.any():
            replacement = self.rng.uniform(low, high, size=particle.shape)
            particle = np.where(mask, replacement, particle)
        return particle

    def fit_predict(self, data: np.ndarray) -> FitResult:
        n_features = data.shape[1]
        shape = (self.n_clusters, n_features)
        particles = np.array([init_centers_random(data, self.n_clusters, self.rng) for _ in range(self.n_particles)])
        velocities = self.rng.normal(0, 0.1, size=particles.shape)
        pbest = particles.copy()
        pbest_scores = np.array([compute_sse(data, p) for p in particles])
        gbest = pbest[int(np.argmin(pbest_scores))].copy()
        gbest_score = float(np.min(pbest_scores))
        history = [gbest_score]
        low = data.min(axis=0)
        high = data.max(axis=0)

        for t in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * (t / max(1, self.max_iter - 1))
            for i in range(self.n_particles):
                r1 = self.rng.random(size=shape)
                r2 = self.rng.random(size=shape)
                velocities[i] = w * velocities[i] + self.c1 * r1 * (pbest[i] - particles[i]) + self.c2 * r2 * (
                    gbest - particles[i]
                )
                particles[i] = self._escape_boundary(particles[i] + velocities[i], low, high)
                score = compute_sse(data, particles[i])
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = particles[i].copy()
            best_idx = int(np.argmin(pbest_scores))
            if pbest_scores[best_idx] < gbest_score:
                gbest_score = float(pbest_scores[best_idx])
                gbest = pbest[best_idx].copy()
            history.append(gbest_score)

        labels, centers, _, refine_history = refine_kmeans(data, gbest, self.rng)
        history.extend(refine_history)
        return FitResult(labels=labels, centers=centers, history=history)


class IHPSOKMeansClusterer(PSOKMeansClusterer):
    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        max_iter: int = MAX_ITER,
        n_particles: int = N_PARTICLES,
        random_state: int = RANDOM_STATE,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        ga_start_ratio: float = 0.5,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.05,
    ):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_particles=n_particles,
            random_state=random_state,
            w_max=w_max,
            w_min=w_min,
            c1=c1,
            c2=c2,
        )
        self.ga_start_ratio = ga_start_ratio
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

    def _adaptive_inertia(self, t: int, a: float = 2.0, b: float = 1.0) -> float:
        ratio = t / max(1, self.max_iter - 1)
        return (self.w_max - self.w_min) * math.exp(-a * (ratio**b)) + self.w_min

    def _roulette_select(self, scores: np.ndarray, size: int) -> np.ndarray:
        weights = 1.0 / (scores + 1e-12)
        probs = weights / weights.sum()
        return self.rng.choice(len(scores), size=size, replace=True, p=probs)

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat1 = p1.reshape(-1)
        flat2 = p2.reshape(-1)
        if len(flat1) < 2:
            return p1.copy(), p2.copy()
        point = self.rng.integers(1, len(flat1))
        c1 = np.concatenate([flat1[:point], flat2[point:]]).reshape(p1.shape)
        c2 = np.concatenate([flat2[:point], flat1[point:]]).reshape(p1.shape)
        return c1, c2

    def _mutate(self, particle: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        mutated = particle.copy()
        mask = self.rng.random(size=mutated.shape) < self.mutation_rate
        if mask.any():
            mutated[mask] += self.rng.normal(0, self.mutation_scale, size=mask.sum())
        return np.clip(mutated, low, high)

    def fit_predict(self, data: np.ndarray) -> FitResult:
        n_features = data.shape[1]
        shape = (self.n_clusters, n_features)
        particles = np.array([init_centers_random(data, self.n_clusters, self.rng) for _ in range(self.n_particles)])
        velocities = self.rng.normal(0, 0.1, size=particles.shape)
        pbest = particles.copy()
        pbest_scores = np.array([compute_sse(data, p) for p in particles])
        gbest = pbest[int(np.argmin(pbest_scores))].copy()
        gbest_score = float(np.min(pbest_scores))
        history = [gbest_score]
        low = data.min(axis=0)
        high = data.max(axis=0)
        ga_start = int(self.max_iter * self.ga_start_ratio)

        for t in range(self.max_iter):
            w = self._adaptive_inertia(t)
            for i in range(self.n_particles):
                r1 = self.rng.random(size=shape)
                r2 = self.rng.random(size=shape)
                velocities[i] = w * velocities[i] + self.c1 * r1 * (pbest[i] - particles[i]) + self.c2 * r2 * (
                    gbest - particles[i]
                )
                particles[i] = np.clip(particles[i] + velocities[i], low, high)
                score = compute_sse(data, particles[i])
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = particles[i].copy()

            if t >= ga_start and self.n_particles >= 4:
                order = np.argsort(pbest_scores)
                elite_count = max(2, self.n_particles // 2)
                elite_idx = order[:elite_count]
                elite_scores = pbest_scores[elite_idx]
                chosen = self._roulette_select(elite_scores, elite_count)
                children: List[np.ndarray] = []
                for j in range(0, elite_count, 2):
                    p1 = pbest[elite_idx[chosen[j]]]
                    p2 = pbest[elite_idx[chosen[(j + 1) % elite_count]]]
                    c1, c2 = self._crossover(p1, p2)
                    children.append(self._mutate(c1, low, high))
                    if len(children) < elite_count:
                        children.append(self._mutate(c2, low, high))
                worst_idx = order[-elite_count:]
                for idx, child in zip(worst_idx, children[:elite_count]):
                    particles[idx] = child
                    pbest[idx] = child.copy()
                    pbest_scores[idx] = compute_sse(data, child)

            best_idx = int(np.argmin(pbest_scores))
            if pbest_scores[best_idx] < gbest_score:
                gbest_score = float(pbest_scores[best_idx])
                gbest = pbest[best_idx].copy()
            history.append(gbest_score)

        labels, centers, _, refine_history = refine_kmeans(data, gbest, self.rng)
        history.extend(refine_history)
        return FitResult(labels=labels, centers=centers, history=history)


class MOPSOKMeansClusterer(BaseClusterer):
    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        max_iter: int = MAX_ITER,
        n_particles: int = N_PARTICLES,
        random_state: int = RANDOM_STATE,
        w_ex: float = 0.9,
        w_en: float = 0.1,
        w_he: float = 0.03,
        c1: float = 2.0,
        c2: float = 2.0,
        archive_size: int = 25,
    ):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        self.n_particles = n_particles
        self.w_ex = w_ex
        self.w_en = w_en
        self.w_he = w_he
        self.c1 = c1
        self.c2 = c2
        self.archive_size = archive_size

    def _cloud_weight(self, t: int) -> float:
        ratio = t / max(1, self.max_iter - 1)
        ex = self.w_ex - 0.45 * ratio
        en_prime = abs(self.rng.normal(self.w_en, self.w_he))
        weight = self.rng.normal(ex, max(en_prime, 1e-3))
        return float(np.clip(weight, 0.2, 0.95))

    @staticmethod
    def _dominates(a: ArchiveMember, b: ArchiveMember) -> bool:
        return (a.sse <= b.sse and a.silhouette >= b.silhouette) and (
            a.sse < b.sse or a.silhouette > b.silhouette
        )

    def _update_archive(self, archive: List[ArchiveMember], candidates: List[ArchiveMember]) -> List[ArchiveMember]:
        pool = archive + candidates
        if not pool:
            return []

        non_dominated: List[ArchiveMember] = []
        for item in pool:
            dominated = False
            for other in pool:
                if other is item:
                    continue
                if self._dominates(other, item):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(item)

        unique: List[ArchiveMember] = []
        seen = set()
        for item in non_dominated:
            key = tuple(np.round(item.position.reshape(-1), 6))
            if key not in seen:
                unique.append(item)
                seen.add(key)

        if len(unique) <= self.archive_size:
            return unique

        order = np.argsort([m.sse - m.silhouette for m in unique])
        return [unique[i] for i in order[: self.archive_size]]

    def _choose_leader(self, archive: List[ArchiveMember]) -> np.ndarray:
        if not archive:
            raise ValueError("Empty archive.")
        scores = np.array([1.0 / (m.sse + 1e-12) + max(m.silhouette, 0.0) for m in archive], dtype=float)
        probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(archive)) / len(archive)
        idx = self.rng.choice(len(archive), p=probs)
        return archive[idx].position.copy()

    def _escape_boundary(self, particle: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        mask = (particle < low) | (particle > high)
        if mask.any():
            replacement = self.rng.uniform(low, high, size=particle.shape)
            particle = np.where(mask, replacement, particle)
        return particle

    def fit_predict(self, data: np.ndarray) -> FitResult:
        n_features = data.shape[1]
        shape = (self.n_clusters, n_features)
        particles = np.array([init_centers_random(data, self.n_clusters, self.rng) for _ in range(self.n_particles)])
        velocities = self.rng.normal(0, 0.1, size=particles.shape)
        pbest = particles.copy()
        pbest_members: List[ArchiveMember] = []
        for p in particles:
            labels = assign_labels(data, p)
            pbest_members.append(
                ArchiveMember(
                    position=p.copy(),
                    sse=compute_sse(data, p),
                    silhouette=sample_silhouette(data, labels, self.rng),
                )
            )

        archive = self._update_archive([], pbest_members)
        history = [min((m.sse for m in archive), default=np.inf)]
        low = data.min(axis=0)
        high = data.max(axis=0)

        for t in range(self.max_iter):
            weight = self._cloud_weight(t)
            for i in range(self.n_particles):
                leader = self._choose_leader(archive) if archive else pbest[i]
                r1 = self.rng.random(size=shape)
                r2 = self.rng.random(size=shape)
                velocities[i] = weight * velocities[i] + self.c1 * r1 * (pbest[i] - particles[i]) + self.c2 * r2 * (
                    leader - particles[i]
                )
                particles[i] = np.clip(self._escape_boundary(particles[i] + velocities[i], low, high), low, high)

            candidates: List[ArchiveMember] = []
            for i in range(self.n_particles):
                labels = assign_labels(data, particles[i])
                cand = ArchiveMember(
                    position=particles[i].copy(),
                    sse=compute_sse(data, particles[i]),
                    silhouette=sample_silhouette(data, labels, self.rng),
                )
                candidates.append(cand)
                if self._dominates(cand, pbest_members[i]):
                    pbest[i] = particles[i].copy()
                    pbest_members[i] = cand

            archive = self._update_archive(archive, candidates)
            history.append(min((m.sse for m in archive), default=history[-1]))

        best_member = max(archive, key=lambda m: (m.silhouette, -m.sse)) if archive else min(
            pbest_members, key=lambda m: m.sse
        )
        labels, centers, _, refine_history = refine_kmeans(data, best_member.position, self.rng)
        history.extend(refine_history)
        return FitResult(labels=labels, centers=centers, history=history)


def build_model(name: str, seed: int) -> BaseClusterer:
    if name == "KMeans-Random":
        return KMeansRandomClusterer(random_state=seed)
    if name == "KMeans++":
        return KMeansPlusPlusClusterer(random_state=seed)
    if name == "PSO-KM":
        return PSOKMeansClusterer(random_state=seed)
    if name == "IHPSO-KM":
        return IHPSOKMeansClusterer(random_state=seed)
    if name == "MOPSO-KM":
        return MOPSOKMeansClusterer(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def run_model(name: str, data: np.ndarray, seed: int) -> Tuple[Dict[str, float], FitResult]:
    model = build_model(name, seed)
    result = model.fit_predict(data)
    metrics = evaluate_clustering(result.labels, data)
    return metrics, result
