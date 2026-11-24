import sys
from pathlib import Path
import numpy as np
from PIL import Image


def load_frames(frames_dir: Path):
    paths = sorted(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"],
        key=lambda p: p.name,
    )
    if not paths:
        return [], np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    grid_feats = []
    hist_feats = []

    for p in paths:
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            small_rgb = np.asarray(rgb.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            gray = np.asarray(rgb.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
            grad = np.hypot(gx, gy)

            spatial_vec = np.concatenate([small_rgb.ravel(), grad.ravel()])
            grid_feats.append(spatial_vec)

            hist_r = np.histogram(small_rgb[:, :, 0], bins=8, range=(0.0, 1.0))[0]
            hist_g = np.histogram(small_rgb[:, :, 1], bins=8, range=(0.0, 1.0))[0]
            hist_b = np.histogram(small_rgb[:, :, 2], bins=8, range=(0.0, 1.0))[0]
            hist_vec = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
            norm = np.linalg.norm(hist_vec)
            if norm > 0:
                hist_vec /= norm
            hist_feats.append(hist_vec)

    names = [p.name for p in paths]
    return names, np.vstack(grid_feats), np.vstack(hist_feats)


def build_cost_matrix(grid_feats: np.ndarray, hist_feats: np.ndarray) -> np.ndarray:
    n = len(grid_feats)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    diff_grid = grid_feats[:, None, :] - grid_feats[None, :, :]
    grid_dist = np.mean(np.abs(diff_grid), axis=2)

    diff_hist = hist_feats[:, None, :] - hist_feats[None, :, :]
    hist_dist = np.linalg.norm(diff_hist, axis=2)

    g_max = grid_dist.max()
    h_max = hist_dist.max()
    g_norm = grid_dist / g_max if g_max > 0 else grid_dist
    h_norm = hist_dist / h_max if h_max > 0 else hist_dist

    cost = 0.8 * g_norm + 0.2 * h_norm
    np.fill_diagonal(cost, 1e9)
    return cost


def path_cost(order, cost: np.ndarray) -> float:
    if len(order) <= 1:
        return 0.0
    idx_a = np.array(order[:-1], dtype=np.int64)
    idx_b = np.array(order[1:], dtype=np.int64)
    return float(np.sum(cost[idx_a, idx_b]))


def nearest_neighbor_path(start: int, cost: np.ndarray):
    n = cost.shape[0]
    visited = [False] * n
    order = [start]
    visited[start] = True

    for _ in range(n - 1):
        curr = order[-1]
        best_next = -1
        best_val = float("inf")
        for nxt in range(n):
            if not visited[nxt] and cost[curr, nxt] < best_val:
                best_val = cost[curr, nxt]
                best_next = nxt
        order.append(best_next)
        visited[best_next] = True

    return order


def two_opt_open_path(order, cost: np.ndarray):
    n = len(order)
    if n < 4:
        return order

    best = list(order)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            a = best[i]
            b = best[i + 1]
            cost_ab = cost[a, b]
            for j in range(i + 2, n):
                c = best[j]
                old_edges = cost_ab
                new_edges = cost[a, c]
                if j + 1 < n:
                    d = best[j + 1]
                    old_edges += cost[c, d]
                    new_edges += cost[b, d]

                if new_edges + 1e-12 < old_edges:
                    best[i + 1 : j + 1] = reversed(best[i + 1 : j + 1])
                    improved = True
                    break
            if improved:
                break

    return best


def solve_order(names, grid_feats: np.ndarray, hist_feats: np.ndarray):
    n = len(names)
    if n == 0:
        return []
    if n == 1:
        return names

    cost = build_cost_matrix(grid_feats, hist_feats)
    starts = range(n) if n <= 120 else np.linspace(0, n - 1, 120, dtype=int).tolist()

    best_order = None
    best_score = float("inf")

    for s in starts:
        cand = nearest_neighbor_path(int(s), cost)
        cand = two_opt_open_path(cand, cost)
        score = path_cost(cand, cost)
        if score < best_score:
            best_score = score
            best_order = cand

    ordered_names = [names[i] for i in best_order]
    assert len(ordered_names) == n and len(set(ordered_names)) == n, "Invalid frame permutation"
    return ordered_names


def main():
    frames_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/frames")
    if not frames_dir.exists() or not frames_dir.is_dir():
        sys.exit(1)

    names, grid_feats, hist_feats = load_frames(frames_dir)
    for name in solve_order(names, grid_feats, hist_feats):
        print(name)


if __name__ == "__main__":
    main()
