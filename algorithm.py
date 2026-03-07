import numpy as np
from collections import deque
import time


class CLIQUEAlgorithm:
    """
    Cài đặt thuật toán CLIQUE (Grid-based Clustering).
    """

    def __init__(self, X: np.ndarray, k: int, xi: int, log_callback=None):
        self.X = X
        self.k = k
        self.xi = xi
        self.log = log_callback if log_callback else print

        self.labels_ = None
        self.n_clusters_ = 0
        self.dense_units_ = []
        self.cluster_cells_ = []
        self.density_grid_ = None
        self.x_edges_ = None
        self.y_edges_ = None

    def _build_grid(self):
        self.log("━" * 55)
        self.log("[Bước 1] Khởi tạo không gian lưới...")
        t0 = time.time()

        x_min, x_max = self.X[:, 0].min(), self.X[:, 0].max()
        y_min, y_max = self.X[:, 1].min(), self.X[:, 1].max()

        pad_x = (x_max - x_min) * 0.01 + 1e-6
        pad_y = (y_max - y_min) * 0.01 + 1e-6

        self.x_edges_ = np.linspace(x_min - pad_x, x_max + pad_x, self.k + 1)
        self.y_edges_ = np.linspace(y_min - pad_y, y_max + pad_y, self.k + 1)

        dx, dy = self.x_edges_[1] - self.x_edges_[0], self.y_edges_[1] - self.y_edges_[0]
        self.log(f"   Kích thước lưới: {self.k} × {self.k} = {self.k ** 2} ô")
        self.log(f"   Kích thước mỗi ô: Δx={dx:.3f}, Δy={dy:.3f}")
        self.log(f"   ✓ Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

    def _compute_density(self):
        self.log("━" * 55)
        self.log("[Bước 2] Tính mật độ từng ô lưới...")
        t0 = time.time()

        H, _, _ = np.histogram2d(self.X[:, 0], self.X[:, 1], bins=(self.x_edges_, self.y_edges_))
        self.density_grid_ = H

        non_empty = int(np.sum(H > 0))
        self.dense_units_ = [(i, j) for i in range(self.k) for j in range(self.k) if H[i, j] >= self.xi]

        self.log(f"   Ô có dữ liệu: {non_empty}")
        self.log(f"   → Số ô đạt ngưỡng mật độ (MinPts={self.xi}): {len(self.dense_units_)}")
        self.log(f"   ✓ Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

    def _connected_components(self):
        self.log("━" * 55)
        self.log("[Bước 3] Gom nhóm ô dày đặc liền kề (BFS)...")
        t0 = time.time()

        dense_set = set(self.dense_units_)
        visited = set()
        self.cluster_cells_ = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for start_cell in self.dense_units_:
            if start_cell in visited:
                continue
            cluster = set()
            queue = deque([start_cell])
            visited.add(start_cell)

            while queue:
                cell = queue.popleft()
                cluster.add(cell)
                ci, cj = cell
                for di, dj in directions:
                    neighbor = (ci + di, cj + dj)
                    if neighbor in dense_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            self.cluster_cells_.append(cluster)

        self.n_clusters_ = len(self.cluster_cells_)
        self.log(f"   → Số cụm tìm được: {self.n_clusters_}")
        self.log(f"   ✓ Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

    def _assign_labels(self):
        self.log("━" * 55)
        self.log("[Bước 4] Gán nhãn cụm cho dữ liệu...")
        t0 = time.time()

        cell_to_cluster = {cell: cid for cid, cells in enumerate(self.cluster_cells_) for cell in cells}

        ix = np.clip(np.searchsorted(self.x_edges_[1:], self.X[:, 0]), 0, self.k - 1)
        iy = np.clip(np.searchsorted(self.y_edges_[1:], self.X[:, 1]), 0, self.k - 1)

        self.labels_ = np.full(len(self.X), -1, dtype=int)
        for idx in range(len(self.X)):
            cell = (ix[idx], iy[idx])
            if cell in cell_to_cluster:
                self.labels_[idx] = cell_to_cluster[cell]

        noise_count = int(np.sum(self.labels_ == -1))
        self.log(f"   → Điểm thuộc cụm: {len(self.X) - noise_count}")
        self.log(f"   → Điểm nhiễu (noise): {noise_count}")
        self.log(f"   ✓ Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

    def fit(self):
        self.log("═" * 55)
        self.log(f"  BẮT ĐẦU CHẠY CLIQUE (n={len(self.X)}, k={self.k}, MinPts={self.xi})")
        self.log("═" * 55)
        t_start = time.time()

        self._build_grid()
        self._compute_density()

        if not self.dense_units_:
            self.log("\n⚠ Không có ô nào đạt ngưỡng mật độ!")
            self.labels_ = np.full(len(self.X), -1, dtype=int)
            self.n_clusters_ = 0
            return self

        self._connected_components()
        self._assign_labels()

        self.log("━" * 55)
        self.log(f"  ✔ KẾT QUẢ: {self.n_clusters_} cụm. Thời gian: {(time.time() - t_start) * 1000:.1f} ms")
        self.log("═" * 55)
        return self