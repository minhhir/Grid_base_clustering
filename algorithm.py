import numpy as np
from collections import deque
import time

"""
Thuật toán CLIQUE (CLustering In QUEst) cho dữ liệu 2D.

Ý tưởng: chia không gian thành k×k ô (cell), xác định ô có mật độ
điểm >= xi (MinPts), rồi gom các ô dày đặc liền kề thành cụm bằng BFS.

Tham số:
    X  : mảng dữ liệu shape (n, 2)
    k  : số ô trên mỗi trục
    xi : ngưỡng mật độ tối thiểu (MinPts)
"""

class CLIQUEAlgorithm:
    """
    Thuật toán CLIQUE (CLustering In QUEst) cho dữ liệu 2D.

    Ý tưởng: chia không gian thành k×k ô (cell), xác định ô có mật độ
    điểm >= xi (MinPts), rồi gom các ô dày đặc liền kề thành cụm bằng BFS.

    Tham số:
        X  : mảng dữ liệu shape (n, 2)
        k  : số ô trên mỗi trục
        xi : ngưỡng mật độ tối thiểu (MinPts)
    """

    def __init__(self, X: np.ndarray, k: int, xi: int, log_callback=None):
        self.X   = X
        self.k   = k
        self.xi  = xi
        self.log = log_callback if log_callback else print

        self.labels_        = None
        self.n_clusters_    = 0
        self.dense_units_   = []
        self.cluster_cells_ = []
        self.density_grid_  = None
        self.x_edges_       = None
        self.y_edges_       = None

    def prepare_grid(self):
        self._build_grid()
        self._compute_density()
        return self
    #pipeline chính: build grid → compute density → connected components → assign labels
    def fit(self):
        self.log(f"[CLIQUE] n={len(self.X)}, k={self.k}, MinPts={self.xi}")
        t0 = time.time()
        self._build_grid()
        self._compute_density()
        if not self.dense_units_:
            self.log("Không có ô nào đạt ngưỡng mật độ!")
            self.labels_     = np.full(len(self.X), -1, dtype=int)
            self.n_clusters_ = 0
            return self
        self._connected_components()
        self._assign_labels()
        self.log(f"Kết quả: {self.n_clusters_} cụm | {(time.time()-t0)*1000:.1f} ms")
        return self

    def _build_grid(self):
        self.log("Khởi tạo không gian lưới...")
        t0 = time.time()
        x_min, x_max = self.X[:, 0].min(), self.X[:, 0].max()
        y_min, y_max = self.X[:, 1].min(), self.X[:, 1].max()
        pad_x = (x_max - x_min) * 0.01 + 1e-6
        pad_y = (y_max - y_min) * 0.01 + 1e-6
        self.x_edges_ = np.linspace(x_min - pad_x, x_max + pad_x, self.k + 1)
        self.y_edges_ = np.linspace(y_min - pad_y, y_max + pad_y, self.k + 1)
        dx, dy = self.x_edges_[1] - self.x_edges_[0], self.y_edges_[1] - self.y_edges_[0]
        self.log(f"Lưới {self.k}×{self.k} | Δx={dx:.3f}, Δy={dy:.3f} | {(time.time()-t0)*1000:.1f} ms")

    def _compute_density(self):
        self.log("Tính mật độ ô lưới...")
        t0 = time.time()
        H, _, _ = np.histogram2d(self.X[:, 0], self.X[:, 1],
                                  bins=(self.x_edges_, self.y_edges_))
        self.density_grid_ = H
        self.dense_units_  = [(i, j) for i in range(self.k)
                               for j in range(self.k) if H[i, j] >= self.xi]
        self.log(f"Ô có dữ liệu: {int(np.sum(H>0))} | "
                 f"Ô đạt ngưỡng: {len(self.dense_units_)} | {(time.time()-t0)*1000:.1f} ms")
    #hàm kết nối các ô dày đặc liền kề bằng BFS (Chebyshev dist = 1 → 8-connectivity)
    def _connected_components(self):
        self.log("BFS gom cụm...")
        t0 = time.time()
        dense_set = set(self.dense_units_)
        visited   = set()
        self.cluster_cells_ = []
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        for start in self.dense_units_:
            if start in visited:
                continue
            cluster, queue = set(), deque([start])
            visited.add(start)
            while queue:
                cell = queue.popleft()
                cluster.add(cell)
                ci, cj = cell
                for di, dj in dirs:
                    nb = (ci+di, cj+dj)
                    if nb in dense_set and nb not in visited:
                        visited.add(nb); queue.append(nb)
            self.cluster_cells_.append(cluster)
        self.n_clusters_ = len(self.cluster_cells_)
        self.log(f"Số cụm: {self.n_clusters_} | {(time.time()-t0)*1000:.1f} ms")
    # assign_labels: map điểm dữ liệu → cụm dựa trên ô chứa nó (nearest node)
    def _assign_labels(self):
        self.log("Gán nhãn điểm...")
        t0 = time.time()
        cluster_map = np.full((self.k, self.k), -1, dtype=int)
        for cid, cells in enumerate(self.cluster_cells_):
            for (ci, cj) in cells:
                cluster_map[ci, cj] = cid
        ix = np.clip(np.searchsorted(self.x_edges_[1:], self.X[:, 0]), 0, self.k-1)
        iy = np.clip(np.searchsorted(self.y_edges_[1:], self.X[:, 1]), 0, self.k-1)
        self.labels_ = cluster_map[ix, iy]
        noise = int(np.sum(self.labels_ == -1))
        self.log(f"Thuộc cụm: {len(self.X)-noise} | Noise: {noise} | {(time.time()-t0)*1000:.1f} ms")


"""
    GCBD – Grid-Based Clustering Using Boundary Detection.
    Nguồn: Du, M.; Wu, F. Entropy 2022, 24, 1606.
    https://doi.org/10.3390/e24111606

    Ba cải tiến chính so với CLIQUE:
      1. Mật độ tính tại NÚT (node = giao điểm lưới) bằng bilinear kernel,
         thay vì đếm điểm trong ô.
      2. Phát hiện biên lặp (iterative boundary detection, T vòng):
         mỗi vòng loại node dưới ngưỡng percentile-10 → tự động
         phân biệt core node và boundary node, không cần MinPts.
      3. Gán boundary node theo hàng xóm có mật độ cao nhất (DPC-inspired):
         giảm đáng kể noise giả, xử lý tốt cụm mật độ biến thiên & chồng lấp.

    Tham số:
        X : mảng dữ liệu shape (n, 2)
        l : số khoảng chia trên mỗi trục (tương đương k trong CLIQUE)
        T : số vòng lặp phát hiện biên (paper khuyến nghị 2–12)
    """


class GCBDAlgorithm:
    def __init__(self, X: np.ndarray, l: int, T: int, log_callback=None):
        self.X   = X
        self.l   = l
        self.T   = T
        self.log = log_callback if log_callback else print

        self.labels_        = None
        self.n_clusters_    = 0
        self.cluster_cells_ = []   # list of set of (i,j) node-index – core clusters

        # Thuộc tính tương thích với CLIQUE để GUI dùng chung
        self.density_grid_  = None   # mật độ node shape (l+1, l+1)
        self.x_edges_       = None   # toạ độ thực của các node (dùng vẽ grid lines)
        self.y_edges_       = None
        self.dense_units_   = []     # danh sách core node (i, j)

        # Nội bộ
        self._x_min = self._x_range = None
        self._y_min = self._y_range = None
    """ Public API: có thể gọi riêng từng bước nếu muốn xem intermediate result (mật độ node) mà không gán nhãn.
        self.prepare_grid() → xây lưới chuẩn + tính mật độ node (để xem trước)
    """
    def prepare_grid(self):
        self._build_standard_grid()
        self._compute_node_density(self.X)
        return self

    def fit(self):
        self.log(f"[GCBD] n={len(self.X)}, l={self.l}, T={self.T}")
        t_start = time.time()

        # Bước 1 – Lưới chuẩn + mật độ ban đầu
        self._build_standard_grid()
        self.log("Bước 1 – Tính mật độ node ban đầu...")
        rho = self._compute_node_density(self.X)

        active_pts   = np.ones(len(self.X), dtype=bool)
        active_nodes = rho > 0

        rho_snapshot = rho.copy()
        node_cluster = {}

        # Bước 2 – Phát hiện biên lặp
        self.log(f"Bước 2 – Phát hiện biên ({self.T} vòng)...")
        t2 = time.time()

        boundary_order = []

        # Mỗi vòng chỉ cần index vào mảng này, không gọi lại _scale + round
        nn_all = self._nearest_node(self.X)   # shape (n, 2), int32

        for t in range(1, self.T + 1):
            active_rho_vals = rho[active_nodes]
            if len(active_rho_vals) == 0:
                break
            tau = np.percentile(active_rho_vals, 10)

            remove_mask = active_nodes & (rho <= tau)
            removed = list(zip(*np.where(remove_mask))) if remove_mask.any() else []
            if not removed:
                self.log(f"  Vòng {t}: không có node nào bị loại – dừng sớm.")
                break

            boundary_order.extend(removed)

            # Snapshot mật độ của node sắp bị loại TRƯỚC KHI trừ điểm khỏi rho
            # → rho_snapshot[v] = mật độ tại thời điểm node v còn hoạt động
            rho_snapshot[remove_mask] = rho[remove_mask]
            active_nodes[remove_mask] = False

            #dùng mảng nearest node đã tính sẵn để tìm điểm nào bị ảnh hưởng bởi node bị loại
            deactivated = np.array([], dtype=int)
            if active_pts.any():
                ap_idx = np.where(active_pts)[0]
                ni_ap  = nn_all[ap_idx, 0]
                nj_ap  = nn_all[ap_idx, 1]
                keep_local = active_nodes[ni_ap, nj_ap]   # fully vectorized, no Python loop
                deactivated = ap_idx[~keep_local]
                active_pts[deactivated] = False

            # 0.1% điểm có thể bị ảnh hưởng mỗi vòng → O(n) vẫn nhanh hơn O((l+1)^2) khi l=100
            if len(deactivated) > 0:
                rho = self._subtract_density(rho, self.X[deactivated])

            # Zero inactive nodes để percentile vòng tiếp không bị nhiễm
            rho[~active_nodes] = 0.0

            self.log(f"  Vòng {t}: loại {len(removed)} node | "
                     f"còn {int(active_nodes.sum())} node, {int(active_pts.sum())} điểm")

        self.log(f"Phát hiện biên: {(time.time()-t2)*1000:.1f} ms")

        # Core nodes
        core_coords = list(zip(*np.where(active_nodes))) if active_nodes.any() else []
        self.dense_units_ = core_coords

        if not core_coords:
            self.log("Không có core node nào! Toàn bộ là noise.")
            self.labels_     = np.full(len(self.X), -1, dtype=int)
            self.n_clusters_ = 0
            return self

        # Bước 3 – Merge core nodes bằng BFS (Chebyshev dist = 1 → 8-connectivity)
        self.log("Bước 3 – Gom core nodes (BFS 8-connectivity)...")
        t3 = time.time()
        core_set = set(core_coords)
        visited  = set()
        self.cluster_cells_ = []
        dirs8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

        for start in core_coords:
            if start in visited:
                continue
            cluster, queue = set(), deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                cluster.add(node)
                ni, nj = node
                for di, dj in dirs8:
                    nb = (ni+di, nj+dj)
                    if nb in core_set and nb not in visited:
                        visited.add(nb); queue.append(nb)
            self.cluster_cells_.append(cluster)

        self.n_clusters_ = len(self.cluster_cells_)
        for cid, cells in enumerate(self.cluster_cells_):
            for node in cells:
                node_cluster[node] = cid
        self.log(f"Số cụm core: {self.n_clusters_} | {(time.time()-t3)*1000:.1f} ms")

        # Bước 4 – Gán boundary nodes (từ ngoài vào trong theo thứ tự bị loại ngược)
        self.log("Bước 4 – Gán boundary nodes (DPC-inspired)...")
        t4 = time.time()
        for v in reversed(boundary_order):
            vi, vj = v
            best_cid, best_rho = -1, -1.0
            for di, dj in dirs8:
                nb = (vi+di, vj+dj)
                if nb in node_cluster:
                    nb_rho = float(rho_snapshot[nb])
                    if nb_rho > best_rho:
                        best_rho, best_cid = nb_rho, node_cluster[nb]
            if best_cid >= 0:
                node_cluster[v] = best_cid
        self.log(f"Gán boundary: {(time.time()-t4)*1000:.1f} ms")

        # Bước 5 – Map điểm dữ liệu → cụm
        self.log("Bước 5 – Gán nhãn điểm dữ liệu...")
        nearest = self._nearest_node(self.X)
        self.labels_ = np.array([
            node_cluster.get((int(nn[0]), int(nn[1])), -1) for nn in nearest
        ], dtype=int)
        noise = int(np.sum(self.labels_ == -1))
        self.log(f"Thuộc cụm: {len(self.X)-noise} | Noise: {noise}")
        self.log(f"[GCBD] Hoàn thành: {self.n_clusters_} cụm | "
                 f"{(time.time()-t_start)*1000:.1f} ms tổng")
        return self

    """
    Lưới chuẩn (Definition 1 trong bài báo):
    scale dữ liệu về [1, l+1], node ở toạ độ nguyên.
    """
    def _build_standard_grid(self):
        """
        Lưới chuẩn (Definition 1 trong bài báo):
        scale dữ liệu về [1, l+1], node ở toạ độ nguyên.
        """
        self.log("Xây lưới chuẩn (standard grid)...")
        x_min, x_max = self.X[:, 0].min(), self.X[:, 0].max()
        y_min, y_max = self.X[:, 1].min(), self.X[:, 1].max()
        self._x_min   = x_min
        self._y_min   = y_min
        self._x_range = max(x_max - x_min, 1e-9)
        self._y_range = max(y_max - y_min, 1e-9)

        # Toạ độ thực của node (để vẽ và tính nhãn)
        self.x_edges_ = np.linspace(x_min, x_max, self.l + 1)
        self.y_edges_ = np.linspace(y_min, y_max, self.l + 1)
        self.log(f"Lưới {self.l}×{self.l} | {(self.l+1)**2} node giao")

    """
    Hàm scale Φ (Eq. 3):  Φ(xij) = l*(xij - min_j)/(max_j - min_j) + 1
    Kết quả ∈ [1, l+1].
    """
    def _scale(self, X: np.ndarray) -> np.ndarray:
        s = np.empty_like(X, dtype=np.float64)
        s[:, 0] = self.l * (X[:, 0] - self._x_min) / self._x_range + 1.0
        s[:, 1] = self.l * (X[:, 1] - self._y_min) / self._y_range + 1.0
        return np.clip(s, 1.0, self.l + 1.0 - 1e-9)

    """
    Mật độ node (Eq. 4–5):
        ρ_v = Σ_xi ∏_j max(1 - |Φ(xij) - vj|, 0)

    Mỗi điểm chỉ ảnh hưởng đến 4 node góc của ô chứa nó.
    Dùng np.bincount thay np.add.at → fully vectorized, nhanh ~10× với 1M điểm.
    """
    def _compute_node_density(self, X: np.ndarray, silent: bool = False) -> np.ndarray:
        if not silent:
            self.log("Tính mật độ node (bilinear kernel)...")
        t0 = time.time()
        n_nodes = (self.l + 1) ** 2

        if len(X) == 0:
            rho = np.zeros((self.l + 1, self.l + 1), dtype=np.float64)
            self.density_grid_ = rho
            return rho

        S  = self._scale(X)
        ix = np.clip((np.floor(S[:, 0]) - 1).astype(np.int32), 0, self.l - 1)
        iy = np.clip((np.floor(S[:, 1]) - 1).astype(np.int32), 0, self.l - 1)
        fx = (S[:, 0] - np.floor(S[:, 0])).astype(np.float32)
        fy = (S[:, 1] - np.floor(S[:, 1])).astype(np.float32)
        W  = self.l + 1   # stride theo chiều y

        # Flatten 2D index → 1D rồi dùng bincount (C-level, không có Python loop)
        rho_flat = (
            np.bincount(ix     * W + iy,     weights=(1-fx)*(1-fy), minlength=n_nodes) +
            np.bincount((ix+1) * W + iy,     weights=   fx *(1-fy), minlength=n_nodes) +
            np.bincount(ix     * W + (iy+1), weights=(1-fx)*   fy,  minlength=n_nodes) +
            np.bincount((ix+1) * W + (iy+1), weights=   fx *   fy,  minlength=n_nodes)
        )
        rho = rho_flat.reshape(self.l + 1, self.l + 1)
        self.density_grid_ = rho
        if not silent:
            self.log(f"Node ρ>0: {int(np.sum(rho>0))} / {n_nodes} | "
                     f"{(time.time()-t0)*1000:.1f} ms")
        return rho

    """
    Trừ đóng góp của các điểm X_removed khỏi mảng mật độ rho.
    Incremental update: O(len(X_removed)) thay vì O(n_total).
    """
    def _subtract_density(self, rho: np.ndarray, X_removed: np.ndarray) -> np.ndarray:
        if len(X_removed) == 0:
            return rho
        S  = self._scale(X_removed)
        ix = np.clip((np.floor(S[:, 0]) - 1).astype(np.int32), 0, self.l - 1)
        iy = np.clip((np.floor(S[:, 1]) - 1).astype(np.int32), 0, self.l - 1)
        fx = (S[:, 0] - np.floor(S[:, 0])).astype(np.float32)
        fy = (S[:, 1] - np.floor(S[:, 1])).astype(np.float32)
        W  = self.l + 1
        n_nodes = W * W
        rho_flat = rho.ravel()
        rho_flat -= (
            np.bincount(ix     * W + iy,     weights=(1-fx)*(1-fy), minlength=n_nodes) +
            np.bincount((ix+1) * W + iy,     weights=   fx *(1-fy), minlength=n_nodes) +
            np.bincount(ix     * W + (iy+1), weights=(1-fx)*   fy,  minlength=n_nodes) +
            np.bincount((ix+1) * W + (iy+1), weights=   fx *   fy,  minlength=n_nodes)
        )
        return np.maximum(rho_flat.reshape(W, W), 0.0)  # clip âm do float rounding

    """
    Node gần nhất cho mỗi điểm: làm tròn toạ độ đã scale → index 0-based.
    (Section 4.4.3: dùng hàm round thay vì tính khoảng cách – O(n))
    """
    def _nearest_node(self, X: np.ndarray) -> np.ndarray:
        S  = self._scale(X)
        ni = np.clip(np.round(S[:, 0]).astype(int) - 1, 0, self.l)
        nj = np.clip(np.round(S[:, 1]).astype(int) - 1, 0, self.l)
        return np.stack([ni, nj], axis=1)