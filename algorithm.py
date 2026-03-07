import numpy as np
from collections import deque
import time

# lớp CLIQUEAlgorithm được thiết kế để thực hiện phân cụm theo thuật toán CLIQUE trên dữ liệu 2D.
class CLIQUEAlgorithm:
    #hàm khởi tạo nhận vào dữ liệu X, số lượng ô k, ngưỡng mật độ xi và một callback để log (mặc định là print).
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

# hàm _build_grid tạo ra lưới 2D dựa trên phạm vi dữ liệu và số lượng ô k. Nó tính toán biên giới của lưới và kích thước mỗi ô, đồng thời log thông tin chi tiết về quá trình này.
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
        self.log(f"Kích thước lưới: {self.k} × {self.k} = {self.k ** 2} ô")
        self.log(f"Kích thước mỗi ô: Δx={dx:.3f}, Δy={dy:.3f}")
        self.log(f"Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

#hàm _compute_density sử dụng np.histogram2d để đếm số điểm trong mỗi ô lưới, sau đó xác định những ô nào đạt ngưỡng mật độ xi. Nó cũng log số lượng ô có dữ liệu và số ô đạt ngưỡng.
    def _compute_density(self):
        self.log("Tính mật độ từng ô lưới...")
        t0 = time.time()

        H, _, _ = np.histogram2d(self.X[:, 0], self.X[:, 1], bins=(self.x_edges_, self.y_edges_))
        self.density_grid_ = H

        non_empty = int(np.sum(H > 0))
        self.dense_units_ = [(i, j) for i in range(self.k) for j in range(self.k) if H[i, j] >= self.xi]

        self.log(f"Ô có dữ liệu: {non_empty}")
        self.log(f"Số ô đạt ngưỡng mật độ (MinPts={self.xi}): {len(self.dense_units_)}")
        self.log(f"Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

#hàm _connected_components thực hiện tìm kiếm theo chiều rộng (BFS) để gom nhóm các ô dày đặc liền kề thành các cụm. Nó sử dụng một hàng đợi để duyệt qua các ô và một tập hợp để theo dõi những ô đã được thăm. Kết quả là một danh sách các cụm, mỗi cụm là một tập hợp các ô, và số lượng cụm được log ra.
    def _connected_components(self):
        self.log("Gom nhóm ô dày đặc liền kề (BFS)...")
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
        self.log(f"Số cụm tìm được: {self.n_clusters_}")
        self.log(f"Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

#hàm _assign_labels gán nhãn cụm cho từng điểm dữ liệu dựa trên vị trí của chúng trong lưới. Nó tạo một bản đồ từ ô lưới sang nhãn cụm, sau đó sử dụng np.searchsorted để xác định ô chứa mỗi điểm và gán nhãn tương ứng. Điểm nào không thuộc ô nào sẽ được gán nhãn -1 (nhiễu). Cuối cùng, số lượng điểm thuộc cụm và điểm nhiễu được log ra.
    def _assign_labels(self):
        self.log("Gán nhãn cụm (Vectorized)...")
        t0 = time.time()

        cluster_map = np.full((self.k, self.k), -1, dtype=int)
        for cid, cells in enumerate(self.cluster_cells_):
            for (ci, cj) in cells:
                cluster_map[ci, cj] = cid

        ix = np.clip(np.searchsorted(self.x_edges_[1:], self.X[:, 0]), 0, self.k - 1)
        iy = np.clip(np.searchsorted(self.y_edges_[1:], self.X[:, 1]), 0, self.k - 1)

        self.labels_ = cluster_map[ix, iy]

        noise_count = int(np.sum(self.labels_ == -1))
        self.log(f"Điểm thuộc cụm: {len(self.X) - noise_count}")
        self.log(f"Điểm nhiễu (noise): {noise_count}")
        self.log(f"Hoàn thành trong {(time.time() - t0) * 1000:.1f} ms")

# hàm fit là hàm chính để chạy toàn bộ quá trình phân cụm. Nó log thông tin về quá trình chạy, gọi các hàm phụ để xây dựng lưới, tính mật độ, tìm kiếm các thành phần liên thông và gán nhãn. Cuối cùng, nó log kết quả cuối cùng và thời gian thực thi.
    def fit(self):
        self.log(f"Chạy CLiQUe (n={len(self.X)}, k={self.k}, MinPts={self.xi})")
        t_start = time.time()

        self._build_grid()
        self._compute_density()

        if not self.dense_units_:
            self.log("\nKhông có ô nào đạt ngưỡng mật độ!")
            self.labels_ = np.full(len(self.X), -1, dtype=int)
            self.n_clusters_ = 0
            return self

        self._connected_components()
        self._assign_labels()
        self.log(f"Kết quả: {self.n_clusters_} cụm. Thời gian: {(time.time() - t_start) * 1000:.1f} ms")
        return self