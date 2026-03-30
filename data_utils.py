import numpy as np
import csv

def generate_mock_data(n_samples: int, n_centers: int, std: float) -> np.ndarray:
    rng = np.random.default_rng()
    centers = rng.uniform(-8, 8, size=(n_centers, 2))
    X_list = []
    samples_per_center = n_samples // n_centers

    for i in range(n_centers):
        n_points = samples_per_center + (1 if i < n_samples % n_centers else 0)
        cluster_points = rng.normal(loc=centers[i], scale=std, size=(n_points, 2))
        X_list.append(cluster_points)

    return np.vstack(X_list)


def load_data_from_csv(filepath: str) -> np.ndarray:
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                nums = [float(x) for x in row if x.strip()]
                if len(nums) >= 2:
                    data.append(nums[:2])
            except ValueError:
                continue  # Bỏ qua dòng tiêu đề hoặc dòng không hợp lệ

    if not data:
        raise ValueError("File CSV rỗng hoặc không chứa dữ liệu số hợp lệ.")

    return np.array(data)