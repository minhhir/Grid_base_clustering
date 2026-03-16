import numpy as np
import csv


def generate_mock_data(n_samples: int, n_centers: int, std: float) -> np.ndarray:
    """
    Tạo dữ liệu giả gồm các cụm Gaussian phân bố trong mặt phẳng 2D.

    Args:
        n_samples: Tổng số điểm dữ liệu cần tạo.
        n_centers: Số cụm (tâm) ngẫu nhiên.
        std: Độ lệch chuẩn của mỗi cụm (kiểm soát độ phân tán).

    Returns:
        Mảng NumPy shape (n_samples, 2).
    """
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
    """
    Đọc dữ liệu 2D từ file CSV.

    Bỏ qua các dòng tiêu đề hoặc dòng không chứa đủ 2 giá trị số.

    Args:
        filepath: Đường dẫn tới file CSV.

    Returns:
        Mảng NumPy shape (n, 2).

    Raises:
        ValueError: Nếu file rỗng hoặc không có dòng số hợp lệ nào.
    """
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