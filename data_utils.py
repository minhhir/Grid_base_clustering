import numpy as np
import csv

# hàm generate_mock_data tạo ra dữ liệu giả cho phân cụm bằng cách tạo ra một số lượng mẫu n_samples được phân bố xung quanh n_centers trung tâm với độ lệch chuẩn std. Nó sử dụng numpy để tạo ra các điểm dữ liệu và trả về một mảng NumPy chứa tất cả các điểm.
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

# ham load_data_from_csv đọc dữ liệu từ một file CSV và trả về một mảng NumPy. Nó sử dụng csv.reader để đọc từng dòng của file, cố gắng chuyển đổi các giá trị thành số thực và chỉ giữ lại hai giá trị đầu tiên của mỗi dòng nếu chúng hợp lệ. Nếu file không chứa dữ liệu số hợp lệ, nó sẽ ném ra một lỗi ValueError.
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
                continue

    if not data:
        raise ValueError("File CSV rỗng hoặc không chứa dữ liệu số hợp lệ.")

    return np.array(data)