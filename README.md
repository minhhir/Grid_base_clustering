# Phân cụm Dựa trên Lưới — CLIQUE & GCBD

**Bài tập lớn môn Trí tuệ nhân tạo — HK2 2025-2026 — Đề tài số 28**

| | |
|---|---|
| **Sinh viên** | Hoàng Năng Minh — MSSV: 11***** |
| **Lớp** | CNTT 65B |
| **Ngôn ngữ** | Python 3.8+ |
| **Thư viện** | numpy ~1.26, matplotlib ~3.8 |

---

## Mục lục

1. [Cơ sở lý thuyết](#1-cơ-sở-lý-thuyết)
2. [Hai thuật toán được cài đặt](#2-hai-thuật-toán-được-cài-đặt)
3. [Cấu trúc mã nguồn](#3-cấu-trúc-mã-nguồn)
4. [Cài đặt và chạy](#4-cài-đặt-và-chạy)
5. [Hướng dẫn sử dụng GUI](#5-hướng-dẫn-sử-dụng-gui)
6. [Chọn tham số đúng](#6-chọn-tham-số-đúng)
7. [Định dạng file CSV](#7-định-dạng-file-csv)

---

## 1. Cơ sở lý thuyết

Phân cụm dựa trên lưới (Grid-based Clustering) chia không gian dữ liệu thành một lưới các ô hữu hạn rồi thực hiện tính toán trên lưới thay vì trên từng điểm. Ưu điểm cốt lõi: **độ phức tạp thời gian không phụ thuộc vào số điểm n**, mà chỉ phụ thuộc vào độ phân giải lưới — phù hợp cho tập dữ liệu hàng triệu điểm.

---

## 2. Hai thuật toán được cài đặt

### 2.1 CLIQUE *(CLustering In QUEst)*

Thuật toán gốc của Agrawal et al. (1998), hoạt động qua 4 bước:

**Bước 1 — Khởi tạo lưới:** Chia không gian thành k × k ô chữ nhật bằng nhau, không chồng lấp.

**Bước 2 — Tính mật độ:** Đếm số điểm rơi vào mỗi ô bằng `numpy.histogram2d` (O(n)). Ô có số điểm ≥ MinPts (ký hiệu ξ) được đánh dấu là **ô dày đặc (dense unit)**.

**Bước 3 — Gom cụm (BFS):** Duyệt đồ thị theo chiều rộng (BFS) trên các ô dày đặc. Hai ô chung cạnh (4-connectivity) thuộc cùng một thành phần liên thông → cùng một cụm.

**Bước 4 — Gán nhãn:** Điểm nằm trong ô thuộc cụm nào thì nhận nhãn cụm đó. Điểm ở ô không dày đặc → nhãn `-1` (noise).

**Tham số:**

| Tham số | Ký hiệu | Ý nghĩa |
|---------|---------|---------|
| Số ô mỗi trục | k | Độ phân giải lưới. k lớn → chi tiết hơn, chậm hơn một chút |
| Ngưỡng mật độ | ξ (MinPts) | Số điểm tối thiểu để ô được coi là dày đặc |

---

### 2.2 GCBD *(Grid-based Clustering using Boundary Detection)*

Cải tiến của Du & Wu (Entropy 2022, doi:10.3390/e24111606). Khắc phục nhược điểm lớn nhất của CLIQUE: không cần đặt MinPts thủ công.

**Ba cải tiến chính so với CLIQUE:**

**① Mật độ tại nút (node), không phải tại ô:** Mỗi điểm dữ liệu đóng góp vào mật độ của 4 nút góc gần nhất theo trọng số bilinear (khoảng cách tỉ lệ nghịch). Giúp biểu diễn mật độ mịn và liên tục hơn.

**② Phát hiện biên lặp (T vòng):** Mỗi vòng, 10% nút hoạt động có mật độ thấp nhất (percentile-10) bị loại ra khỏi core. Quá trình này tự động phân biệt vùng đặc (core) và vùng biên mà không cần MinPts.

**③ Gán boundary node theo hàng xóm mật độ cao nhất:** Các nút biên bị loại được gán lại vào cụm của hàng xóm 8-connected có mật độ cao nhất tại thời điểm bị loại (cảm hứng từ DPC). Giảm đáng kể noise giả.

**Tham số:**

| Tham số | Ký hiệu | Ý nghĩa |
|---------|---------|---------|
| Số khoảng chia | l | Tương đương k trong CLIQUE. Quyết định kích thước ô |
| Số vòng lặp biên | T | Số vòng lặp phát hiện biên. Bài báo khuyến nghị 2–12 |

---

## 3. Cấu trúc mã nguồn

```
Grid_base_clustering/
├── main.py          # Điểm khởi chạy — tạo Tk root, gọi GridClusteringApp
├── gui.py           # Giao diện Tkinter + Matplotlib; điều phối 2 thuật toán
├── algorithm.py     # Lõi toán học: CLIQUEAlgorithm, GCBDAlgorithm
├── data_utils.py    # Sinh dữ liệu ngẫu nhiên; đọc/ghi CSV
├── requirements.txt # Phụ thuộc thư viện
└── README.md
```

**Luồng dữ liệu chính:**

```
GUI (gui.py)
  │  nhập tham số
  ▼
CLIQUEAlgorithm / GCBDAlgorithm  (algorithm.py)
  │  .fit() → labels_, n_clusters_, cluster_cells_
  ▼
GUI vẽ kết quả lên matplotlib canvas
  │  tuỳ chọn
  ▼
data_utils.py → xuất CSV
```

---

## 4. Cài đặt và chạy

### Yêu cầu

- Python **3.8** trở lên
- pip

### Cài thư viện

```bash
pip install numpy~=1.26.0 matplotlib~=3.8.0
```

hoặc dùng file có sẵn:

```bash
pip install -r requirements.txt
```

### Khởi chạy

```bash
python main.py
```

---

## 5. Hướng dẫn sử dụng GUI

Giao diện chia làm hai vùng: **thanh điều khiển bên trái** và **khu vực hiển thị bên phải** (tab Biểu đồ và tab Nhật ký).

### Bước 1 — Chuẩn bị dữ liệu

Có hai cách nạp dữ liệu:

**Tạo ngẫu nhiên:** Nhập số điểm, số cụm thực tế, độ phân tán rồi nhấn *Tạo ngẫu nhiên*. Dữ liệu được sinh từ các phân phối Gaussian với tâm ngẫu nhiên trong `[-8, 8]².

**Tải file CSV:** Nhấn *Tải file CSV* và chọn file (xem định dạng ở mục 7).

### Bước 2 — Chọn thuật toán và nhập tham số

Chọn **CLIQUE** hoặc **GCBD** bằng radio button. Panel tham số tương ứng sẽ hiện ra.

Phía dưới ô nhập tham số có **dòng gợi ý màu vàng** — cập nhật tự động khi bạn thay đổi bất kỳ thông số nào. Đọc gợi ý này trước khi chạy để tránh kết quả sai (xem chi tiết ở mục 6).

### Bước 3 — Xem lưới trung gian *(tuỳ chọn)*

Nhấn *Xem Lưới trung gian* để quan sát không gian bị chia lưới trước khi phân cụm:

- **CLIQUE:** Hiển thị lưới k×k, tô cam các ô đạt MinPts, in số điểm trong mỗi ô.
- **GCBD:** Hiển thị mật độ tại từng nút bằng kích thước và màu sắc điểm (vàng → đỏ = mật độ tăng dần).

### Bước 4 — Chạy phân cụm

Nhấn *Chạy Phân Cụm*. Thuật toán chạy trên luồng riêng (không đóng băng GUI). Kết quả hiển thị trên tab Biểu đồ; log chi tiết (thời gian từng bước, số ô dày đặc, số cụm, số noise) hiển thị trên tab Nhật ký.

Thanh trạng thái phía dưới trái tóm tắt: số cụm tìm được, số điểm thuộc cụm, số noise.

### Bước 5 — Xuất kết quả *(tuỳ chọn)*

Nhấn *Xuất kết quả* để lưu file CSV với ba cột: `x`, `y`, `cluster` (cluster = -1 nghĩa là noise).

---

## 6. Chọn tham số đúng

### CLIQUE — MinPts (ξ)

MinPts phải tương xứng với **mật độ thực tế** của dữ liệu, không phải một con số cố định.

**Công thức ước lượng nhanh:**

```
MinPts ≈ (n / n_cụm) / (4 × std / cell_size)² × 0.05
```

Trong đó `cell_size = khoảng_giá_trị / k`.

**Ví dụ thực tế:**

| n | k | std | MinPts gợi ý | Lý do |
|---|---|-----|-------------|-------|
| 400 | 10 | 1.2 | ~3 | Mật độ thấp, ô dày đặc có vài chục điểm |
| 10,000,000 | 10 | 1.2 | ~8,000 | Mật độ TB = 100k/ô, rìa cụm vẫn có hàng nghìn điểm |
| 10,000,000 | 10 | 0.5 | ~55,000 | Cụm rất đặc, rìa tuy thưa vẫn có hàng chục nghìn điểm |

**Dấu hiệu MinPts sai:**
- Quá thấp → các cụm riêng biệt bị nối lại thành một (như ảnh minh hoạ ở trên).
- Quá cao → mất cụm, gần hết điểm bị gán là noise.

GUI tự động gợi ý và cảnh báo (`⚠`) khi MinPts lệch xa giá trị khuyến nghị.

---

### GCBD — l và T

**Tham số l (số khoảng chia):**

Quy tắc quan trọng: `cell_size` phải **nhỏ hơn std** của cụm để bilinear kernel phân biệt được tâm cụm và vùng biên.

```
l tối thiểu ≈ khoảng_giá_trị / (std × 0.5)
```

| std | l gợi ý (với range ≈ 16) |
|-----|--------------------------|
| 0.5 | ≥ 64 |
| 1.2 | ≥ 27 (mặc định 20 có thể chấp nhận) |
| 3.0 | ≥ 11 |

Nếu l quá nhỏ so với std: toàn bộ cụm chỉ chiếm 1–2 nút → kernel nhòe mật độ sang vùng giữa hai cụm → BFS nối chúng lại (lỗi tương tự CLIQUE với MinPts thấp).

**Tham số T (số vòng lặp biên):**

Bài báo gốc khuyến nghị T trong khoảng **2–12**. Tăng T khi:
- l lớn (nhiều nút → cần thêm vòng để lọc hết biên).
- Cụm có hình dạng bất thường hoặc mật độ không đều.

Giảm T khi cụm nhỏ để tránh bị ăn mòn core.

GUI cũng gợi ý T theo l và cảnh báo khi T > 12.

---

## 7. Định dạng file CSV

File CSV đầu vào cần ít nhất **2 cột số** trên mỗi dòng, đại diện cho toạ độ x và y. Dòng tiêu đề (nếu có) được bỏ qua tự động.

```csv
x,y
1.23,4.56
-2.10,3.88
...
```

File CSV đầu ra (xuất kết quả) có 3 cột:

```csv
x,y,cluster
1.23,4.56,0
-2.10,3.88,1
0.00,9.99,-1
```

Cluster = `-1` là noise (điểm không thuộc cụm nào).
