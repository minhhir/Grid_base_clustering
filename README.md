# 🎯 Phân cụm dựa trên Lưới (Grid-Based Clustering) - Thuật toán CLIQUE

Dự án này là bài tập lớn môn **Trí tuệ nhân tạo (HK2 2025-2026)**, triển khai phần mềm trực quan hóa thuật toán phân cụm dựa trên lưới (Grid-based Clustering) sử dụng phương pháp **CLIQUE**.

🎓 **Sinh viên thực hiện:** Hoàng Năng Minh (MSSV: 11236154)  
🏫 **Lớp:** CNTT 65B  
👨‍🏫 **Giảng viên hướng dẫn:** GVC. TS. Lưu Minh Tuấn  
📌 **Đề tài:** Số 28

---

## 📖 1. Kiến thức nền tảng (Cơ sở lý thuyết)

Phương pháp phân cụm dựa trên lưới (Grid-based Clustering) chia không gian dữ liệu nhiều chiều thành một số lượng hữu hạn các ô (cells) để tạo thành một cấu trúc lưới. Mọi thao tác tính toán và phân cụm đều được thực hiện trực tiếp trên lưới này thay vì trên từng điểm dữ liệu riêng lẻ, giúp thuật toán có tốc độ xử lý cực kỳ nhanh và không bị phụ thuộc vào số lượng điểm dữ liệu ban đầu.

**Thuật toán CLIQUE (CLustering In QUEst) hoạt động qua 4 bước chính:**
1. **Khởi tạo lưới:** Không gian dữ liệu được chia đều thành $k \times k$ ô chữ nhật không chồng chéo.
2. **Tính toán mật độ (Density):** Đếm số lượng điểm dữ liệu rơi vào từng ô. Các ô có số điểm vượt qua một ngưỡng MinPts ($\xi$) cho trước được đánh dấu là các **Ô dày đặc (Dense Units)**.
3. **Gom cụm (Clustering):** Sử dụng thuật toán duyệt đồ thị (Breadth-First Search - BFS) để tìm các thành phần liên thông. Các ô dày đặc nằm liền kề nhau (chung cạnh) sẽ được gom lại thành một cụm hoàn chỉnh.
4. **Gán nhãn:** Các điểm dữ liệu nằm trong các ô thuộc cụm sẽ được gán nhãn của cụm đó. Các điểm nằm ở các ô không đạt ngưỡng mật độ bị coi là nhiễu (Noise).

---

## ✨ 2. Tính năng nổi bật của phần mềm

- **Tối ưu hóa hiệu năng cao:** Thuật toán được viết hoàn toàn bằng `numpy` thuần (histogram2d), không phụ thuộc vào các thư viện nặng như `scikit-learn` hay `pandas`, giúp tính toán mượt mà với tập dữ liệu lớn.
- **Giao diện trực quan (GUI):** Xây dựng bằng `Tkinter` kết hợp `matplotlib`, cho phép người dùng tương tác trực tiếp, thay đổi tham số ($k$, $\xi$) và xem kết quả ngay lập tức.
- **Minh họa từng bước (Step-by-step):** Hiển thị rõ ràng các bước trung gian (vẽ lưới, đánh dấu ô dày đặc) giúp dễ dàng hiểu cách thuật toán hoạt động.
- **Hỗ trợ I/O linh hoạt:** Cho phép sinh dữ liệu ngẫu nhiên giả lập hoặc nhập/xuất tọa độ qua tệp `.csv`.
- **Nhật ký chạy (Log):** Theo dõi chi tiết các thông số toán học và thời gian thực thi (ms) của từng bước.

---

## 📂 3. Cấu trúc mã nguồn (Project Structure)

Mã nguồn được thiết kế theo chuẩn Kỹ thuật phần mềm (OOP), chia thành các module độc lập:

```text
Grid_base_clustering/
 ├── algorithm.py      # Lõi thuật toán toán học CLIQUE (Chia lưới, đếm mật độ, BFS)
 ├── data_utils.py     # Module sinh dữ liệu ngẫu nhiên và đọc/ghi file CSV
 ├── gui.py            # Giao diện người dùng Tkinter & vẽ biểu đồ Matplotlib
 ├── main.py           # File khởi chạy ứng dụng
 ├── build_exe.bat     # Script tự động đóng gói dự án thành file .exe siêu nhẹ
 └── README.md         # Tài liệu hướng dẫn
```
⚙️ 4. Hướng dẫn Cài đặt & Sử dụngCách 1: Chạy trực tiếp bằng Python (Dành cho Developer)Yêu cầu môi trường: Python 3.8+Clone repository về máy:Bashgit clone [https://github.com/minhhir/Grid_base_clustering.git](https://github.com/minhhir/Grid_base_clustering.git)
cd Grid_base_clustering
Cài đặt các thư viện cần thiết:Bashpip install numpy matplotlib
Khởi chạy phần mềm:Bashpython main.py
Cách 2: Đóng gói thành phần mềm chạy độc lập (.exe)Nếu bạn muốn gửi phần mềm cho giảng viên hoặc chạy trên máy tính không cài sẵn Python:Nhấn đúp chuột vào file build_exe.bat.Script sẽ tự động dọn dẹp, loại bỏ các thư viện thừa và dùng PyInstaller nén mã nguồn lại.Lấy phần mềm hoàn chỉnh tại thư mục dist/GridClustering_De28.exe (dung lượng đã được tối ưu siêu nhỏ).🎮 5. Các bước thao tác trên giao diệnKhởi tạo dữ liệu: Tại thẻ ① Dữ liệu, nhập số lượng điểm hoặc nhấn Tải file CSV để nạp dữ liệu.Thiết lập tham số: Tại thẻ ② Tham số CLIQUE, nhập số ô chia ($k$) và ngưỡng mật độ ($\xi$).Xem bước trung gian: Nhấn Xem Lưới trung gian để quan sát cách không gian bị chia cắt và các ô dày đặc (màu cam) xuất hiện.Phân cụm: Nhấn Chạy Phân Cụm để gom các ô liên thông. Đọc tab Nhật ký để xem chi tiết tốc độ chạy.Xuất file: Nhấn Xuất kết quả để lưu dữ liệu đã được gán nhãn cụm ra tệp .csv.
