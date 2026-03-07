import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import csv

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithm import CLIQUEAlgorithm
from data_utils import generate_mock_data, load_data_from_csv

CLUSTER_COLORS = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6", "#F39C12", "#1ABC9C", "#E67E22"]

#lớp GridClusteringApp được thiết kế để tạo giao diện người dùng cho phần mềm phân cụm dựa trên lưới (CLIQUE). Nó sử dụng thư viện Tkinter để xây dựng giao diện và Matplotlib để hiển thị biểu đồ. Lớp này bao gồm các phương thức để tạo giao diện, xử lý sự kiện từ người dùng, và tương tác với thuật toán CLIQUE để thực hiện phân cụm và hiển thị kết quả.
class GridClusteringApp:
    #hàm khởi tạo nhận vào một đối tượng Tkinter root, thiết lập tiêu đề, kích thước và màu nền cho cửa sổ chính. Nó cũng khởi tạo các biến để lưu trữ dữ liệu và mô hình phân cụm, sau đó gọi phương thức để xây dựng giao diện người dùng và tạo dữ liệu mẫu ban đầu.
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Phần mềm Phân cụm Dựa trên Lưới (CLIQUE) – Đề tài 28")
        self.root.geometry("1100x700")
        self.root.configure(bg="#2C3E50")

        self.X = None
        self.model = None

        self._build_ui()
        self.cmd_generate_data()
    #hàm _create_section tạo ra một LabelFrame với tiêu đề và kiểu dáng được định nghĩa sẵn, sau đó đóng gói nó vào bố cục của giao diện. Nó trả về đối tượng LabelFrame để có thể thêm các widget con vào đó sau này.
    def _create_section(self, parent, title):
        f = tk.LabelFrame(parent, text=f" {title} ", bg="#2C3E50", fg="#ECF0F1", font=("Segoe UI", 9, "bold"), bd=1,
                          relief="groove", padx=8, pady=6)
        f.pack(fill=tk.X, pady=(0, 8))
        return f

    #hàm _create_entry tạo ra một Entry widget được liên kết với một biến Tkinter (textvariable) để người dùng có thể nhập dữ liệu. Nó cũng định dạng kiểu dáng của Entry và đóng gói nó vào bố cục.
    def _create_entry(self, parent, var):
        e = tk.Entry(parent, textvariable=var, bg="#34495E", fg="#ECF0F1", insertbackground="white",
                     font=("Segoe UI", 10), bd=0, relief="flat")
        e.pack(fill=tk.X, pady=(2, 6))

    #hàm _create_btn tạo ra một Button widget với văn bản, lệnh callback, màu nền và kiểu dáng được định nghĩa sẵn. Nó đóng gói nút vào bố cục và trả về đối tượng Button để có thể quản lý trạng thái của nó sau này (ví dụ: bật/tắt).
    def _create_btn(self, parent, text, cmd, color):
        # Trả về đối tượng Button để quản lý trạng thái Disabled/Normal
        btn = tk.Button(parent, text=text, command=cmd, bg=color, fg="white", font=("Segoe UI", 9, "bold"), bd=0,
                        relief="flat", activebackground="#1A6FA8", cursor="hand2", pady=6)
        btn.pack(fill=tk.X, pady=3)
        return btn

    #hàm _build_ui tạo ra cấu trúc chính của giao diện người dùng, bao gồm thanh tiêu đề, khu vực thân chính, và hai panel bên trái và bên phải. Nó gọi các phương thức phụ để xây dựng nội dung cho từng panel.
    def _build_ui(self):
        title_bar = tk.Frame(self.root, bg="#1A252F", height=50)
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="  PHÂN CỤM DỰA TRÊN LƯỚI (CLIQUE)  –  Đề tài 28", bg="#1A252F", fg="#ECF0F1",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=10, pady=10)

        body = tk.Frame(self.root, bg="#2C3E50")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = tk.Frame(body, bg="#2C3E50", width=270)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(body, bg="#2C3E50")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_left_panel(left)
        self._build_right_panel(right)

    #hàm _build_left_panel tạo ra các phần thông tin sinh viên, dữ liệu, tham số CLIQUE và điều khiển trong panel bên trái. Nó sử dụng các phương thức phụ để tạo các widget như Label, Entry và Button, đồng thời liên kết các nút với các phương thức xử lý sự kiện tương ứng (ví dụ: cmd_generate_data, cmd_load_csv, cmd_show_grid, cmd_run_algo, cmd_export). Nó cũng thiết lập một biến StringVar để hiển thị trạng thái hiện tại của ứng dụng.
    def _build_left_panel(self, parent):
        info = self._create_section(parent, "Thông tin Sinh viên")
        for line in ["Đề tài 28: Grid-based Clustering", "Môn: Trí tuệ nhân tạo", "SV: Hoàng Năng Minh - 11236154",
                     "Lớp: CNTT 65B"]:
            tk.Label(info, text=line, bg="#2C3E50", fg="#F1C40F" if "Minh" in line else "#ECF0F1",
                     font=("Segoe UI", 8, "bold" if "Minh" in line else "normal")).pack(anchor=tk.W)

        data_sec = self._create_section(parent, "Dữ liệu")
        tk.Label(data_sec, text="Số điểm dữ liệu:", bg="#2C3E50", fg="#BDC3C7").pack(anchor=tk.W)
        self.n_samples_var = tk.IntVar(value=400)
        self._create_entry(data_sec, self.n_samples_var)

        tk.Label(data_sec, text="Số cụm thực tế:", bg="#2C3E50", fg="#BDC3C7").pack(anchor=tk.W)
        self.n_centers_var = tk.IntVar(value=3)
        self._create_entry(data_sec, self.n_centers_var)

        self.btn_gen = self._create_btn(data_sec, "Tạo ngẫu nhiên", self.cmd_generate_data, "#27AE60")
        self.btn_load = self._create_btn(data_sec, "Tải file CSV", self.cmd_load_csv, "#8E44AD")

        param_sec = self._create_section(parent, "Tham số CLIQUE")
        tk.Label(param_sec, text="Số ô chia mỗi trục (k):", bg="#2C3E50", fg="#BDC3C7").pack(anchor=tk.W)
        self.k_var = tk.IntVar(value=10)
        self._create_entry(param_sec, self.k_var)

        tk.Label(param_sec, text="Ngưỡng mật độ (MinPts ξ):", bg="#2C3E50", fg="#BDC3C7").pack(anchor=tk.W)
        self.xi_var = tk.IntVar(value=6)
        self._create_entry(param_sec, self.xi_var)

        ctrl_sec = self._create_section(parent, "Điều khiển")
        self.btn_grid = self._create_btn(ctrl_sec, "Xem Lưới trung gian", self.cmd_show_grid, "#E67E22")
        self.btn_run = self._create_btn(ctrl_sec, "Chạy Phân Cụm", self.cmd_run_algo, "#E74C3C")
        self.btn_export = self._create_btn(ctrl_sec, "Xuất kết quả", self.cmd_export, "#7F8C8D")

        stat_sec = self._create_section(parent, "Trạng thái")
        self.stat_text = tk.StringVar(value="Sẵn sàng.")
        tk.Label(stat_sec, textvariable=self.stat_text, bg="#2C3E50", fg="#2ECC71", font=("Courier", 9),
                 justify=tk.LEFT).pack(anchor=tk.W)

    #hàm _build_right_panel tạo ra một Notebook (tabbed interface) trong panel bên phải với hai tab: "Biểu đồ" để hiển thị biểu đồ phân cụm và "Nhật ký" để hiển thị các thông điệp log từ quá trình phân cụm. Nó sử dụng Matplotlib để tạo biểu đồ và một Text widget để hiển thị log, đồng thời thiết lập kiểu dáng cho các widget này để phù hợp với giao diện tổng thể của ứng dụng.
    def _build_right_panel(self, parent):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#2C3E50", borderwidth=0)
        style.configure("TNotebook.Tab", background="#34495E", foreground="#ECF0F1", font=("Segoe UI", 9, "bold"),
                        padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", "#2980B9")])

        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill=tk.BOTH, expand=True)

        chart_tab = tk.Frame(self.nb, bg="#1E2B38")
        self.nb.add(chart_tab, text="Biểu đồ ")

        self.fig = Figure(figsize=(7, 5), facecolor="#1E2B38")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        log_tab = tk.Frame(self.nb, bg="#1A252F")
        self.nb.add(log_tab, text="Nhật ký ")
        self.log_text = tk.Text(log_tab, bg="#0D1B2A", fg="#2ECC71", font=("Courier New", 9), bd=0, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    #hàm _log là một phương thức tiện ích để ghi các thông điệp log vào Text widget trong tab "Nhật ký". Nó cho phép cập nhật giao diện ngay lập tức sau khi thêm log mới, giúp người dùng theo dõi tiến trình của thuật toán phân cụm.
    def _log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    #hàm _setup_ax thiết lập lại trục của biểu đồ với tiêu đề mới và kiểu dáng được định nghĩa sẵn. Nó xóa nội dung cũ trên trục, đặt màu nền, màu sắc cho các tick và spines, và cập nhật tiêu đề với màu sắc và kích thước font phù hợp.
    def _setup_ax(self, title):
        self.ax.clear()
        self.ax.set_facecolor("#1E2B38")
        self.ax.tick_params(colors="#BDC3C7", labelsize=7)
        for spine in self.ax.spines.values(): spine.set_edgecolor("#4A6274")
        self.ax.set_title(title, color="#ECF0F1", fontsize=10)

    #hàm _validate_params kiểm tra tính hợp lệ của các tham số k và xi được nhập bởi người dùng. Nó đảm bảo rằng k lớn hơn 1 và xi lớn hơn hoặc bằng 1. Nếu có lỗi trong quá trình kiểm tra (ví dụ: người dùng nhập giá trị không phải là số nguyên), nó sẽ hiển thị một thông báo lỗi và trả về None cho cả hai tham số.
    def _validate_params(self):
        try:
            k = self.k_var.get()
            xi = self.xi_var.get()
            if k <= 1 or xi < 1:
                raise ValueError("k > 1 và MinPts >= 1")
            return k, xi
        except (tk.TclError, ValueError):
            messagebox.showerror("Lỗi dữ liệu", "Vui lòng nhập số nguyên hợp lệ cho k (>1) và MinPts (>=1).")
            return None, None
    #hàm cmd_generate_data được gọi khi người dùng nhấn nút "Tạo ngẫu nhiên". Nó lấy số lượng mẫu và số cụm từ các Entry, kiểm tra tính hợp lệ của chúng, sau đó sử dụng hàm generate_mock_data để tạo ra dữ liệu mới. Dữ liệu này sau đó được hiển thị trên biểu đồ và thông tin về số điểm được tạo ra được cập nhật trong trạng thái. Nếu có lỗi trong quá trình tạo dữ liệu, nó sẽ hiển thị một thông báo lỗi.
    def cmd_generate_data(self):
        try:
            n = self.n_samples_var.get()
            c = self.n_centers_var.get()
        except tk.TclError:
            messagebox.showerror("Lỗi nhập liệu", "Vui lòng nhập số nguyên hợp lệ.")
            return

        if n < 10 or c < 1:
            messagebox.showerror("Lỗi logic", "Số điểm tối thiểu là 10, Số cụm tối thiểu là 1.")
            return

        try:
            self.X = generate_mock_data(n, c, 1.2)
            self._setup_ax("Dữ liệu ban đầu")
            self.ax.scatter(self.X[:, 0], self.X[:, 1], s=12, c="#3498DB", alpha=0.7, edgecolors="none")
            self.canvas.draw()
            self.stat_text.set(f"Đã tạo {len(self.X)} điểm.")
            self._log(f"[Dữ liệu] Khởi tạo {len(self.X)} điểm ngẫu nhiên.")
        except Exception as e:
            messagebox.showerror("Lỗi hệ thống", f"Phát sinh lỗi:\n{e}")

    #hàm cmd_load_csv được gọi khi người dùng nhấn nút "Tải file CSV". Nó mở một hộp thoại để người dùng chọn file CSV, sau đó sử dụng hàm load_data_from_csv để đọc dữ liệu từ file. Dữ liệu này được hiển thị trên biểu đồ và thông tin về số điểm được tải lên được cập nhật trong trạng thái. Nếu có lỗi trong quá trình nạp dữ liệu (ví dụ: file không hợp lệ hoặc không chứa dữ liệu số), nó sẽ hiển thị một thông báo lỗi.
    def cmd_load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path: return

        try:
            self.X = load_data_from_csv(path)
            if len(self.X) < 2:
                raise ValueError("Tập dữ liệu quá nhỏ.")

            self._setup_ax(f"Dữ liệu từ: {os.path.basename(path)}")
            self.ax.scatter(self.X[:, 0], self.X[:, 1], s=12, c="#3498DB", alpha=0.7, edgecolors="none")
            self.canvas.draw()
            self.stat_text.set(f"Đã tải {len(self.X)} điểm.")
            self._log(f"[File] Đã tải {len(self.X)} điểm từ CSV.")
        except Exception as e:
            messagebox.showerror("Lỗi nạp dữ liệu", f"Không thể xử lý file này:\n{e}")

    #hàm cmd_show_grid được gọi khi người dùng nhấn nút "Xem Lưới trung gian". Nó kiểm tra xem dữ liệu đã được nạp hay chưa, sau đó lấy các tham số k và xi từ Entry và kiểm tra tính hợp lệ của chúng. Nếu mọi thứ hợp lệ, nó tạo một instance của CLIQUEAlgorithm với dữ liệu và tham số đã cho, xây dựng lưới và tính mật độ cho từng ô. Cuối cùng, nó hiển thị lưới trên biểu đồ, tô màu các ô đạt ngưỡng mật độ bằng màu cam và hiển thị số lượng điểm trong mỗi ô đó.
    def cmd_show_grid(self):
        if self.X is None: return
        k, xi = self._validate_params()
        if k is None: return

        m = CLIQUEAlgorithm(self.X, k, xi, self._log)
        m._build_grid()
        m._compute_density()

        self._setup_ax(f"Lưới {k}x{k} - Ô màu cam đạt MinPts={xi}")
        xe, ye, H = m.x_edges_, m.y_edges_, m.density_grid_

        for i in range(k):
            for j in range(k):
                if H[i, j] >= xi:
                    self.ax.add_patch(
                        mpatches.Rectangle((xe[i], ye[j]), xe[i + 1] - xe[i], ye[j + 1] - ye[j], facecolor="#F39C12",
                                           alpha=0.4))
                    self.ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, str(int(H[i, j])), color="#F1C40F",
                                 ha="center", va="center", fontsize=7)

        for x in xe: self.ax.axvline(x, color="#4A6274", lw=0.5, ls="--")
        for y in ye: self.ax.axhline(y, color="#4A6274", lw=0.5, ls="--")
        self.ax.scatter(self.X[:, 0], self.X[:, 1], s=10, c="#3498DB", alpha=0.6, edgecolors="none")
        self.canvas.draw()
        self.nb.select(0)

    #hàm cmd_run_algo được gọi khi người dùng nhấn nút "Chạy Phân Cụm". Nó kiểm tra xem dữ liệu đã được nạp hay chưa, sau đó lấy các tham số k và xi từ Entry và kiểm tra tính hợp lệ của chúng. Nếu mọi thứ hợp lệ, nó tạo một instance của CLIQUEAlgorithm với dữ liệu và tham số đã cho, sau đó gọi phương thức fit() để thực hiện phân cụm. Kết quả phân cụm được hiển thị trên biểu đồ với các màu khác nhau cho từng cụm và các điểm nhiễu được đánh dấu bằng dấu "x". Sau khi hoàn thành, nó cập nhật trạng thái với số lượng cụm tìm được. Nút "Chạy Phân Cụm" cũng được khóa trong quá trình chạy để ngăn người dùng nhấn nhiều
    def cmd_run_algo(self):
        if self.X is None: return
        k, xi = self._validate_params()
        if k is None: return

        self.nb.select(1)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        # Khắc phục: Khóa nút trong lúc thuật toán chạy
        self.btn_run.configure(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            self.model = CLIQUEAlgorithm(self.X, k, xi, self._log).fit()

            self._setup_ax(f"Kết quả: {self.model.n_clusters_} cụm")
            xe, ye = self.model.x_edges_, self.model.y_edges_

            for cid, cells in enumerate(self.model.cluster_cells_):
                col = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                for (ci, cj) in cells:
                    self.ax.add_patch(
                        mpatches.Rectangle((xe[ci], ye[cj]), xe[ci + 1] - xe[ci], ye[cj + 1] - ye[cj], facecolor=col,
                                           alpha=0.2))

            for cid in range(self.model.n_clusters_):
                mask = self.model.labels_ == cid
                self.ax.scatter(self.X[mask, 0], self.X[mask, 1], s=14, color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])

            noise = self.model.labels_ == -1
            if noise.any():
                self.ax.scatter(self.X[noise, 0], self.X[noise, 1], s=10, color="#7F8C8D", marker="x")

            self.canvas.draw()
            self.nb.select(0)
            self.stat_text.set(f"Hoàn thành: {self.model.n_clusters_} cụm.")
        finally:
            self.btn_run.configure(state=tk.NORMAL)

    #hàm cmd_export được gọi khi người dùng nhấn nút "Xuất kết quả". Nó kiểm tra xem mô hình phân cụm đã được tạo ra hay chưa, sau đó mở một hộp thoại để người dùng chọn vị trí và tên file CSV để lưu kết quả. Kết quả được lưu dưới dạng file CSV với ba cột: x, y và cluster (nơi cluster là nhãn cụm hoặc -1 cho điểm nhiễu). Nếu có lỗi trong quá trình lưu file (ví dụ: file đang mở ở nơi khác), nó sẽ hiển thị một thông báo lỗi.
    def cmd_export(self):
        if self.model is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="ket_qua.csv")
        if not path: return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "cluster"])
                for (x, y), lbl in zip(self.X, self.model.labels_):
                    writer.writerow([round(x, 4), round(y, 4), int(lbl)])
            messagebox.showinfo("OK", "Đã lưu kết quả.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file (file có thể đang mở ở nơi khác):\n{e}")