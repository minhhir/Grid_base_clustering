import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import csv
import threading

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithm import CLIQUEAlgorithm, GCBDAlgorithm
from data_utils import generate_mock_data, load_data_from_csv

_FALLBACK_COLORS = [
    "#E74C3C", "#2ECC71", "#3498DB", "#9B59B6",
    "#F39C12", "#1ABC9C", "#E67E22", "#E91E63",
    "#00BCD4", "#8BC34A",
]


def _cluster_color(cid: int, n_total: int) -> str:
    """Màu cho cụm cid: dùng bảng cố định khi <= 10 cụm, colormap khi nhiều hơn."""
    if n_total <= len(_FALLBACK_COLORS):
        return _FALLBACK_COLORS[cid % len(_FALLBACK_COLORS)]
    rgba = cm.get_cmap("tab20")(cid % 20)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
    )


class GridClusteringApp:
    """
    Giao diện đồ họa hỗ trợ hai thuật toán:
        • CLIQUE  – phân cụm dựa trên ngưỡng mật độ ô cứng (MinPts)
        • GCBD   – phân cụm dựa trên phát hiện biên lặp (Du & Wu, 2022)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Phân cụm Dựa trên Lưới – CLIQUE & GCBD  |  Đề tài 28")
        self.root.geometry("1150x720")
        self.root.configure(bg="#2C3E50")

        self.X: np.ndarray | None = None
        self.model = None
        self._log_buffer: list[str] = []

        self._build_ui()
        self.cmd_generate_data()

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _section(self, parent, title: str) -> tk.LabelFrame:
        f = tk.LabelFrame(parent, text=f" {title} ", bg="#2C3E50", fg="#ECF0F1",
                          font=("Segoe UI", 9, "bold"), bd=1, relief="groove",
                          padx=8, pady=6)
        f.pack(fill=tk.X, pady=(0, 7))
        return f

    def _label(self, parent, text: str):
        tk.Label(parent, text=text, bg="#2C3E50", fg="#BDC3C7",
                 font=("Segoe UI", 8)).pack(anchor=tk.W)

    def _entry(self, parent, var) -> tk.Entry:
        e = tk.Entry(parent, textvariable=var, bg="#34495E", fg="#ECF0F1",
                     insertbackground="white", font=("Segoe UI", 10), bd=0, relief="flat")
        e.pack(fill=tk.X, pady=(2, 5))
        return e

    def _btn(self, parent, text: str, cmd, color: str) -> tk.Button:
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                      font=("Segoe UI", 9, "bold"), bd=0, relief="flat",
                      activebackground="#1A6FA8", cursor="hand2", pady=6)
        b.pack(fill=tk.X, pady=3)
        return b

    def _build_ui(self):
        # Title bar
        tb = tk.Frame(self.root, bg="#1A252F", height=50)
        tb.pack(fill=tk.X)
        tk.Label(tb, text="  PHÂN CỤM DỰA TRÊN LƯỚI  –  CLIQUE & GCBD  –  Đề tài 28",
                 bg="#1A252F", fg="#ECF0F1",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=10, pady=10)

        body = tk.Frame(self.root, bg="#2C3E50")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = tk.Frame(body, bg="#2C3E50", width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(body, bg="#2C3E50")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        # Thông tin SV
        info = self._section(parent, "Thông tin Sinh viên")
        for line in ["Đề tài 28: Grid-based Clustering",
                     "Môn: Trí tuệ nhân tạo",
                     "SV: Hoàng Năng Minh - 11236154"]:
            tk.Label(info, text=line, bg="#2C3E50",
                     fg="#F1C40F" if "Minh" in line else "#ECF0F1",
                     font=("Segoe UI", 8, "bold" if "Minh" in line else "normal")
                     ).pack(anchor=tk.W)

        # Dữ liệu
        ds = self._section(parent, "Dữ liệu")
        self._label(ds, "Số điểm dữ liệu:")
        self.n_samples_var = tk.IntVar(value=400)
        self._entry(ds, self.n_samples_var)

        self._label(ds, "Số cụm thực tế:")
        self.n_centers_var = tk.IntVar(value=3)
        self._entry(ds, self.n_centers_var)

        self._label(ds, "Độ phân tán (std):")
        self.std_var = tk.DoubleVar(value=1.2)
        self._entry(ds, self.std_var)

        self.btn_gen  = self._btn(ds, "Tạo ngẫu nhiên", self.cmd_generate_data, "#27AE60")
        self.btn_load = self._btn(ds, "Tải file CSV",   self.cmd_load_csv,      "#8E44AD")

        # Chọn thuật toán
        algo_sec = self._section(parent, "Chọn thuật toán")
        self.algo_var = tk.StringVar(value="CLIQUE")

        algo_frame = tk.Frame(algo_sec, bg="#2C3E50")
        algo_frame.pack(fill=tk.X)
        for alg in ("CLIQUE", "GCBD"):
            tk.Radiobutton(
                algo_frame, text=alg, variable=self.algo_var, value=alg,
                command=self._on_algo_change,
                bg="#2C3E50", fg="#ECF0F1", selectcolor="#1A252F",
                activebackground="#2C3E50", font=("Segoe UI", 9, "bold")
            ).pack(side=tk.LEFT, padx=6, pady=2)

        # Mô tả ngắn thuật toán đang chọn
        self.algo_desc_var = tk.StringVar()
        tk.Label(algo_sec, textvariable=self.algo_desc_var, bg="#2C3E50",
                 fg="#95A5A6", font=("Segoe UI", 7), wraplength=240,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

        # Tham số
        param_sec = self._section(parent, "Tham số")

        # CLIQUE params
        self.frm_clique = tk.Frame(param_sec, bg="#2C3E50")
        self._label(self.frm_clique, "Số ô mỗi trục (k):")
        self.k_var = tk.IntVar(value=10)
        self._entry(self.frm_clique, self.k_var)
        self._label(self.frm_clique, "Ngưỡng mật độ (MinPts ξ):")
        self.xi_var = tk.IntVar(value=6)
        self._entry(self.frm_clique, self.xi_var)
        self.frm_clique.pack(fill=tk.X)

        # GCBD params
        self.frm_gcbd = tk.Frame(param_sec, bg="#2C3E50")
        self._label(self.frm_gcbd, "Số khoảng chia (l):")
        self.l_var = tk.IntVar(value=20)
        self._entry(self.frm_gcbd, self.l_var)
        self._label(self.frm_gcbd, "Số vòng lặp phát hiện biên (T):")
        self.T_var = tk.IntVar(value=5)
        self._entry(self.frm_gcbd, self.T_var)
        # frm_gcbd ẩn ban đầu

        # Điều khiển
        ctrl = self._section(parent, "Điều khiển")
        self.btn_grid   = self._btn(ctrl, "Xem Lưới trung gian", self.cmd_show_grid, "#E67E22")
        self.btn_run    = self._btn(ctrl, "Chạy Phân Cụm",       self.cmd_run_algo, "#E74C3C")
        self.btn_export = self._btn(ctrl, "Xuất kết quả",         self.cmd_export,   "#7F8C8D")

        # Trạng thái
        st = self._section(parent, "Trạng thái")
        self.stat_var = tk.StringVar(value="Sẵn sàng.")
        tk.Label(st, textvariable=self.stat_var, bg="#2C3E50", fg="#2ECC71",
                 font=("Courier", 9), justify=tk.LEFT).pack(anchor=tk.W)

        self._on_algo_change()   # khởi tạo trạng thái panel tham số

    def _build_right(self, parent):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#2C3E50", borderwidth=0)
        style.configure("TNotebook.Tab", background="#34495E", foreground="#ECF0F1",
                        font=("Segoe UI", 9, "bold"), padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", "#2980B9")])

        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill=tk.BOTH, expand=True)

        chart_tab = tk.Frame(self.nb, bg="#1E2B38")
        self.nb.add(chart_tab, text="Biểu đồ ")

        self.fig = Figure(figsize=(7, 5), facecolor="#1E2B38")
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        log_tab = tk.Frame(self.nb, bg="#1A252F")
        self.nb.add(log_tab, text="Nhật ký ")
        self.log_text = tk.Text(log_tab, bg="#0D1B2A", fg="#2ECC71",
                                font=("Courier New", 9), bd=0, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Algo switcher
    # ------------------------------------------------------------------

    def _on_algo_change(self):
        alg = self.algo_var.get()
        if alg == "CLIQUE":
            self.frm_gcbd.pack_forget()
            self.frm_clique.pack(fill=tk.X)
            self.algo_desc_var.set(
                "CLIQUE: chia lưới k×k, đếm điểm mỗi ô, gom ô ≥ MinPts liền kề."
            )
        else:
            self.frm_clique.pack_forget()
            self.frm_gcbd.pack(fill=tk.X)
            self.algo_desc_var.set(
                "GCBD: mật độ tại node lưới (bilinear), loại biên lặp T vòng "
                "(percentile-10), gán boundary theo hàng xóm mật độ cao nhất. "
                "Không cần đặt MinPts."
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self._log_buffer.append(msg)

    def _flush_log(self):
        if not self._log_buffer:
            return
        self.log_text.configure(state=tk.NORMAL)
        for line in self._log_buffer:
            self.log_text.insert(tk.END, line + "\n")
        self._log_buffer.clear()
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _setup_ax(self, title: str):
        self.ax.clear()
        self.ax.set_facecolor("#1E2B38")
        self.ax.tick_params(colors="#BDC3C7", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#4A6274")
        self.ax.set_title(title, color="#ECF0F1", fontsize=10)

    def _add_legend(self, n_clusters: int):
        handles = [mpatches.Patch(color=_cluster_color(c, n_clusters),
                                  label=f"Cụm {c}") for c in range(n_clusters)]
        handles.append(mpatches.Patch(color="#7F8C8D", label="Noise"))
        self.ax.legend(handles=handles, loc="upper right", fontsize=7,
                       facecolor="#2C3E50", labelcolor="#ECF0F1", framealpha=0.8)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _plot_sample(self, max_pts: int = 20_000):
        """
        Visual Downsampling: thuật toán chạy trên 100% dữ liệu,
        nhưng Matplotlib chỉ vẽ tối đa max_pts điểm đại diện.
        Trả về (X_plot, labels_plot | None, sampled: bool).
        """
        n = len(self.X)
        if n <= max_pts:
            lbl = self.model.labels_ if self.model is not None else None
            return self.X, lbl, False
        idx = np.random.choice(n, max_pts, replace=False)
        lbl = self.model.labels_[idx] if self.model is not None else None
        return self.X[idx], lbl, True

    def _validate_clique(self):
        try:
            k  = self.k_var.get()
            xi = self.xi_var.get()
            if k <= 1 or xi < 1:
                raise ValueError
            return k, xi
        except (tk.TclError, ValueError):
            messagebox.showerror("Lỗi tham số",
                                 "k phải là số nguyên > 1 và MinPts >= 1.")
            return None, None

    def _validate_gcbd(self):
        try:
            l = self.l_var.get()
            T = self.T_var.get()
            if l < 2 or T < 1:
                raise ValueError
            return l, T
        except (tk.TclError, ValueError):
            messagebox.showerror("Lỗi tham số",
                                 "l phải là số nguyên >= 2 và T >= 1.")
            return None, None

    def _set_controls(self, state: str):
        for b in (self.btn_gen, self.btn_load, self.btn_grid,
                  self.btn_run, self.btn_export):
            b.configure(state=state)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def cmd_generate_data(self):
        try:
            n   = self.n_samples_var.get()
            c   = self.n_centers_var.get()
            std = self.std_var.get()
        except tk.TclError:
            messagebox.showerror("Lỗi nhập liệu", "Vui lòng nhập giá trị hợp lệ.")
            return
        if n < 10 or c < 1:
            messagebox.showerror("Lỗi logic", "Số điểm >= 10, số cụm >= 1.")
            return
        if std <= 0:
            messagebox.showerror("Lỗi logic", "std phải > 0.")
            return

        self.X = generate_mock_data(n, c, std)
        self.model = None
        X_plot, _, sampled = self._plot_sample()
        title = f"Dữ liệu ban đầu" + (f" (hiển thị {len(X_plot):,}/{len(self.X):,})" if sampled else "")
        self._setup_ax(title)
        self.ax.scatter(X_plot[:, 0], X_plot[:, 1], s=12, c="#3498DB", alpha=0.7, edgecolors="none")
        self.canvas.draw()
        self.stat_var.set(f"Đã tạo {len(self.X):,} điểm.")
        self._log(f"[Dữ liệu] {len(self.X):,} điểm ngẫu nhiên (std={std}).")
        self._flush_log()

    def cmd_load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.X = load_data_from_csv(path)
        except (ValueError, OSError) as e:
            messagebox.showerror("Lỗi nạp dữ liệu", str(e))
            return
        if len(self.X) < 2:
            messagebox.showerror("Lỗi", "Tập dữ liệu quá nhỏ.")
            return

        self.model = None
        X_plot, _, sampled = self._plot_sample()
        title = f"Dữ liệu: {os.path.basename(path)}" + (f" (hiển thị {len(X_plot):,}/{len(self.X):,})" if sampled else "")
        self._setup_ax(title)
        self.ax.scatter(X_plot[:, 0], X_plot[:, 1], s=12, c="#3498DB", alpha=0.7, edgecolors="none")
        self.canvas.draw()
        self.stat_var.set(f"Đã tải {len(self.X):,} điểm.")
        self._log(f"[File] {len(self.X):,} điểm từ CSV.")
        self._flush_log()

    def cmd_show_grid(self):
        if self.X is None:
            return
        alg = self.algo_var.get()

        # Tái dụng model nếu đã có cùng thuật toán + tham số
        if (self.model is not None
                and getattr(self.model, '_algo_tag', None) == alg
                and self.model.density_grid_ is not None):
            m = self.model
            self._draw_grid(m, alg)
            return

        if alg == "CLIQUE":
            k, xi = self._validate_clique()
            if k is None: return
            m = CLIQUEAlgorithm(self.X, k, xi, self._log)
            m.prepare_grid()
            self._flush_log()
            self._draw_grid_clique(m)
        else:
            l, T = self._validate_gcbd()
            if l is None: return
            m = GCBDAlgorithm(self.X, l, T, self._log)
            m.prepare_grid()
            self._flush_log()
            self._draw_grid_gcbd(m)

    def _draw_grid(self, m, alg: str):
        if alg == "CLIQUE":
            self._draw_grid_clique(m)
        else:
            self._draw_grid_gcbd(m)

    def _draw_grid_clique(self, m: CLIQUEAlgorithm):
        k, xi = m.k, m.xi
        xe, ye, H = m.x_edges_, m.y_edges_, m.density_grid_
        self._setup_ax(f"[CLIQUE] Lưới {k}×{k} – ô cam đạt MinPts={xi}")
        for i in range(k):
            for j in range(k):
                if H[i, j] >= xi:
                    self.ax.add_patch(mpatches.Rectangle(
                        (xe[i], ye[j]), xe[i+1]-xe[i], ye[j+1]-ye[j],
                        facecolor="#F39C12", alpha=0.4))
                    self.ax.text((xe[i]+xe[i+1])/2, (ye[j]+ye[j+1])/2,
                                 str(int(H[i, j])), color="#F1C40F",
                                 ha="center", va="center", fontsize=7)
        for x in xe: self.ax.axvline(x, color="#4A6274", lw=0.5, ls="--")
        for y in ye: self.ax.axhline(y, color="#4A6274", lw=0.5, ls="--")
        X_plot, _, _ = self._plot_sample()
        self.ax.scatter(X_plot[:, 0], X_plot[:, 1], s=10, c="#3498DB", alpha=0.6, edgecolors="none")
        self.canvas.draw()
        self.nb.select(0)

    def _draw_grid_gcbd(self, m: GCBDAlgorithm):
        """
        Vẽ lưới GCBD: scatter-plot mật độ node bằng kích thước/màu điểm.
        Node có rho > 0 được hiển thị; node có rho cao hơn thì lớn hơn.
        """
        xe, ye = m.x_edges_, m.y_edges_
        H = m.density_grid_
        self._setup_ax(f"[GCBD] Lưới {m.l}×{m.l} – mật độ node (bilinear)")

        # Vẽ grid lines
        for x in xe: self.ax.axvline(x, color="#4A6274", lw=0.4, ls="--")
        for y in ye: self.ax.axhline(y, color="#4A6274", lw=0.4, ls="--")

        # Vẽ node theo mật độ
        ni_arr, nj_arr = np.where(H > 0)
        if len(ni_arr) > 0:
            rho_vals = H[ni_arr, nj_arr]
            rho_norm = rho_vals / rho_vals.max()
            node_x_coords = xe[ni_arr]
            node_y_coords = ye[nj_arr]
            self.ax.scatter(node_x_coords, node_y_coords,
                            s=rho_norm * 120 + 10,
                            c=rho_vals, cmap="YlOrRd", alpha=0.8,
                            edgecolors="none", zorder=3)

        X_plot, _, _ = self._plot_sample()
        self.ax.scatter(X_plot[:, 0], X_plot[:, 1], s=8, c="#3498DB", alpha=0.4,
                        edgecolors="none", zorder=2)
        self.canvas.draw()
        self.nb.select(0)

    def cmd_run_algo(self):
        if self.X is None:
            return
        alg = self.algo_var.get()

        if alg == "CLIQUE":
            k, xi = self._validate_clique()
            if k is None: return
        else:
            l, T = self._validate_gcbd()
            if l is None: return

        # Xóa log cũ
        self.nb.select(1)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._log_buffer.clear()

        self._set_controls(tk.DISABLED)
        self.stat_var.set("Đang chạy thuật toán...")
        X_snap = self.X.copy()

        def _worker():
            if alg == "CLIQUE":
                model = CLIQUEAlgorithm(X_snap, k, xi, self._log).fit()
            else:
                model = GCBDAlgorithm(X_snap, l, T, self._log).fit()
            model._algo_tag = alg
            self.root.after(0, lambda: _done(model))

        def _done(model):
            self.model = model
            self._flush_log()
            self._draw_result(model, alg)
            self._set_controls(tk.NORMAL)
            noise = int(np.sum(model.labels_ == -1))
            self.stat_var.set(
                f"[{alg}] {model.n_clusters_} cụm | "
                f"{len(self.X)-noise:,} điểm thuộc cụm | "
                f"{noise:,} noise (tổng {len(self.X):,})"
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _draw_result(self, model, alg: str):
        """Vẽ kết quả phân cụm với visual downsampling."""
        n = model.n_clusters_
        xe, ye = model.x_edges_, model.y_edges_
        X_plot, labels_plot, sampled = self._plot_sample()
        sample_note = f" — hiển thị {len(X_plot):,}/{len(self.X):,} điểm" if sampled else ""
        self._setup_ax(f"[{alg}] Kết quả: {n} cụm{sample_note}")

        # Tô vùng core cluster (luôn dùng toàn bộ cell/node – nhanh vì số cell nhỏ)
        if alg == "CLIQUE":
            for cid, cells in enumerate(model.cluster_cells_):
                col = _cluster_color(cid, n)
                for (ci, cj) in cells:
                    self.ax.add_patch(mpatches.Rectangle(
                        (xe[ci], ye[cj]), xe[ci+1]-xe[ci], ye[cj+1]-ye[cj],
                        facecolor=col, alpha=0.18))
        else:
            dx = (xe[-1] - xe[0]) / model.l * 0.5
            dy = (ye[-1] - ye[0]) / model.l * 0.5
            for cid, cells in enumerate(model.cluster_cells_):
                col = _cluster_color(cid, n)
                for (ni, nj) in cells:
                    cx, cy = xe[ni], ye[nj]
                    self.ax.add_patch(mpatches.Rectangle(
                        (cx - dx, cy - dy), 2*dx, 2*dy,
                        facecolor=col, alpha=0.18))

        # Scatter chỉ trên tập mẫu (tránh crash Matplotlib với 1M điểm)
        for cid in range(n):
            mask = labels_plot == cid
            if mask.any():
                self.ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                                s=14, color=_cluster_color(cid, n), edgecolors="none")

        noise = labels_plot == -1
        if noise.any():
            self.ax.scatter(X_plot[noise, 0], X_plot[noise, 1],
                            s=10, color="#7F8C8D", marker="x")

        self._add_legend(n)
        self.canvas.draw()
        self.nb.select(0)

    def cmd_export(self):
        if self.model is None:
            messagebox.showwarning("Chưa có kết quả", "Vui lòng chạy phân cụm trước.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="ket_qua.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "cluster"])
                for (x, y), lbl in zip(self.X, self.model.labels_):
                    w.writerow([round(x, 4), round(y, 4), int(lbl)])
            messagebox.showinfo("Thành công", f"Đã lưu:\n{path}")
        except OSError as e:
            messagebox.showerror("Lỗi lưu file", str(e))