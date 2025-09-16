import os
import shutil
import glob
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import joblib  
import torch
import torch.nn as nn
import customtkinter as ctk
from tkinter import filedialog, messagebox
from spectral import envi

from src.preprocessing import Preprocessing
from src.utils import save_predictions_by_source, calculate_rmse_by_source, print_avg_predicted_ratios
from models.module import SimpleCNN_MLP


# 設定字型來支援中文字符
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 或者使用其他支援中文的字型
matplotlib.rcParams["axes.unicode_minus"] = False  # 確保負號能正確顯示

# ================================
# 基本設定
# ================================
matplotlib.use("TkAgg")
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

UPLOAD_DIR = "./uploaded"
OUTPUT_DIR = "./data"
OUTPUT_NPY_PREPROCESSING_DIR = "./preprocessing_data"
RESULT_DIR = "./result"  # 新增 result 資料夾路徑
WEIGHT_PATH = "./weight/SimpleCNN_MLP_final.pt"
PCA_PATH = "./weight/pca_model.pkl"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_NPY_PREPROCESSING_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)  # 確保 result 資料夾存在

x, y = 333, 220
fixed_width = 25
fixed_height = 300

# ================================
# SAM Function
# ================================
def sam(hsi_cube, d_point):
    h, w, b = hsi_cube.shape
    r = hsi_cube.reshape(-1, b).T
    rd = np.dot(d_point, r)
    r_abs = np.linalg.norm(r, axis=0)
    d_abs = np.linalg.norm(d_point)
    tmp = rd / (r_abs * d_abs + 1e-8)
    tmp = np.clip(tmp, -1.0, 1.0)
    sam_rd = np.arccos(tmp)
    return sam_rd.reshape(h, w)

# ================================
# 上傳檔案 & ROI 擷取
# ================================
def process_roi_for_uploaded_hdrs(progress, total_files):
    processed = 0
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".hdr"):
            hdr_path = os.path.join(UPLOAD_DIR, filename)
            label_result.configure(text=f"處理中: {filename}")
            app.update_idletasks()

            data = envi.open(hdr_path)
            np_data = np.asarray(data.open_memmap(writable=True))
            d_point = np_data[y, x, :]
            sam_map = sam(np_data, d_point)

            fig, ax = plt.subplots()
            img = ax.imshow(sam_map, cmap="jet")
            plt.colorbar(img)
            ax.set_title(f"SAM Map - {filename}\n請框選 ROI ({fixed_width}x{fixed_height})\n(Ctrl+滾輪向下放大)")

            rect_coords = {}

            def onselect(eclick, erelease):
                x1, y1 = int(eclick.xdata), int(eclick.ydata)
                x2 = x1 + fixed_width
                y2 = y1 + fixed_height

                if x2 > sam_map.shape[1]:
                    x1 = sam_map.shape[1] - fixed_width
                    x2 = sam_map.shape[1]
                if y2 > sam_map.shape[0]:
                    y1 = sam_map.shape[0] - fixed_height
                    y2 = sam_map.shape[0]

                rect_coords["x1"], rect_coords["y1"] = x1, y1
                rect_coords["x2"], rect_coords["y2"] = x2, y2

                rect = plt.Rectangle((x1, y1), fixed_width, fixed_height, edgecolor="red", facecolor="none", lw=2)
                ax.add_patch(rect)
                fig.canvas.draw()

            # 啟用 Ctrl + 滾輪縮放（向上放大）
            def on_scroll(event):
                if event.inaxes and (event.guiEvent.state & 4):  # 檢查 Ctrl 鍵 (state & 4 表示 Ctrl)
                    scale_factor = 1.1 if event.button == 'up' else 0.9 if event.button == 'down' else 1
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:  # 確保座標有效
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        ax.set_xlim(xlim[0] + (xlim[0] - x) * (1 - 1/scale_factor), 
                                  xlim[1] + (xlim[1] - x) * (1 - 1/scale_factor))
                        ax.set_ylim(ylim[0] + (ylim[0] - y) * (1 - 1/scale_factor), 
                                  ylim[1] + (ylim[1] - y) * (1 - 1/scale_factor))
                        fig.canvas.draw()

            fig.canvas.mpl_connect('scroll_event', on_scroll)

            toggle_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], spancoords="pixels", interactive=False)
            
            # 最大化視窗
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed')  # 最大化視窗（Windows 環境）
            plt.show()  # 僅用於 SAM 圖，阻塞直到關閉

            if rect_coords:
                x1, y1 = rect_coords["x1"], rect_coords["y1"]
                x2, y2 = rect_coords["x2"], rect_coords["y2"]
                mask = np.zeros(sam_map.shape, dtype=bool)
                mask[y1:y2, x1:x2] = True
                roi_spectra = np_data[mask, :]
                npy_name = filename.replace(".hdr", f"_roi_{fixed_width}x{fixed_height}.npy")
                np.save(os.path.join(OUTPUT_DIR, npy_name), roi_spectra)

                # 清除 frame_spectra 中的舊內容，顯示圓餅圖
                for widget in frame_spectra.winfo_children():
                    widget.destroy()
                label_result.configure(text=f"正在推論: {filename}")
                app.update_idletasks()
                display_ratio_and_rmse("SNV_PCABANDSELECT", 50)

            processed += 1
            progress.set(int((processed / total_files) * 100))
            app.update_idletasks()

    # 所有檔案處理完成後顯示提示視窗一次
    label_result.configure(text="✅ 所有檔案處理完成！")
    progress_bar.set(100)
    messagebox.showinfo("完成", "全部 HDR 已完成 ROI 處理並儲存為 .npy")

def upload_and_process_files():
    file_paths = filedialog.askopenfilenames(title="選擇 HDR + RAW 檔", filetypes=[("HDR/RAW files", "*.hdr *.raw")])
    if not file_paths:
        messagebox.showerror("錯誤", "未選擇任何檔案")
        return

    hdr_files = [f for f in file_paths if f.endswith(".hdr")]
    raw_files = [f for f in file_paths if f.endswith(".raw")]
    raw_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in raw_files}
    paired_files = []

    for hdr in hdr_files:
        base = os.path.splitext(os.path.basename(hdr))[0]
        if base in raw_dict:
            paired_files.append((hdr, raw_dict[base]))
        else:
            messagebox.showerror("錯誤", f"缺少與 {base}.hdr 配對的 .raw 檔案")
            return

    if not paired_files:
        messagebox.showerror("錯誤", "未找到可配對的檔案")
        return

    label_result.configure(text="開始上傳檔案...")
    progress_bar.set(0)
    app.update_idletasks()

    for i, (hdr, raw) in enumerate(paired_files, start=1):
        shutil.copy(hdr, os.path.join(UPLOAD_DIR, os.path.basename(hdr)))
        shutil.copy(raw, os.path.join(UPLOAD_DIR, os.path.basename(raw)))
        percent = int((i / len(paired_files)) * 100)
        progress_bar.set(percent)
        label_result.configure(text=f"上傳中: {i}/{len(paired_files)} 組（{percent}%）")
        app.update_idletasks()

    label_result.configure(text="✅ 上傳完成，開始 ROI 分析...")
    process_roi_for_uploaded_hdrs(progress_bar, len(paired_files))

# ================================
# 推論 (載入權重)
# ================================
def run_model_inference(mode: str, band_num: int):
    preprocessin_data_path = "preprocessing_data"
    amplification_factor = 2.0

    if os.path.exists(preprocessin_data_path):
        shutil.rmtree(preprocessin_data_path)
    os.makedirs(preprocessin_data_path, exist_ok=True)

    file_paths = sorted(glob.glob(f"./{OUTPUT_DIR}/*.npy"))
    if not file_paths:
        messagebox.showerror("錯誤", "找不到 ROI npy 檔案")
        return {}, {}

    preprocessing_paths = [os.path.join(preprocessin_data_path, os.path.basename(path)) for path in file_paths]

    # ★ 載入訓練時的 PCA 模型
    if os.path.exists(PCA_PATH):
        pca = joblib.load(PCA_PATH)
        preprocessor = Preprocessing(
            n_components=band_num,
            mode=mode,
            amplification_factor=amplification_factor,
            pca_model=pca
        )
    else:
        messagebox.showerror("錯誤", "找不到 PCA 模型，請先確認訓練時已存下 pca_model.pkl")
        return {}, {}

    # 用舊的 PCA 做前處理
    preprocessor.preprocess(file_paths, preprocessing_paths)

    # === 準備輸入資料 ===
    test_X_list, test_sources = [], []
    for path in preprocessing_paths:
        data = np.load(path)
        
        # 👉 取 ROI 平均光譜，保持和訓練資料一致
        data_mean = np.mean(data, axis=0, keepdims=True)  # shape = (1, bands)

        X = torch.tensor(data_mean, dtype=torch.float32).unsqueeze(1)  # shape = (1, 1, bands)
        test_X_list.append(X)
        test_sources.append(path)  # 每個檔案只對應一個平均光譜

    test_X = torch.cat(test_X_list, dim=0)

    # === 載入模型權重 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN_MLP(input_channels=1, input_dim=band_num, hidden_dim=64, output_dim=2)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # === 推論 ===
    with torch.no_grad():
        pred_Y = model(test_X.to(device)).cpu()

    dummy_Y = torch.zeros_like(pred_Y)  # 沒有真值，只放 placeholder
    results_by_source = save_predictions_by_source(test_sources, pred_Y, dummy_Y)

    print_avg_predicted_ratios(results_by_source)
    return results_by_source, {}

# ================================
# UI 顯示比例 & RMSE
# ================================
def display_ratio_and_rmse(mode, band_num):
    results_by_source, rmse_by_source = run_model_inference(mode, band_num)
    if not results_by_source:
        return

    # 清空之前的圖表
    for widget in frame_spectra.winfo_children():
        widget.destroy()

    sources = list(results_by_source.keys())
    cotton_vals, poly_vals = [], []
    for src, records in results_by_source.items():
        cotton_vals.append(np.mean([r["Predicted_cotton"] for r in records]) * 100)
        poly_vals.append(np.mean([r["Predicted_poly"] for r in records]) * 100)

    # 顯示圓環圖，確保不跳出視窗
    fig1, ax1 = plt.subplots()
    ax1.pie([np.mean(cotton_vals), np.mean(poly_vals)], labels=["Cotton", "Polyester"], autopct="%1.2f%%", startangle=90)
    ax1.axis('equal')  # 確保圓形顯示
    canvas1 = FigureCanvasTkAgg(fig1, master=frame_spectra)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="left", expand=True, fill="both")
    plt.close(fig1)  # 關閉圖形以避免潛在的視窗問題

# ================================
# 清除資料夾
# ================================
def clear_all_data():
    try:
        for folder in [UPLOAD_DIR, OUTPUT_DIR, OUTPUT_NPY_PREPROCESSING_DIR, RESULT_DIR]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"⚠️ 無法刪除 {file_path}: {e}")
        messagebox.showinfo("完成", "已清除 upload, data, preprocessing_data, result 資料")
        label_result.configure(text="📂 已清空所有資料夾")
        progress_bar.set(0)
    except Exception as e:
        messagebox.showerror("錯誤", f"清除資料時發生問題: {e}")

# ================================
# 主 UI
# ================================
app = ctk.CTk()
app.title("紗線高光譜辨識系統")
app.geometry("1600x1000")

frame_main = ctk.CTkFrame(app)
frame_main.pack(fill="both", expand=True, padx=20, pady=20)

# 使用 frame 來水平排列按鈕
button_frame = ctk.CTkFrame(frame_main)
button_frame.pack(pady=10)

btn_upload = ctk.CTkButton(button_frame, text="上傳並 ROI", command=upload_and_process_files, width=200, height=50,
                           font=ctk.CTkFont(size=18, weight="bold"), text_color="black")
btn_upload.pack(side="left", padx=10)

btn_clear = ctk.CTkButton(button_frame, text="清除資料", command=clear_all_data, width=200, height=50,
                          font=ctk.CTkFont(size=18, weight="bold"), text_color="black")
btn_clear.pack(side="left", padx=10)

global label_result, progress_bar, frame_spectra
label_result = ctk.CTkLabel(frame_main, text="尚未處理任何檔案", font=ctk.CTkFont(size=18, weight="bold"))
label_result.pack(pady=10)

progress_bar = ctk.CTkProgressBar(frame_main, width=400)
progress_bar.set(0)
progress_bar.pack(pady=10)

frame_spectra = ctk.CTkFrame(frame_main)
frame_spectra.pack(fill="both", expand=True, padx=10, pady=10)

app.mainloop()