import numpy as np
from sklearn.decomposition import PCA, NMF
from src.utils import flatten_3d_to_2d
import joblib
class Preprocessing:
    def __init__(self, n_components=22, mode="PCA_BANDSELECT_AMPLIFY_ERROR", amplification_factor=2.0, pca_model=None):
        self.n_components = n_components
        self.mode = mode
        self.amplification_factor = amplification_factor
        self.pca_model = pca_model
        if self.mode in ("NMF", "NMF_AMPLIFY_ERROR"):
            self.model = NMF(
                n_components=self.n_components,
                init='nndsvd',
                tol=1e-5,
                max_iter=1000,
                random_state=42
            )

    def load_and_flatten(self, file_paths):
        """
        Load each .npy file. If 3D, flatten to 2D; if already 2D, keep as is.
        Returns a list of 2D arrays.
        """
        data = []
        for path in file_paths:
            array = np.load(path)
            if array.ndim == 3:
                flattened = flatten_3d_to_2d(array)
            elif array.ndim == 2:
                flattened = array
            else:
                raise ValueError(f"Input array at {path} must be 2D or 3D. Found ndim={array.ndim}.")
            data.append(flattened)
        return data

    def amplify_error_and_normalize(self, data_list, k=2.0):
        combined = np.vstack(data_list)
        mean_vec = np.mean(combined, axis=0)
        amplified = k * (combined - mean_vec)
        min_val, max_val = amplified.min(), amplified.max()
        normalized = (amplified - min_val) / (max_val - min_val)

        result = []
        start = 0
        for arr in data_list:
            rows = arr.shape[0]
            result.append(normalized[start:start + rows])
            start += rows
        return result

    def snv(self, input_data):
        '''
        input_data : hyperspectral data can be
                    3d [width , height , bands]
                    2d [average spectral samples, bands]
        '''
        d = len(input_data.shape) - 1
        mean = input_data.mean(d)
        std = input_data.std(d)
        res = (input_data - mean[..., None]) / std[..., None]
        return res
    # def snv(self, X):
    #     """
    #     對光譜資料 X 套用 SNV (Standard Normal Variate) 正規化
    #
    #     參數:
    #         X: ndarray, shape = (n_samples, n_bands)
    #             原始光譜資料，n_samples 是樣本數，n_bands 是波段數
    #
    #     回傳:
    #         X_snv: ndarray, shape = (n_samples, n_bands)
    #             經過 SNV 正規化後的光譜資料
    #     """
    #     # Step 1: 對每條光譜進行 SNV 正規化
    #     X_snv = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    #     return X_snv

    def pca_band_selection(self, data_list, num_bands=None, save_path=None):
        """
        Select top bands based on PCA on 2D data_list.
        If self.pca_model exists, use it instead of fitting a new one.
        """
        if num_bands is None:
            num_bands = self.n_components

        # 對齊 band 維度
        dims = [arr.shape[1] for arr in data_list]
        if len(set(dims)) > 1:
            min_dim = min(dims)
            print(f"Inconsistent band dimensions {dims}, truncating to {min_dim} bands.")
            data_list = [arr[:, :min_dim] for arr in data_list]

        if self.pca_model is None:  # 訓練模式：fit 新的 PCA
            combined = np.vstack(data_list)
            self.pca_model = PCA(n_components=combined.shape[1])
            self.pca_model.fit(combined)

            # ★ 存檔
            if save_path is not None:
                joblib.dump(self.pca_model, save_path)
                print(f"✅ PCA 模型已存到 {save_path}")

        # 使用已有的 PCA model
        comp = np.abs(self.pca_model.components_)
        contrib = comp[:3].sum(axis=0)
        idx = np.argsort(contrib)[::-1][:num_bands]
        idx.sort()
        print(f"Selected bands (1-based): {(idx + 1)}")
        return [arr[:, idx] for arr in data_list]

    def apply_pca(self, data_list, save_path=None):
        """
        Apply PCA reduction on 2D data_list.
        If self.pca_model exists, use it; otherwise fit a new one.
        """
        dims = [arr.shape[1] for arr in data_list]
        if len(set(dims)) > 1:
            min_dim = min(dims)
            print(f"Inconsistent band dims {dims}, truncating to {min_dim}.")
            data_list = [arr[:, :min_dim] for arr in data_list]

        combined = np.maximum(np.vstack(data_list), 0)
        if self.pca_model is None:  # 訓練模式
            self.pca_model = PCA(n_components=self.n_components)
            self.pca_model.fit(combined)
            if save_path is not None:
                joblib.dump(self.pca_model, save_path)
                print(f"✅ PCA 模型已存到 {save_path}")

        return [self.pca_model.transform(arr) + 1 for arr in data_list]

    def apply_nmf(self, data_list):
        """
        Apply NMF reduction on 2D data_list. Ensures uniform dims.
        """
        dims = [arr.shape[1] for arr in data_list]
        if len(set(dims)) > 1:
            min_dim = min(dims)
            print(f"Inconsistent band dims {dims}, truncating to {min_dim}.")
            data_list = [arr[:, :min_dim] for arr in data_list]
        combined = np.vstack(data_list)
        self.model.fit(combined)
        return [self.model.transform(arr) * 5 for arr in data_list]

    def save_data(self, data_list, save_paths):
        for arr, path in zip(data_list, save_paths):
            np.save(path, arr)

    def first_derivative(self, X):
        """
        計算光譜資料 X 的一階導數

        參數:
            X: ndarray, shape = (n_samples, n_bands)
                原始光譜資料，n_samples 是樣本數，n_bands 是波段數

        回傳:
            X_deriv: ndarray, shape = (n_samples, n_bands-1)
                經過一階導數處理後的光譜資料
        """
        # 一階微分
        X_deriv = np.diff(X, axis=1)

        # Min-Max normalization (逐 row 正規化每個樣本)
        X_min = X_deriv.min(axis=1, keepdims=True)
        X_max = X_deriv.max(axis=1, keepdims=True)
        X_normalized = (X_deriv - X_min) / (X_max - X_min + 1e-8)  # 加上 1e-8 防止除以 0

        return X_normalized

    def preprocess(self, file_paths, save_paths):
        data_list = self.load_and_flatten(file_paths)
        if self.mode == "NMF":
            result = self.apply_nmf(data_list)

        if self.mode == "DERIVATIVE":
            print("Applying first derivative preprocessing...")
            data_list = [self.first_derivative(data) for data in data_list]
            result = data_list
        elif self.mode == "PCA_BANDSELECT":
            result = self.pca_band_selection(data_list)
        elif self.mode == "NMF_AMPLIFY_ERROR":
            result = self.apply_nmf(data_list)
            result = self.amplify_error_and_normalize(result, k=self.amplification_factor)
        elif self.mode == "SNV_PCABANDSELECT":  # 新增模式
            print("Applying SNV followed by PCA_BANDSELECT preprocessing...")
            # 先執行 SNV
            snv_data = [self.snv(data) for data in data_list]
            # 再執行 PCA_BANDSELECT
            result = self.pca_band_selection(snv_data, save_path="./weight/pca_model.pkl")
        elif self.mode == "SNV":
            print("Applying SNV preprocessing...")
            result = [self.snv(data) for data in data_list]
        elif self.mode == "PCA_BANDSELECT_AMPLIFY_ERROR":
            result = self.pca_band_selection(data_list)
            result = self.amplify_error_and_normalize(result, k=self.amplification_factor)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        self.save_data(result, save_paths)


