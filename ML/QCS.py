import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from scipy.stats import zscore
import os
import cv2


class DatasetAnalyzer:
    def __init__(self, data, target=None, data_type="tabular", sample_size=50):
        self.data = data
        self.target = target
        self.data_type = data_type
        self.sample_size = sample_size
        self.analysis = {}

    def analyze(self):
        if self.data_type == "tabular":
            self._analyze_tabular()
        elif self.data_type == "image":
            self._analyze_image()
        elif self.data_type == "video":
            self._analyze_video()
        else:
            raise ValueError("Unsupported data_type")
        return self.analysis

    # ========== TABULAR ANALYSIS ==========
    def _analyze_tabular(self):
        df = self.data
        n_samples, n_features = df.shape
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        numeric_cols = df.select_dtypes(include=["number"]).columns

        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        explained_var = 0
        if len(numeric_cols) > 1:
            try:
                pca = PCA(n_components=min(5, len(numeric_cols)))
                pca.fit(StandardScaler().fit_transform(df_encoded[numeric_cols]))
                explained_var = np.sum(pca.explained_variance_ratio_)
            except Exception:
                pass

        avg_mi = 0
        if self.target and self.target in df.columns:
            y = df[self.target]
            X = df.drop(columns=[self.target])
            X_enc = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)) if col.dtype == "object" else col)
            try:
                mi = mutual_info_classif(X_enc, y, discrete_features="auto")
                avg_mi = np.mean(mi)
            except Exception:
                pass

        noise_score = self._estimate_noise(df_encoded[numeric_cols]) if len(numeric_cols) > 0 else 0
        consistency_score = 1 - noise_score
        corr_redundancy = self._correlation_redundancy(df_encoded[numeric_cols]) if len(numeric_cols) > 1 else 0
        missing_ratio = df.isna().sum().sum() / (n_samples * n_features)
        outlier_fraction = self._outlier_fraction(df_encoded[numeric_cols]) if len(numeric_cols) > 0 else 0

        self.analysis = {
            "type": "tabular",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_categorical": len(categorical_cols),
            "n_numeric": len(numeric_cols),
            "explained_var_pca": explained_var,
            "avg_mutual_info": avg_mi,
            "noise_score": noise_score,
            "consistency_score": consistency_score,
            "correlation_redundancy": corr_redundancy,
            "missing_ratio": missing_ratio,
            "outlier_fraction": outlier_fraction
        }

    # ========== IMAGE ANALYSIS ==========
    def _analyze_image(self):
        image_dir = self.data
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            self.analysis = {"type": "image", "n_samples": 0}
            return

        sizes, brightness, contrast = [], [], []
        for f in files[:self.sample_size]:
            img = cv2.imread(os.path.join(image_dir, f))
            if img is not None:
                sizes.append(img.shape)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness.append(np.mean(gray))
                contrast.append(np.std(gray))

        if sizes:
            avg_h = np.mean([s[0] for s in sizes])
            avg_w = np.mean([s[1] for s in sizes])
            channels = sizes[0][2] if len(sizes[0]) > 2 else 1
            avg_brightness = np.mean(brightness)
            avg_contrast = np.mean(contrast)
        else:
            avg_h = avg_w = channels = avg_brightness = avg_contrast = 0

        self.analysis = {
            "type": "image",
            "n_samples": len(files),
            "avg_height": avg_h,
            "avg_width": avg_w,
            "channels": channels,
            "avg_brightness": avg_brightness,
            "avg_contrast": avg_contrast
        }

    def _analyze_video(self):
        self.analysis = {"type": "video", "note": "Video analysis not yet implemented"}

    # ========== HELPERS ==========
    def _estimate_noise(self, df_num):
        try:
            z = np.abs(zscore(df_num))
            noise = np.mean(np.std(z, axis=0))
            return min(noise / 5, 1)
        except Exception:
            return 0

    def _correlation_redundancy(self, df_num):
        corr = df_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        redundant = (upper > 0.8).sum().sum()
        redundancy_ratio = redundant / (len(df_num.columns) * (len(df_num.columns)-1)/2)
        return redundancy_ratio

    def _outlier_fraction(self, df_num):
        z = np.abs(zscore(df_num))
        return np.mean(z > 3)


# =====================================================================
class ModelSelector:
    def __init__(self, analysis, thresholds=None):
        self.analysis = analysis
        self.thresholds = thresholds or {
            "high_dim": 50,
            "high_sample": 10000,
            "high_mi": 0.05,
            "low_noise": 0.3,
            "low_missing": 0.1
        }

    def recommend(self):
        dtype = self.analysis["type"]
        scores = {m: 0.0 for m in ["mlp", "cnn", "svm", "vqc", "qkernel", "qnn"]}
        rationale = []

        # ========== TABULAR ==========
        if dtype == "tabular":
            n_features = self.analysis["n_features"]
            n_samples = self.analysis["n_samples"]
            mi = self.analysis["avg_mutual_info"]
            noise = self.analysis["noise_score"]
            missing = self.analysis["missing_ratio"]
            explained = self.analysis["explained_var_pca"]

            # scoring logic
            scores["svm"] += 0.6 if n_samples < 500 else 0.2
            scores["mlp"] += 0.8 if n_features < self.thresholds["high_dim"] else 0.4
            scores["mlp"] += 0.2 if noise < self.thresholds["low_noise"] else 0
            scores["qkernel"] += 0.7 if noise > 0.5 or missing > self.thresholds["low_missing"] else 0.3
            scores["vqc"] += 0.6 if explained > 0.7 else 0.2
            scores["qnn"] += 0.7 if n_samples > self.thresholds["high_sample"] else 0.3

            rationale.append("Model scores computed from data dimensionality, noise, and structure.")

        # ========== IMAGE ==========
        elif dtype == "image":
            h, w = self.analysis["avg_height"], self.analysis["avg_width"]
            if h > 28 and w > 28:
                scores["cnn"] += 0.9
                scores["mlp"] += 0.5
            else:
                scores["mlp"] += 0.8
                scores["cnn"] += 0.4
            scores["vqc"] += 0.4
            rationale.append("Image size used to favor CNNs for spatial structure.")

        # ========== VIDEO ==========
        elif dtype == "video":
            scores["qnn"] = 0.9
            scores["cnn"] = 0.6
            rationale.append("Video data benefits from temporal-spatial modeling — QNN favored.")

        # normalize
        max_score = max(scores.values()) or 1
        for k in scores:
            scores[k] = round(scores[k] / max_score, 2)

        # rank and colorize
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for i, (model, score) in enumerate(ranked):
            color = "green" if i == 0 else "yellow" if i == 1 else "red"
            result.append({
                "model": model,
                "score": score,
                "color": color
            })

        return {"ranked_models": result, "rationale": rationale}


# =====================================================================
if __name__ == "__main__":
    df = pd.DataFrame({
        "age": [23, 45, 31, 35, 62],
        "salary": [50000, 80000, 60000, 72000, 100000],
        "dept": ["IT", "HR", "IT", "Finance", "HR"],
        "target": [0, 1, 0, 1, 1]
    })

    analyzer = DatasetAnalyzer(df, target="target", data_type="tabular")
    analysis = analyzer.analyze()
    print("Analysis:", analysis)

    selector = ModelSelector(analysis)
    result = selector.recommend()
    print("\nRanked Models:")
    for r in result["ranked_models"]:
        print(f"{r['color'].upper():<6} | {r['model']} (score={r['score']})")
