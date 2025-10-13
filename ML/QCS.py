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

        # Encode categorical data
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        # ----- 1. PCA for dimensionality -----
        explained_var = 0
        if len(numeric_cols) > 1:
            try:
                pca = PCA(n_components=min(5, len(numeric_cols)))
                pca.fit(StandardScaler().fit_transform(df_encoded[numeric_cols]))
                explained_var = np.sum(pca.explained_variance_ratio_)
            except Exception:
                pass

        # ----- 2. Mutual Information -----
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

        # ----- 3. Noise and Consistency -----
        noise_score = self._estimate_noise(df_encoded[numeric_cols]) if len(numeric_cols) > 0 else 0
        consistency_score = 1 - noise_score

        # ----- 4. Correlation Redundancy -----
        corr_redundancy = self._correlation_redundancy(df_encoded[numeric_cols]) if len(numeric_cols) > 1 else 0

        # ----- 5. Missing Data -----
        missing_ratio = df.isna().sum().sum() / (n_samples * n_features)

        # ----- 6. Outlier Fraction -----
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
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
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
        self.analysis = {
            "type": "video",
            "note": "Video analysis not yet implemented"
        }

    # ========== HELPER FUNCTIONS ==========
    def _estimate_noise(self, df_num):
        """Estimate data noise using variance of z-scores."""
        try:
            z = np.abs(zscore(df_num))
            noise = np.mean(np.std(z, axis=0))
            return min(noise / 5, 1)  # normalize to [0,1]
        except Exception:
            return 0

    def _correlation_redundancy(self, df_num):
        """Estimate redundancy (high correlations between features)."""
        corr = df_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        redundant = (upper > 0.8).sum().sum()
        redundancy_ratio = redundant / (len(df_num.columns) * (len(df_num.columns)-1)/2)
        return redundancy_ratio

    def _outlier_fraction(self, df_num):
        """Simple outlier ratio based on z-score threshold."""
        z = np.abs(zscore(df_num))
        return np.mean(z > 3)


# =====================================================================
class ModelSelector:
    def __init__(self, analysis, thresholds=None):
        self.analysis = analysis
        # Allow user to tweak thresholds easily
        self.thresholds = thresholds or {
            "high_dim": 50,
            "high_sample": 10000,
            "high_mi": 0.05,
            "low_noise": 0.3,
            "low_missing": 0.1
        }

    def recommend(self):
        dtype = self.analysis["type"]
        rationale = []

        if dtype == "tabular":
            n_features = self.analysis["n_features"]
            n_samples = self.analysis["n_samples"]
            mi = self.analysis["avg_mutual_info"]
            noise = self.analysis["noise_score"]
            missing = self.analysis["missing_ratio"]
            explained = self.analysis["explained_var_pca"]

            if n_features < self.thresholds["high_dim"] and mi > self.thresholds["high_mi"]:
                model = "MLP"
                rationale.append("Low feature dimensionality with good mutual information.")
            elif explained > 0.7 and noise < self.thresholds["low_noise"]:
                model = "Autoencoder (AEC)"
                rationale.append("High PCA explained variance and consistent features.")
            elif n_samples > self.thresholds["high_sample"]:
                model = "MLP or QNN"
                rationale.append("Large dataset suitable for classical or hybrid models.")
            elif noise > 0.5 or missing > self.thresholds["low_missing"]:
                model = "Tree-based (Random Forest / XGBoost)"
                rationale.append("Data shows high noise or missingness, tree models handle this well.")
            else:
                model = "Quantum Convolutional Network (QCN)"
                rationale.append("Moderate structure and small size, good for quantum exploration.")

            confidence = round(1 - min(noise, 0.9), 2)
            return {"recommended_model": model, "confidence": confidence, "rationale": rationale}

        elif dtype == "image":
            if self.analysis["avg_height"] > 28 and self.analysis["avg_width"] > 28:
                model = "CNN"
                rationale = ["Image size indicates spatial features; CNN recommended."]
            else:
                model = "MLP"
                rationale = ["Small flat images suitable for simple feedforward networks."]
            return {"recommended_model": model, "confidence": 0.9, "rationale": rationale}

        elif dtype == "video":
            return {
                "recommended_model": "3D CNN or CNN + RNN",
                "confidence": 0.8,
                "rationale": ["Temporal and spatial data suggest CNN–RNN hybrid."]
            }

        else:
            return {"recommended_model": "Unknown", "confidence": 0.0, "rationale": ["Unsupported type."]}


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
    print("Recommendation:", result)
