"""
Unified Dataset Analyzer + Model Selector
-----------------------------------------
Automatically selects and trains the best model based on dataset type and complexity.

Supported models:
 - MLP (classical baseline)
 - Hybrid (classical + quantum)
 - Baseline QNN (simple 2-qubit)
 - AEC + Experimental QNN pipeline (merged)
 - CNN (for images)
 - 3D CNN placeholder (for video)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import os
import cv2

# --- Import model training functions ---
from MLP_split import train_mlp             # Classical MLP
from Hybrid import train_hybrid             # Hybrid model
from qnn_generic import train_qnn as train_baseline_qnn  # Baseline QNN
from AEC_QNN_pipeline import train_aec_qnn_pipeline      # AEC + Experimental QNN merged pipeline


# -------------------------------------------------------
# 🧠 Dataset Analyzer
# -------------------------------------------------------
class DatasetAnalyzer:
    def __init__(self, data, target=None, data_type="tabular"):
        self.data = data
        self.target = target
        self.data_type = data_type
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

    # ---- Tabular ----
    def _analyze_tabular(self):
        df = self.data
        n_samples, n_features = df.shape
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        numeric_cols = df.select_dtypes(include=["number"]).columns

        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        # PCA variance
        try:
            pca = PCA(n_components=min(5, len(numeric_cols)))
            pca.fit(df_encoded[numeric_cols])
            explained_var = np.sum(pca.explained_variance_ratio_)
        except Exception:
            explained_var = 0

        # Mutual information with target (if exists)
        if self.target and self.target in df.columns:
            y = df[self.target]
            X = df.drop(columns=[self.target])
            X_enc = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)) if col.dtype=="object" else col)
            try:
                mi = mutual_info_classif(X_enc, y, discrete_features="auto")
                avg_mi = np.mean(mi)
            except Exception:
                avg_mi = 0
        else:
            avg_mi = 0

        self.analysis = {
            "type": "tabular",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_categorical": len(categorical_cols),
            "n_numeric": len(numeric_cols),
            "explained_var_pca": explained_var,
            "avg_mutual_info": avg_mi,
        }

    # ---- Image ----
    def _analyze_image(self):
        image_dir = self.data
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not files:
            self.analysis = {"type": "image", "n_samples": 0}
            return

        sizes = []
        for f in files[:50]:
            img = cv2.imread(os.path.join(image_dir, f))
            if img is not None:
                sizes.append(img.shape)

        if sizes:
            avg_h = np.mean([s[0] for s in sizes])
            avg_w = np.mean([s[1] for s in sizes])
            channels = sizes[0][2] if len(sizes[0]) > 2 else 1
        else:
            avg_h, avg_w, channels = 0, 0, 0

        self.analysis = {
            "type": "image",
            "n_samples": len(files),
            "avg_height": avg_h,
            "avg_width": avg_w,
            "channels": channels,
        }

    # ---- Video ----
    def _analyze_video(self):
        self.analysis = {
            "type": "video",
            "note": "Video analysis not yet implemented"
        }


# -------------------------------------------------------
# 🤖 Model Selector
# -------------------------------------------------------
class ModelSelector:
    def __init__(self, analysis):
        self.analysis = analysis

    def recommend(self, data_path=None, target=None):
        dtype = self.analysis["type"]

        if dtype == "tabular":
            n_samples = self.analysis["n_samples"]
            n_features = self.analysis["n_features"]
            explained_var = self.analysis["explained_var_pca"]
            avg_mi = self.analysis["avg_mutual_info"]

            print("\n📊 Dataset Summary:")
            print(f"Samples: {n_samples}, Features: {n_features}, PCA Var: {explained_var:.2f}, MI: {avg_mi:.3f}")

            # --- Decision Rules ---
            if n_samples < 5000 and n_features <= 50:
                print("\n🔹 Small structured dataset → using Baseline QNN")
                train_baseline_qnn(data_path, target)
                return "Baseline QNN"

            elif 5000 <= n_samples < 20000:
                print("\n⚛️ Medium-sized dataset → using AEC + Experimental QNN pipeline")
                train_aec_qnn_pipeline(data_path, target)
                return "AEC + Experimental QNN"

            elif n_samples >= 20000:
                print("\n🧠 Large dataset → using Hybrid Quantum-Classical model")
                train_hybrid(data_path, target)
                return "Hybrid Quantum-Classical"

            else:
                print("\n⚙️ Defaulting to classical MLP model")
                train_mlp(data_path, target)
                return "MLP (Classical Baseline)"

        elif dtype == "image":
            print("\n🖼️ Image dataset detected — use CNN or Vision Transformer model.")
            return "CNN (Future integration)"

        elif dtype == "video":
            print("\n🎥 Video dataset detected — use 3D CNN (future work).")
            return "3D CNN (future)"

        else:
            return "Unknown"
