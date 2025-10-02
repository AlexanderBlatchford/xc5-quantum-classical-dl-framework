import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import os
import cv2

# TODO: Assert data type of dataset eg tabular/etc
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

    def _analyze_tabular(self):
        df = self.data
        n_samples, n_features = df.shape
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        numeric_cols = df.select_dtypes(include=["number"]).columns

        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        try:
            pca = PCA(n_components=min(5, n_features))
            pca.fit(df_encoded[numeric_cols])
            explained_var = np.sum(pca.explained_variance_ratio_)
        except Exception:
            explained_var = 0

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

    def _analyze_image(self):
        image_dir = self.data
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not files:
            self.analysis = {"type": "image", "n_samples": 0}
            return

        sizes = []
        for f in files[:50]:  # sample first 50 images
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

    def _analyze_video(self):
        # Placeholder
        self.analysis = {
            "type": "video",
            "note": "Video analysis not yet implemented"
        }

class ModelSelector:
    def __init__(self, analysis):
        self.analysis = analysis

    def recommend(self):
        dtype = self.analysis["type"]

        if dtype == "tabular":
            n_samples = self.analysis["n_samples"]
            n_features = self.analysis["n_features"]
            explained_var = self.analysis["explained_var_pca"]
            avg_mi = self.analysis["avg_mutual_info"]

            # Simple heuristic scoring
            if n_features < 50 and avg_mi > 0.05:
                return "MLP"
            elif n_features >= 50 and explained_var > 0.7:
                return "AEC"
            elif n_samples > 10000:
                return "MLP/QNN"
            else:
                return "QCN"

        elif dtype == "image":
            if self.analysis["avg_height"] > 28 and self.analysis["avg_width"] > 28:
                return "CNN"
            else:
                return "MLP"

        elif dtype == "video":
            return "CNN (extend to 3D CNN later)"

        else:
            return "Unknown"


if __name__ == "__main__":
    # Example with tabular
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
    print("Recommended model:", selector.recommend())
