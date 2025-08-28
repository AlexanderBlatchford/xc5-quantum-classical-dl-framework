import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(csv_path, target, test_size=0.2):
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=test_size, random_state=42), le.classes_
