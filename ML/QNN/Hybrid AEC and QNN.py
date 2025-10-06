"""
AEC ➜ Experimental QNN (Clean) Pipeline
--------------------------------------
This script connects Autoencoder (AEC) encoder to Experimental QNN model.
It performs:
1. Data loading
2. Feature encoding with AEC
3. Quantum classification using the Experimental QNN (Clean)
4. Real-time training and evaluation (no model saving)

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Import your Experimental QNN (Clean) training function
from new_qnn_generic_clean import train_qnn

# ---------------------------------------------------------
# 🧠 Step 1: Define Autoencoder for feature extraction
# ---------------------------------------------------------
def build_autoencoder(input_dim, encoding_dim=4):
    """Creates and returns an Autoencoder model"""
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder

# ---------------------------------------------------------
# 🧩 Step 2: Training and Using Encoder + QNN
# ---------------------------------------------------------
def train_aec_qnn_pipeline(data_path, target_col, encoding_dim=4, qnn_epochs=25):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build and train Autoencoder
    print("\n🚀 Training Autoencoder for Feature Compression...")
    autoencoder, encoder = build_autoencoder(input_dim=X_train.shape[1], encoding_dim=encoding_dim)
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=16, verbose=0)

    # Extract compressed features
    print("🔹 Extracting latent features from Autoencoder...")
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Combine encoded features + target into new DataFrame for QNN
    encoded_df = pd.DataFrame(X_train_encoded, columns=[f"feat_{i}" for i in range(X_train_encoded.shape[1])])
    encoded_df[target_col] = y_train

    # Save temporarily for QNN input (QNN expects CSV path)
    encoded_csv_path = "encoded_dataset.csv"
    encoded_df.to_csv(encoded_csv_path, index=False)

    print("\n⚛️ Training Experimental QNN (Clean) on Encoded Features...")
    train_qnn(encoded_csv_path, target_col)

    print("\n✅ AEC → Experimental QNN Pipeline Complete!")

# ---------------------------------------------------------
# 🧩 Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example: adjust file name and target column for your dataset
    train_aec_qnn_pipeline(
        data_path="dataset.csv",  # your tabular dataset path
        target_col="label",       # name of target column
        encoding_dim=4,
        qnn_epochs=25
    )
