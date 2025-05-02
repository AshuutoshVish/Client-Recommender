from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load


def log_transform_safe(x):
    x = x.copy()
    x[x <= 0] = 1e-10
    return np.log(x)

def clean_binary(X):
    return X.apply(lambda col: col.map(lambda val: 1 if pd.to_numeric(val, errors='coerce') > 0 else 0).astype(int))

def clean_categorical(X):
    return X.apply(lambda col: col.map(lambda val: str(val).strip().lower() if pd.notnull(val) else 'others'))


def clean_numerical(X):
    return X.apply(lambda col: pd.to_numeric(col, errors='coerce'))

def clean_ordinal(X):
    return X.apply(lambda col: pd.to_numeric(col, errors='coerce')).clip(lower=0)


# --- Load Assets ---
pipeline = load('preprocess_pipeline.pkl')
embeddings = np.load('embeddings.npy')
original_df = pd.read_csv('original_data.csv')

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructed = self.decoder(embeddings)
        return reconstructed, embeddings


data = pd.read_csv('original_data.csv')
input_dim = pipeline.transform(data).shape[1]
model = Autoencoder(input_dim)
model.load_state_dict(torch.load('Autoencoder.pth', map_location='cpu'))
model.eval()
embeddings = np.load('embeddings.npy')
candidates_df = data


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.form.to_dict()
        user_input_df = pd.DataFrame([user_input])
        user_input_df = pipeline.transform(user_input_df)

        with torch.no_grad():
            _, user_embedding = model(torch.tensor(user_input_df.values, dtype=torch.float32))

        similarities = cosine_similarity(user_embedding.numpy(), embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:5]

        recommendations = candidates_df.iloc[top_indices].to_dict(orient='records')
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)