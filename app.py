from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

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

# Load model, embeddings, and pipeline
original_df = pd.read_csv('original_data.csv')
embeddings = np.load('embeddings.npy')
pipeline = joblib.load('preprocess_pipeline.pkl')


# Load the original dataset, model, and embeddings
data = pd.read_csv('original_data.csv')
input_dim = pipeline.transform(data).shape[1]
model = Autoencoder(input_dim)
model.load_state_dict(torch.load('Autoencoder.pth', map_location='cpu'))
model.eval()
embeddings = np.load('embeddings.npy')
candidates_df = data



app = Flask(__name__)

@app.route('/home')
def myhome():
    return render_template('home.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    candidate = {key: request.form[key] for key in request.form}    
    sample_df = pd.DataFrame([candidate])
    sample_processed = pipeline.transform(sample_df)
    sample_tensor = torch.tensor(sample_processed, dtype=torch.float32)
    
    # Get the embedding for sample
    with torch.no_grad():
        _, sample_embedding = model(sample_tensor)
    
    # Compute similarities
    sample_embedding_np = sample_embedding.numpy()
    similarities = cosine_similarity(sample_embedding_np, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:5]

    candidates = []
    for idx in top_indices:
        similarity_score = similarities[idx]
        candidate_details = candidates_df.iloc[idx].to_dict()
        candidates.append({
            "index": idx,
            "similarity": f"{similarity_score:.4f}",
            "details": candidate_details
        })
    
    # Render to result page
    return render_template('results.html', candidates=candidates, top_n=5)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
