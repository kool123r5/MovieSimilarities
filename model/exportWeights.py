import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pandas as pd
import json


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, user_ids, movie_ids):
        u = self.user_emb(user_ids)
        m = self.movie_emb(movie_ids)
        dot = (u * m).sum(dim=1)
        bias = self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze()
        return dot + bias


df = pd.read_csv("./ratings_big.csv")
_, movie_uniques = pd.factorize(df["movieId"])
_, user_uniques = pd.factorize(df["userId"])

movieid_to_idx = {mid: idx for idx, mid in enumerate(movie_uniques)}
idx_to_movieid = {idx: mid for mid, idx in movieid_to_idx.items()}

model = MatrixFactorization(len(user_uniques), len(movie_uniques), 256)
model.load_state_dict(
    torch.load("./movie_embeddings.pth", map_location="cpu", weights_only=True)
)

movie_embeddings = (
    F.normalize(model.movie_emb.weight.detach(), dim=1).cpu().numpy().astype(np.float32)
)

num_movies, dim = movie_embeddings.shape
movie_embeddings.tofile("embeddings.bin")
