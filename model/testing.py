import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

df = pd.read_csv("./ratings_big.csv")
_, movie_uniques = pd.factorize(df["movieId"])
_, user_uniques = pd.factorize(df["userId"])

movieid_to_idx = {mid: idx for idx, mid in enumerate(movie_uniques)}
idx_to_movieid = {idx: mid for mid, idx in movieid_to_idx.items()}


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


model = MatrixFactorization(len(user_uniques), len(movie_uniques), 256)
model.load_state_dict(
    torch.load("./movie_embeddings.pth", map_location="cpu", weights_only=True)
)
model.eval()
movie_embeddings = F.normalize(model.movie_emb.weight.detach(), dim=1)

import json

with open("idx_to_id.json", "w") as f:
    json.dump(idx_to_movieid, f)

with open("id_to_idx.json", "w") as f:
    json.dump(movieid_to_idx, f)


def get_similar_movies(movie_id, top_k=10):
    idx = movieid_to_idx[movie_id]
    print(idx)
    v = movie_embeddings[idx]
    sims = torch.mv(movie_embeddings, v)
    topk_indices = torch.topk(sims, top_k + 1).indices.tolist()

    results = []
    for i in topk_indices:
        if i != idx:
            results.append((idx_to_movieid[i], sims[i].item()))
            if len(results) == top_k:
                break
    return results


print(get_similar_movies(1, top_k=5))
