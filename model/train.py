import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import time
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from pathlib import Path


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim, sparse=True)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim, sparse=True)
        self.user_bias = nn.Embedding(num_users, 1, sparse=True)
        self.movie_bias = nn.Embedding(num_movies, 1, sparse=True)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.movie_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids, movie_ids):
        u = self.user_emb(user_ids)  # (B, k)
        m = self.movie_emb(movie_ids)  # (B, k)
        dot = (u * m).sum(dim=1)  # (B,)
        bias = self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze()
        return dot + bias


df = pd.read_csv("./ratings_big.csv")
if "timestamp" in df.columns:
    df = df.drop("timestamp", axis=1)

user_codes, user_uniques = pd.factorize(df["userId"])
movie_codes, movie_uniques = pd.factorize(df["movieId"])

df["user_idx"] = user_codes
df["movie_idx"] = movie_codes

num_users = len(user_uniques)
num_movies = len(movie_uniques)
print(f"{len(df)} ratings, {num_users} users, {num_movies} movies")


user_tensor = torch.tensor(df["user_idx"].values, dtype=torch.long)
movie_tensor = torch.tensor(df["movie_idx"].values, dtype=torch.long)
rating_tensor = torch.tensor(df["rating"].values, dtype=torch.float32)

batch_size = 2**14
device = "cuda:1"
embed_dim = 256
lr = 2e-3
epochs = 400
prev_model_path = "./movie_embeddings.pth"

dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
model = MatrixFactorization(
    num_users=num_users, num_movies=num_movies, embedding_dim=embed_dim
).to(device)
if Path(prev_model_path).exists():
    weights = torch.load(prev_model_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)
sparse_params = (
    list(model.user_emb.parameters())
    + list(model.movie_emb.parameters())
    + list(model.user_bias.parameters())
    + list(model.movie_bias.parameters())
)

optimizer = torch.optim.SparseAdam(sparse_params, lr=lr)
dataset_len = len(dataset)
scaler = GradScaler(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    t = time.time()
    for batch_users, batch_movies, batch_ratings in data_loader:
        batch_users = batch_users.to(device)
        batch_movies = batch_movies.to(device)
        batch_ratings = batch_ratings.to(device)

        with autocast(device_type=device):
            preds = model(batch_users, batch_movies)
            loss = F.mse_loss(preds, batch_ratings)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()

        running_loss += loss.item() * batch_ratings.size(0)

    avg_loss = running_loss / dataset_len
    print(f"Epoch {epoch}. Loss: {avg_loss:.6f}. Time taken: {(time.time() - t):.1f}s")
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), prev_model_path)
        print(f"Saved model at {prev_model_path}")
torch.save(model.state_dict(), prev_model_path)
print(f"Saved final model at {prev_model_path}")
