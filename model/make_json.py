import pandas as pd
import json

df = pd.read_csv("merged_movies.csv")

movies_list = df[["movieId", "title", "imdbId"]].values.tolist()

with open("movies.json", "w", encoding="utf-8") as f:
    json.dump(movies_list, f, ensure_ascii=False, indent=2)
