import pandas as pd

links = pd.read_csv("links.csv")
movies = pd.read_csv("movies.csv")

movies = movies.drop("genres", axis=1)

merged = pd.merge(movies, links, on="movieId", how="inner")

merged.to_csv("merged_movies.csv", index=False)
