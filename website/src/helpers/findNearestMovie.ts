import { titleToIndex } from "./titleToIndex";
import { getEmbeddings, isLoaded, num_movies, embedding_dim } from "./loadEmbeddings";
import { idxToId } from "@/idxToId";

export type NeighborMovie = { movieID: number; score: number };

export function findNearestMovie(title: string, topK = 10): NeighborMovie[] {
    const queryIdx = titleToIndex(title);

    if (queryIdx == undefined) {
        return [];
    }

    if (!isLoaded()) {
        throw new Error("Embeddings not loaded. Call loadEmbeddings() before findNearestByIndex.");
    }

    const embeddings = getEmbeddings();
    const qOffset = queryIdx * embedding_dim;
    const queryEmbedding = embeddings.subarray(qOffset, qOffset + embedding_dim);

    const top: NeighborMovie[] = [];

    const tryInsert = (cand: NeighborMovie) => {
        if (top.length < topK) {
            top.push(cand);
            top.sort((a, b) => a.score - b.score);
            return;
        }
        if (cand.score <= top[0].score) return;
        top[0] = cand;
        top.sort((a, b) => a.score - b.score);
    };

    for (let i = 0; i < num_movies; i++) {
        if (i === queryIdx) continue;
        const base = i * embedding_dim;
        let s = 0.0;
        for (let d = 0; d < embedding_dim; d++) {
            s += queryEmbedding[d] * embeddings[base + d];
        }

        tryInsert({ movieID: idxToId[i.toString() as keyof typeof idxToId], score: s });
    }

    return top.sort((a, b) => b.score - a.score);
}
