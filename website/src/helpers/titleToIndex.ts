import { movies } from "../movies";
import { idToIdx } from "../idToIdx";

const titleToIdMap = new Map<string, number>();

for (const [id, title] of movies) {
    titleToIdMap.set(title as string, id as number);
}

export function titleToIndex(title: string): number | undefined {
    const movieId = titleToIdMap.get(title);
    if (movieId == undefined) {
        return undefined;
    }
    return idToIdx[movieId.toString() as keyof typeof idToIdx];
}
