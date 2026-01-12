import { movies } from "../movies";

const idToTitleMap = new Map<number, string>();

for (const [id, title] of movies) {
    idToTitleMap.set(id as number, title as string);
}

export const idToTitle = (id: number) => {
    return idToTitleMap.get(id);
};
