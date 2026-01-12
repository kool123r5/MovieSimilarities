export const num_movies = 84432;
export const embedding_dim = 256;

let embeddings: Float32Array | null = null;

export async function loadEmbeddings(binUrl: string = "/embeddings.bin"): Promise<void> {
    const resp = await fetch(binUrl);
    if (!resp.ok) {
        throw new Error(`Failed to fetch embeddings file: ${resp.status} ${resp.statusText}`);
    }

    const buffer = await resp.arrayBuffer();
    const expectedBytes = num_movies * embedding_dim * 4;
    if (buffer.byteLength < expectedBytes) {
        throw new Error(`embeddings.bin too small: got ${buffer.byteLength} bytes, expected at least ${expectedBytes}`);
    }

    const all = new Float32Array(buffer);
    if (all.length < num_movies * embedding_dim) {
        throw new Error(
            `Float32 length mismatch after conversion: got ${all.length}, expected ${num_movies * embedding_dim}`
        );
    }

    embeddings = all.length === num_movies * embedding_dim ? all : all.subarray(0, num_movies * embedding_dim);
}

export function isLoaded(): boolean {
    return embeddings !== null;
}

export function getEmbeddings(): Float32Array {
    if (!embeddings) throw new Error("Embeddings not loaded. Call loadEmbeddings() first.");
    return embeddings;
}
