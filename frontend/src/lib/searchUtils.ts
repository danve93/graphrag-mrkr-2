export function levenshteinDistance(a: string, b: string): number {
    const matrix = [];

    for (let i = 0; i <= b.length; i++) {
        matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
        matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(
                    matrix[i - 1][j - 1] + 1, // substitution
                    Math.min(
                        matrix[i][j - 1] + 1, // insertion
                        matrix[i - 1][j] + 1 // deletion
                    )
                );
            }
        }
    }

    return matrix[b.length][a.length];
}

export interface SearchMatch<T> {
    item: T;
    score: number;
    highlightIndices?: number[][];
}

export function fuzzySearch<T>(
    query: string,
    items: T[],
    keys: (keyof T)[]
): SearchMatch<T>[] {
    if (!query) return [];

    const lowerQuery = query.toLowerCase();

    const matches = items
        .map((item) => {
            let minDistance = Infinity;
            let foundExact = false;

            for (const key of keys) {
                const value = String(item[key]);
                const lowerValue = value.toLowerCase();

                // Exact match or substring match (Highest priority)
                if (lowerValue.includes(lowerQuery)) {
                    minDistance = 0;
                    foundExact = true;
                    break; // Found the best possible match for this item
                }

                // Fuzzy match using Levenshtein distance
                // We only check fuzzy if we haven't found an exact match yet
                // Optimization: Skip fuzzy check for very short strings or very different lengths
                if (Math.abs(lowerValue.length - lowerQuery.length) > 3) {
                    // Primitive length check optimization
                    continue;
                }

                const dist = levenshteinDistance(lowerQuery, lowerValue);
                if (dist < minDistance) {
                    minDistance = dist;
                }
            }

            // Relevance score: lower distance is better.
            // We can also weight it by the length difference to favor closer matches in length.
            return { item, score: minDistance };
        })
        .filter((match) => {
            // Filter out bad matches
            // Exact matches have score 0
            // Fuzzy matches: allow up to 2 changes for queries > 3 chars, 1 change for shorter
            const allowedDistance = lowerQuery.length > 3 ? 2 : 1;
            return match.score <= allowedDistance;
        });

    // Sort by score (ascending, 0 is best)
    return matches.sort((a, b) => a.score - b.score);
}
