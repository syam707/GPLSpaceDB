namespace VDAlgorithmicEngine;

using System;
using System.IO;

/// <summary>
/// Binds the vector store, the metadata log and the HNSW index into one unit, and
/// reloads all three from disk when the graph file is present.
/// </summary>
public sealed class VectorEngine : IDisposable
{
    public PersistentVectorStore VectorStore { get; }
    public MetadataStore MetadataStore { get; }
    public HnswIndex Index { get; }

    /// <summary>Where <see cref="Save"/> writes the graph topology. Defaults to the vector file with a <c>.hnsw</c> extension.</summary>
    public string GraphFilePath { get; }

    /// <summary>True if the graph was reloaded from <see cref="GraphFilePath"/> rather than built fresh.</summary>
    public bool LoadedFromDisk { get; }

    public VectorEngine(
        string vectorFilePath,
        string metadataFilePath,
        int dimensions,
        int m = 16,
        int efConstruction = 100,
        int maxVectors = 10_000_000,
        string? graphFilePath = null)
    {
        var store = new PersistentVectorStore(vectorFilePath, dimensions, maxVectors);
        MetadataStore? metadata = null;

        try
        {
            metadata = new MetadataStore(metadataFilePath);

            string graphPath = graphFilePath ?? DeriveGraphPath(vectorFilePath);
            bool loaded = File.Exists(graphPath);

            // Without this the index was memory-only: vectors.bin and metadata.dat
            // survived a restart but the graph did not, so every stored vector became
            // unreachable and Search returned nothing at all.
            HnswIndex index = loaded
                ? HnswIndex.LoadGraph(graphPath, store)
                : new HnswIndex(store, m, efConstruction);

            VectorStore = store;
            MetadataStore = metadata;
            Index = index;
            GraphFilePath = graphPath;
            LoadedFromDisk = loaded;
        }
        catch
        {
            metadata?.Dispose();
            store.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Swaps the vector file's extension for <c>.hnsw</c>. Done by hand rather than with
    /// <see cref="Path.ChangeExtension"/>, whose return type is nullable.
    /// </summary>
    private static string DeriveGraphPath(string vectorFilePath)
    {
        int dot = vectorFilePath.LastIndexOf('.');
        int separator = vectorFilePath.AsSpan().LastIndexOfAny('/', '\\');

        return dot > separator
            ? string.Concat(vectorFilePath.AsSpan(0, dot), ".hnsw")
            : vectorFilePath + ".hnsw";
    }

    public int Count => Index.Count;
    public int Dimensions => VectorStore.Dimensions;

    /// <summary>
    /// Adds or re-indexes a vector. <see cref="HnswIndex.Insert"/> owns the write to
    /// <see cref="VectorStore"/>, so this no longer writes the vector a second time.
    /// </summary>
    public void Insert(int id, ReadOnlySpan<float> vector, string? metadata = null)
    {
        Index.Insert(id, vector);
        MetadataStore.SetMetadata(id, metadata);
    }

    public VectorSearchResult[] ExecuteSearchWithMetadata(ReadOnlySpan<float> queryVector, int k, int efSearch = 0)
    {
        // Scores come back from the traversal, which already computed them - the old
        // path threw the distances away and recomputed a cosine similarity per hit.
        (int Id, float Score)[] hits = Index.SearchWithScores(queryVector, k, efSearch);

        var results = new VectorSearchResult[hits.Length];
        for (int i = 0; i < hits.Length; i++)
        {
            results[i] = new VectorSearchResult(hits[i].Id, hits[i].Score, MetadataStore.GetMetadata(hits[i].Id));
        }
        return results;
    }

    /// <summary>Persists the graph and flushes both stores. Call with inserts quiesced.</summary>
    public void Save()
    {
        Index.SaveGraph(GraphFilePath);
        VectorStore.Flush();
        MetadataStore.Flush();
    }

    public void Dispose()
    {
        MetadataStore.Dispose();
        VectorStore.Dispose();
    }
}

public static class VectorSearchExtensions
{
    /// <summary>
    /// Attaches scores and metadata to a bare id array.
    /// </summary>
    /// <remarks>
    /// Kept for compatibility. Prefer <see cref="VectorEngine.ExecuteSearchWithMetadata"/>
    /// or <see cref="HnswIndex.SearchWithScores"/>, which reuse the distances the search
    /// already computed instead of recomputing one cosine similarity per result.
    /// </remarks>
    public static VectorSearchResult[] WithMetadata(
        this int[] resultIds, ReadOnlySpan<float> queryVector, PersistentVectorStore store, MetadataStore metaStore)
    {
        ArgumentNullException.ThrowIfNull(resultIds);
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(metaStore);

        var results = new VectorSearchResult[resultIds.Length];
        for (int i = 0; i < resultIds.Length; i++)
        {
            int id = resultIds[i];
            float score = VectorMath.CosineSimilarity(queryVector, store.GetVectorSpan(id));
            results[i] = new VectorSearchResult(id, score, metaStore.GetMetadata(id));
        }
        return results;
    }
}
