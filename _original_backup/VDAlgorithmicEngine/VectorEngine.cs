namespace VDAlgorithmicEngine;

using System;

public class VectorEngine : IDisposable
{
    public PersistentVectorStore VectorStore { get => field; init => field = value; }
    public MetadataStore MetadataStore { get => field; init => field = value; }
    public HnswIndex Index { get => field; init => field = value; }

    public VectorEngine(string vectorFilePath, string metadataFilePath, int dimensions, int m = 16, int efConstruction = 100)
    {
        VectorStore = new PersistentVectorStore(vectorFilePath, dimensions);
        MetadataStore = new MetadataStore(metadataFilePath);
        Index = new HnswIndex(VectorStore, m, efConstruction);
    }

    public void Insert(int id, ReadOnlySpan<float> vector, string metadata)
    {
        VectorStore.WriteVector(id, vector);
        MetadataStore.SetMetadata(id, metadata);
        Index.Insert(id, vector);
    }

    public VectorSearchResult[] ExecuteSearchWithMetadata(ReadOnlySpan<float> queryVector, int k)
    {
        int[] resultIds = Index.Search(queryVector, k);
        return resultIds.WithMetadata(queryVector, VectorStore, MetadataStore);
    }

    public void Dispose()
    {
        VectorStore.Dispose();
        MetadataStore.Dispose();
    }
}

public static class VectorSearchExtensions
{
    public static VectorSearchResult[] WithMetadata(this int[] resultIds, ReadOnlySpan<float> queryVector, PersistentVectorStore store, MetadataStore metaStore)
    {
        var results = new VectorSearchResult[resultIds.Length];
        for (int i = 0; i < resultIds.Length; i++)
        {
            int id = resultIds[i];
            float score = VectorMath.CosineSimilarity(queryVector, store.GetVectorSpan(id));
            string meta = metaStore.GetMetadata(id);
            results[i] = new VectorSearchResult(id, score, meta);
        }
        return results;
    }
}
