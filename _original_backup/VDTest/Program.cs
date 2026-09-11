using System;
using VDAlgorithmicEngine;
using System.IO;

class Program
{
    static void Main()
    {
        string vectorFile = "vectors.bin";
        string metaFile = "metadata.dat";

        if (File.Exists(vectorFile)) File.Delete(vectorFile);
        if (File.Exists(metaFile)) File.Delete(metaFile);

        Console.WriteLine("Initializing VectorEngine...");
        
        using var engine = new VectorEngine(vectorFile, metaFile, 128);

        int numVectors = 1000;
        int dim = 128;
        var rand = new Random(42);
        
        var vectors = new float[numVectors][];
        
        Console.WriteLine($"Inserting {numVectors} vectors with metadata...");
        for (int i = 0; i < numVectors; i++)
        {
            vectors[i] = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                vectors[i][d] = (float)(rand.NextDouble() * 2.0 - 1.0); 
            }
            engine.Insert(i, vectors[i], $"Document_{i}_Meta");
        }

        Console.WriteLine("Testing Search with Metadata...");
        var query = vectors[0];
        
        // JIT warmup
        engine.ExecuteSearchWithMetadata(query, 5);
        
        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
        var results = engine.ExecuteSearchWithMetadata(query, 5);
        long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
        
        Console.WriteLine($"Search results for vector 0 (k=5):");
        foreach(var res in results)
        {
            Console.WriteLine($"  ID: {res.Id}, Score: {res.Score:F4}, Meta: {res.Metadata}");
        }
        
        Console.WriteLine($"Bytes allocated on current thread during search + metadata fetch: {allocatedAfter - allocatedBefore}");

        if (results.Length > 0 && results[0].Id == 0)
        {
            Console.WriteLine("SUCCESS: The closest neighbor to vector 0 is vector 0.");
        }
        else
        {
            Console.WriteLine("FAILURE: The closest neighbor should be vector 0.");
        }
    }
}
