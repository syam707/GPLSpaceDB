using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using VDAlgorithmicEngine;

/// <summary>
/// Regression harness for VDAlgorithmicEngine. Every case below corresponds to a defect
/// that the original 1000-vector smoke test could not see.
/// </summary>
internal static class Program
{
    private const int Dimensions = 128;
    private const int MaxVectors = 20_000;

    private static int _passed;
    private static int _failed;

    private static int Main()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "vdtest-data");
        Directory.CreateDirectory(dir);
        foreach (string stale in Directory.GetFiles(dir)) File.Delete(stale);

        string vectorFile = Path.Combine(dir, "vectors.bin");
        string metaFile = Path.Combine(dir, "metadata.dat");
        string graphFile = Path.Combine(dir, "vectors.hnsw");

        Console.WriteLine($"VDAlgorithmicEngine regression suite  (dim={Dimensions}, maxVectors={MaxVectors:N0})");
        Console.WriteLine(new string('=', 78));

        int n = 4_000;
        float[][] vectors = MakeClusteredVectors(n, Dimensions, clusters: 20, spread: 0.10f, seed: 42);

        using (var engine = new VectorEngine(vectorFile, metaFile, Dimensions, maxVectors: MaxVectors))
        {
            Section("build");
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < n; i++) engine.Insert(i, vectors[i], $"Document_{i}");
            sw.Stop();
            Console.WriteLine($"  inserted {n:N0} vectors in {sw.ElapsedMilliseconds:N0} ms " +
                              $"({n / Math.Max(sw.Elapsed.TotalSeconds, 1e-9):N0}/s)");
            Check("index count matches insert count", engine.Count == n, $"{engine.Count} != {n}");

            Section("retrieval");
            var self = engine.ExecuteSearchWithMetadata(vectors[0], 5);
            Check("nearest neighbour of a stored vector is itself",
                self.Length > 0 && self[0].Id == 0,
                self.Length == 0 ? "no results" : $"got id {self[0].Id}");
            Check("self-similarity is ~1.0",
                self.Length > 0 && MathF.Abs(self[0].Score - 1.0f) < 1e-3f,
                self.Length > 0 ? $"score {self[0].Score}" : "no results");
            Check("metadata round-trips", self.Length > 0 && self[0].Metadata == "Document_0",
                self.Length > 0 ? $"'{self[0].Metadata}'" : "no results");

            double recall = MeasureRecall(engine, vectors, n, k: 10, queries: 200);
            Console.WriteLine($"  recall@10 vs brute force: {recall:P1}");
            Check("recall@10 >= 90%", recall >= 0.90, $"{recall:P1}");

            double recallHighEf = MeasureRecall(engine, vectors, n, k: 10, queries: 200, efSearch: 200);
            Console.WriteLine($"  recall@10 at efSearch=200: {recallHighEf:P1}");
            Check("raising efSearch does not reduce recall", recallHighEf >= recall - 0.01,
                $"{recallHighEf:P1} < {recall:P1}");

            Section("input validation (was silent memory corruption)");
            CheckThrows<ArgumentOutOfRangeException>(
                "id == maxVectors is rejected",
                () => engine.Insert(MaxVectors, vectors[1], "overflow"));
            CheckThrows<ArgumentOutOfRangeException>(
                "negative id is rejected",
                () => engine.Insert(-1, vectors[1], "underflow"));
            CheckThrows<ArgumentException>(
                "all-zero vector is rejected (cosine distance would be NaN)",
                () => engine.Insert(n + 1, new float[Dimensions], "zero"));
            CheckThrows<ArgumentException>(
                "NaN component is rejected",
                () => engine.Insert(n + 2, MakeNaNVector(Dimensions), "nan"));
            CheckThrows<ArgumentException>(
                "wrong dimensionality is rejected on insert",
                () => engine.Insert(n + 3, new float[Dimensions + 1], "wrong dim"));
            CheckThrows<ArgumentException>(
                "wrong dimensionality is rejected on search",
                () => engine.Index.Search(new float[Dimensions - 1], 5));

            Section("re-index an existing id (was IndexOutOfRangeException ~5.8% of the time)");
            bool reindexOk = true;
            string reindexDetail = "";
            try
            {
                var rng = new Random(1234);
                for (int t = 0; t < 600; t++)
                {
                    int target = rng.Next(n);
                    float[] replacement = MakeClusteredVectors(1, Dimensions, 1, 0.9f, seed: 9000 + t)[0];
                    vectors[target] = replacement;
                    engine.Insert(target, replacement, $"Document_{target}_v2");

                    // Traverse afterwards: the old failure surfaced on the next search
                    // through a stale higher-layer edge, not at insert time.
                    engine.Index.Search(replacement, 10);
                }
            }
            catch (Exception ex)
            {
                reindexOk = false;
                reindexDetail = $"{ex.GetType().Name}: {ex.Message}";
            }
            Check("600 re-inserts + searches complete without throwing", reindexOk, reindexDetail);
            Check("re-index does not change the node count", engine.Count == n, $"{engine.Count} != {n}");

            var updated = engine.ExecuteSearchWithMetadata(vectors[7], 10, efSearch: 100);
            Check("a re-indexed vector still finds itself",
                updated.Any(r => r.Id == 7),
                $"got [{string.Join(",", updated.Select(r => r.Id))}]");

            Section("oversized k (was an uncatchable stack overflow)");
            bool bigKOk = true;
            int bigKCount = -1;
            string bigKDetail = "";
            try
            {
                bigKCount = engine.Index.Search(vectors[3], 300_000).Length;
            }
            catch (Exception ex)
            {
                bigKOk = false;
                bigKDetail = $"{ex.GetType().Name}: {ex.Message}";
            }
            Check("k = 300,000 returns without crashing", bigKOk, bigKDetail);
            Check("k = 300,000 is clamped to the index size", bigKOk && bigKCount <= n, $"{bigKCount} > {n}");

            Section("concurrent insert (was a lock-order deadlock + racy Random)");
            bool concurrentOk = false;
            string concurrentDetail = "timed out - possible deadlock in Connect";
            try
            {
                var task = Task.Run(() =>
                {
                    Parallel.For(0, 3_000, i =>
                    {
                        int id = 5_000 + i;
                        float[] v = MakeClusteredVectors(1, Dimensions, 1, 0.9f, seed: 70_000 + i)[0];
                        engine.Insert(id, v, $"Parallel_{id}");
                    });
                });
                if (task.Wait(TimeSpan.FromSeconds(120)))
                {
                    concurrentOk = true;
                    concurrentDetail = "";
                }
            }
            catch (Exception ex)
            {
                concurrentDetail = $"{ex.GetType().Name}: {ex.InnerException?.Message ?? ex.Message}";
            }
            Check("3,000 parallel inserts complete", concurrentOk, concurrentDetail);
            Check("all parallel inserts are present", engine.Count == n + 3_000, $"{engine.Count} != {n + 3_000}");

            Section("allocation on the query path");
            engine.ExecuteSearchWithMetadata(vectors[0], 5); // JIT + pool warmup
            engine.Index.Search(vectors[0], 5);
            long before = GC.GetAllocatedBytesForCurrentThread();
            engine.Index.Search(vectors[0], 5);
            long searchOnly = GC.GetAllocatedBytesForCurrentThread() - before;

            before = GC.GetAllocatedBytesForCurrentThread();
            engine.ExecuteSearchWithMetadata(vectors[0], 5);
            long withMetadata = GC.GetAllocatedBytesForCurrentThread() - before;

            Console.WriteLine($"  Search(k=5)                      : {searchOnly,6:N0} bytes");
            Console.WriteLine($"  ExecuteSearchWithMetadata(k=5)   : {withMetadata,6:N0} bytes");
            Console.WriteLine("  (the traversal itself is pooled; what remains is the returned array)");
            Check("traversal allocates only the result array", searchOnly <= 256, $"{searchOnly} bytes");

            Section("persistence");
            engine.Save();
            Check("graph file was written", File.Exists(graphFile), graphFile);
        }

        // Reopen from disk in a fresh engine.
        using (var reopened = new VectorEngine(vectorFile, metaFile, Dimensions, maxVectors: MaxVectors))
        {
            Check("graph reloaded from disk", reopened.LoadedFromDisk, "built fresh instead");
            Check("reloaded index has the same node count", reopened.Count == n + 3_000,
                $"{reopened.Count} != {n + 3_000}");

            var hits = reopened.ExecuteSearchWithMetadata(vectors[0], 5);
            Check("search works after reload", hits.Length > 0 && hits[0].Id == 0,
                hits.Length == 0 ? "no results (index was memory-only)" : $"got id {hits[0].Id}");
            Check("metadata survived reload", hits.Length > 0 && hits[0].Metadata.StartsWith("Document_0"),
                hits.Length > 0 ? $"'{hits[0].Metadata}'" : "no results");
        }

        Section("on-disk footprint");
        var info = new FileInfo(vectorFile);
        long expected = (long)MaxVectors * Dimensions * sizeof(float);
        Console.WriteLine($"  vectors.bin logical size : {info.Length / (1024.0 * 1024.0):N1} MiB " +
                          $"(reserved for {MaxVectors:N0} slots)");
        Console.WriteLine($"  graph file               : {new FileInfo(graphFile).Length / 1024.0:N0} KiB");
        Check("file matches the reserved capacity", info.Length == expected, $"{info.Length} != {expected}");

        Console.WriteLine();
        Console.WriteLine(new string('=', 78));
        Console.WriteLine(_failed == 0
            ? $"ALL {_passed} CHECKS PASSED"
            : $"{_passed} passed, {_failed} FAILED");
        return _failed == 0 ? 0 : 1;
    }

    // ---------------------------------------------------------------- helpers

    private static void Section(string title)
    {
        Console.WriteLine();
        Console.WriteLine($"-- {title} " + new string('-', Math.Max(0, 74 - title.Length)));
    }

    private static void Check(string what, bool ok, string detail)
    {
        if (ok)
        {
            _passed++;
            Console.WriteLine($"  PASS  {what}");
        }
        else
        {
            _failed++;
            Console.WriteLine($"  FAIL  {what}" + (string.IsNullOrEmpty(detail) ? "" : $"  [{detail}]"));
        }
    }

    private static void CheckThrows<TException>(string what, Action action) where TException : Exception
    {
        try
        {
            action();
            Check(what, false, $"no exception (expected {typeof(TException).Name})");
        }
        catch (TException)
        {
            Check(what, true, "");
        }
        catch (Exception ex)
        {
            Check(what, false, $"threw {ex.GetType().Name}, expected {typeof(TException).Name}");
        }
    }

    private static float[] MakeNaNVector(int dim)
    {
        var v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = 0.1f;
        v[dim / 2] = float.NaN;
        return v;
    }

    /// <summary>
    /// Clustered unit vectors. Uniform Gaussians are the easy case for HNSW - the graph
    /// scores ~100% recall either way - so pruning quality only shows up on data with
    /// cluster structure, which is what real embeddings have.
    /// </summary>
    private static float[][] MakeClusteredVectors(int n, int dim, int clusters, float spread, int seed)
    {
        var rand = new Random(seed);
        var centers = new float[clusters][];
        for (int c = 0; c < clusters; c++) centers[c] = Normalize(Gaussian(rand, dim));

        var result = new float[n][];
        for (int i = 0; i < n; i++)
        {
            float[] center = centers[rand.Next(clusters)];
            float[] noise = Gaussian(rand, dim);
            var v = new float[dim];
            for (int d = 0; d < dim; d++) v[d] = center[d] + spread * noise[d];
            result[i] = Normalize(v);
        }
        return result;
    }

    private static float[] Gaussian(Random rand, int dim)
    {
        var v = new float[dim];
        for (int d = 0; d < dim; d++)
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = rand.NextDouble();
            v[d] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return v;
    }

    private static float[] Normalize(float[] v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++) sum += v[i] * (double)v[i];
        float norm = (float)Math.Sqrt(sum);
        if (norm == 0) { v[0] = 1f; return v; }
        for (int i = 0; i < v.Length; i++) v[i] /= norm;
        return v;
    }

    private static double MeasureRecall(
        VectorEngine engine, float[][] vectors, int n, int k, int queries, int efSearch = 0)
    {
        var rand = new Random(7);
        int hits = 0;
        var scored = new (int Id, float Score)[n];

        for (int q = 0; q < queries; q++)
        {
            int qi = rand.Next(n);
            ReadOnlySpan<float> query = vectors[qi];

            for (int i = 0; i < n; i++)
            {
                scored[i] = (i, VectorMath.CosineSimilarity(query, vectors[i]));
            }
            int[] exact = scored.OrderByDescending(s => s.Score).Take(k).Select(s => s.Id).ToArray();
            int[] got = engine.Index.Search(query, k, efSearch);

            hits += got.Intersect(exact).Count();
        }

        return hits / (double)(queries * k);
    }
}
