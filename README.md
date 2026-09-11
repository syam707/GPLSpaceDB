# VDAlgorithmicEngine

A high-performance vector database engine for Agentic AI architectures, built from
scratch with **C# 14** and **.NET 10**. Vectors live in a memory-mapped file, search
runs over a Hierarchical Navigable Small World (HNSW) graph, and the query path avoids
managed allocation.

## Key Features

1. **Pooled, non-allocating query path.**
   `Search` traverses the graph using `ref struct` min/max heaps over `ArrayPool`
   buffers, a generation-stamped visited set, and `CollectionsMarshal.AsSpan` for
   adjacency reads. Nothing on the traversal itself reaches the managed heap; the only
   allocation a query makes is the array it hands back to you.

2. **Memory-mapped vector storage.**
   Vectors are stored at a fixed stride in a memory-mapped file and read back as
   `ReadOnlySpan<float>` over the mapping, so the GC never scans or tracks the vector
   data. Storage survives process restart, and so does the graph (see *Persistence*).

3. **SIMD hardware acceleration.**
   Distance operations use `System.Numerics.Tensors.TensorPrimitives`, which dispatches
   to AVX2 / AVX-512 where the CPU supports it.

4. **HNSW neighbour-selection heuristic.**
   Pruning uses Algorithm 4 from Malkov & Yashunin rather than simply keeping the
   nearest *m* neighbours, which preserves the long-range links that keep the graph
   navigable on clustered data.

## Codebase Structure

The core library is `VDAlgorithmicEngine`.

| File | Role |
| --- | --- |
| `VectorEngine.cs` | Facade binding store + metadata + index; start here. |
| `IVectorIndex.cs` | The index contract. |
| `HnswIndex.cs` | The HNSW graph: insert, search, pruning, save/load. |
| `PersistentVectorStore.cs` | Memory-mapped, fixed-stride vector storage. |
| `MetadataStore.cs` | Append-only id-to-string log with a read cache. |
| `VectorNode.cs` | One vector's per-layer adjacency. |
| `VectorMath.cs` | SIMD distance wrappers and vector validation. |
| `BoundedPriorityQueue.cs` | Allocation-free `ref struct` min/max heaps. |
| `VisitedSet.cs` | Generation-stamped visited set + its pool. |
| `NativeMemoryManager.cs` | Wraps unmanaged allocations as `Memory<float>`. |

`VDTest` is the regression suite. `VectorDatabase` is an empty placeholder project.

---

## How to Use the Library

### 1. Reference the project

Add a project reference to `VDAlgorithmicEngine`. Standard usage needs no unsafe
blocks in your own project.

### 2. Open an engine

`VectorEngine` is the entry point. It owns the vector file, the metadata log and the
graph, and reloads all three if they already exist on disk.

```csharp
using VDAlgorithmicEngine;

using var engine = new VectorEngine(
    vectorFilePath:   "vectors.bin",
    metadataFilePath: "metadata.dat",
    dimensions:       128,
    m:                16,     // connections per node per layer
    efConstruction:   100,    // candidate breadth while building
    maxVectors:       1_000_000);
```

**`maxVectors` matters.** The vector file is preallocated to
`maxVectors * dimensions * 4` bytes, and it is also the hard ceiling on ids. The
default of 10,000,000 at 128 dimensions reserves a 5.12 GB file. On Windows the file is
flagged sparse before it is sized, so unwritten slots cost no physical disk — but set
`maxVectors` to what you actually need rather than relying on that.

**Ids are slot indices.** They must be dense and within `[0, maxVectors)`. Do not use
hashes or database primary keys directly; keep your own id mapping if you need one.
An out-of-range id throws `ArgumentOutOfRangeException`.

### 3. Insert vectors

```csharp
float[] vectorData = { 0.1f, 0.8f, -0.4f, /* ... 128 dimensions ... */ };

engine.Insert(id: 42, vector: vectorData, metadata: "doc-42");
```

`Insert` writes the vector to the store, records the metadata and links the node into
the graph. Re-calling it with an id that already exists re-indexes that vector in
place, which is how you update an embedding.

Vectors must have a finite, non-zero magnitude — cosine distance is undefined for a
zero vector, and a NaN distance would corrupt ranking, so both are rejected with
`ArgumentException` rather than being allowed into the index.

If you prefer to drive the index directly, `HnswIndex` takes the store in its
constructor and persists vectors itself:

```csharp
using var store = new PersistentVectorStore("vectors.bin", dimensions: 128, maxVectors: 100_000);
IVectorIndex index = new HnswIndex(store, m: 16, efConstruction: 100);

index.Insert(42, vectorData);            // writes to `store` as well as indexing
int[] nearest = index.Search(queryVector, kNeighbors: 5);
```

### 4. Query

```csharp
// Ids only.
int[] nearestIds = engine.Index.Search(queryVector, kNeighbors: 5);

// Ids with cosine similarity and metadata.
VectorSearchResult[] results = engine.ExecuteSearchWithMetadata(queryVector, k: 5);

foreach (var result in results)
{
    Console.WriteLine($"{result.Id}  score={result.Score:F4}  {result.Metadata}");
}
```

Recall is tunable per query. `efSearch` defaults to `max(k, 50)`; raise it to trade
latency for accuracy without rebuilding:

```csharp
var accurate = engine.ExecuteSearchWithMetadata(queryVector, k: 10, efSearch: 200);
```

### 5. Persistence

The vector file and metadata log are written as you go, but the graph is only written
when you ask:

```csharp
engine.Save();   // graph topology + flush both stores
```

Reopening a `VectorEngine` against the same paths reloads the graph automatically;
`engine.LoadedFromDisk` tells you whether it did. Without a saved graph the stored
vectors are still on disk but nothing is reachable, so **call `Save()` before exit** or
you will have to rebuild.

Save with inserts quiesced. The node set is snapshotted, but concurrent linking can
still change adjacency while the file is being written.

## Concurrency

Concurrent `Insert` and concurrent `Search` are supported. Adjacency is guarded per
node-layer, edge updates take their two locks in a fixed id order, and the entry point
is published as a single immutable pair, so readers never see a half-updated graph.

As with every HNSW implementation, concurrent inserts build a slightly different graph
than the same inserts applied serially; recall is equivalent, exact topology is not
reproducible. For a byte-reproducible index, build single-threaded.

## Performance & Tuning Notes

- **Dimensions are fixed per store.** Every vector and every query must match
  `dimensions`; mismatches throw rather than silently producing a wrong distance.
- **Warm up before measuring.** JIT and the first `ArrayPool` rent allocate. Run a
  throwaway `Search` at startup before benchmarking or asserting on allocation.
- **`m` and `efConstruction` cost build time, not query time.** Pruning with the
  diversity heuristic is O(m²) distance computations per shrink, so raising `m`
  makes inserts meaningfully slower.
- **Pre-normalising helps.** If your embeddings are already unit vectors, cosine
  similarity reduces to a dot product; `VectorMath.Dot` skips recomputing both
  magnitudes on every comparison.
- **Metadata is an append-only log.** It is never compacted, so repeatedly updating one
  id grows the file. Pass `flushOnWrite: false` to `MetadataStore` for bulk loads and
  `Flush()` once at the end — the per-record flush is a syscall per insert.

## Running the tests

```
dotnet run -c Release --project VDTest
```

The suite covers retrieval correctness, recall against brute force on clustered data,
input validation, in-place re-indexing, oversized `k`, concurrent insert, query-path
allocation, and a save/reload round trip. It exits non-zero on failure.
