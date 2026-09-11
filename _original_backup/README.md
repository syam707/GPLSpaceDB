# VDAlgorithmicEngine

A high-performance, in-memory Vector Database algorithmic engine tailored for Agentic AI architectures. Built from scratch with **C# 14** and **.NET 10**, this library provides extremely fast, multi-layered graph traversals (HNSW) for vector similarity search.

## Key Features

1. **Zero GC Allocation on Hot Paths**: 
   The most critical feature of this engine is its query efficiency. The `Search` method uses carefully crafted `ref struct` Min/Max heaps, custom `ArrayPool`-backed hash sets, and `CollectionsMarshal.AsSpan` combined with `stackalloc` to traverse the index without allocating a single byte on the managed heap (excluding the final returned integer array).
2. **Unmanaged Memory Management**: 
   Vectors are stored entirely in unmanaged memory using `System.Runtime.InteropServices.NativeMemory`. They are safely exposed to the managed runtime using a custom `NativeMemoryManager<float>`, preventing the .NET Garbage Collector from needing to track or scan millions of vector float arrays.
3. **SIMD Hardware Acceleration**: 
   Vector distance operations (Cosine Similarity and L2 Distance) utilize `System.Numerics.Tensors.TensorPrimitives`. This takes full advantage of CPU hardware intrinsics (AVX2 / AVX-512) for lightning-fast mathematical computations.

## Codebase Structure

The core library is contained in the `VDAlgorithmicEngine` project.
- `IVectorIndex.cs`: The core interface implemented by the engine.
- `HnswIndex.cs`: The main engine implementing Hierarchical Navigable Small World graphs.
- `VectorNode.cs`: Represents a single vector's position and connections across various graph layers.
- `NativeMemoryManager.cs`: Wraps unmanaged vector allocations into a safe `Memory<float>` wrapper.
- `VectorMath.cs`: Provides static utility wrappers for SIMD math primitives.
- `BoundedPriorityQueue.cs`: Contains `ref struct` custom priority queues used for search heuristics without allocating heap memory.

---

## How to Use the Library

### 1. Reference the Project
Add the `VDAlgorithmicEngine` project reference to your application. Since it utilizes unmanaged pointers safely, ensure your consuming project supports unsafe blocks if you interact with the memory wrappers directly, though standard usage only requires standard .NET 10 features.

### 2. Initializing the Engine
Instantiate the `HnswIndex` with your preferred tuning parameters:
- `m`: Maximum number of connections a node can have per layer. (Higher means more accurate but slower inserts).
- `efConstruction`: Size of the dynamic candidate list during graph construction.

```csharp
using VDAlgorithmicEngine;

// Use default parameters (m = 16, efConstruction = 100)
IVectorIndex index = new HnswIndex();

// Or tune for higher accuracy
IVectorIndex preciseIndex = new HnswIndex(m: 32, efConstruction: 200);
```

### 3. Inserting Vectors
Vectors are represented as `ReadOnlySpan<float>`. The unique integer `id` you provide is what the search engine will return during a query.

```csharp
int vectorId = 42;
float[] vectorData = { 0.1f, 0.8f, -0.4f, /* ... up to N dimensions ... */ };

// Insert into the database
index.Insert(vectorId, vectorData);
```
*Note: Depending on your environment, you may wish to pre-normalize your vectors if you are solely relying on Cosine Distance approximations.*

### 4. Querying Nearest Neighbors
To find the `k` most similar vectors to a query vector, use the `Search` method.

```csharp
float[] queryVector = { 0.15f, 0.82f, -0.38f };
int kNeighbors = 5;

// This call executes with Zero GC pressure!
int[] nearestIds = index.Search(queryVector, kNeighbors);

Console.WriteLine("Closest Vectors:");
foreach (var id in nearestIds)
{
    Console.WriteLine($"ID: {id}");
}
```

## Performance & Optimization Notes

- **Dimension Limits:** The engine works best with dense vectors of identical dimensions. Ensure that every inserted vector matches the dimensionality of your query vector, otherwise distance calculations may fault or yield incorrect approximations.
- **Warming up:** The underlying object pools and JIT compilation may allocate slightly on the very first query. To ensure absolute zero-allocation on strict hot paths, run a dummy `Search` query after system startup to initialize all static generic pools.
