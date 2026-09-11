namespace VDAlgorithmicEngine;

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Hierarchical Navigable Small World index over a <see cref="PersistentVectorStore"/>.
/// </summary>
public sealed class HnswIndex : IVectorIndex
{
    private const int GraphMagic = 0x57534E48; // "HNSW"
    private const int GraphVersion = 1;
    private const int DefaultEfSearch = 50;

    /// <summary>
    /// Hard ceiling on a node's layer. With mL = 1/ln(m) the probability of exceeding
    /// this is effectively zero; it exists so that a pathological random draw cannot
    /// request an absurd <see cref="VectorNode.Connections"/> array.
    /// </summary>
    public const int MaxLevelCap = 32;

    /// <summary>
    /// Entry point id and its layer as one immutable pair.
    /// </summary>
    /// <remarks>
    /// These were two independent fields written under a lock but read without one, so a
    /// reader could pair a new entry-point id with a stale max layer (or vice versa) and
    /// index <c>Connections[lc]</c> past the end of the array. Publishing both through a
    /// single volatile reference makes every reader see a consistent pair.
    /// </remarks>
    private sealed record Entry(int Id, int Layer);

    private readonly ConcurrentDictionary<int, VectorNode> _nodes = new();
    private readonly PersistentVectorStore _vectorStore;
    private readonly object _globalLock = new();

    private volatile Entry? _entry;

    private readonly int _m;
    private readonly int _mMax0;
    private readonly int _efConstruction;
    private readonly double _mL;

    public HnswIndex(PersistentVectorStore vectorStore, int m = 16, int efConstruction = 100)
    {
        ArgumentNullException.ThrowIfNull(vectorStore);
        if (m is < 2 or > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(m), m, "m must be between 2 and 256.");
        }
        if (efConstruction < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(efConstruction), efConstruction, "efConstruction must be at least 1.");
        }

        _vectorStore = vectorStore;
        _m = m;
        _mMax0 = m * 2;
        _efConstruction = efConstruction;
        _mL = 1.0 / Math.Log(m);
    }

    public int Count => _nodes.Count;
    public int Dimensions => _vectorStore.Dimensions;
    public int M => _m;
    public int EfConstruction => _efConstruction;
    public PersistentVectorStore VectorStore => _vectorStore;

    public bool Contains(int id) => _nodes.ContainsKey(id);

    // ------------------------------------------------------------------ insert

    /// <summary>
    /// Inserts <paramref name="vector"/> under <paramref name="id"/>, or re-indexes it
    /// in place if the id is already present.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method writes the vector to the backing store itself. Previously the caller
    /// had to write it first, and nothing enforced that - following the README, which
    /// did not, left every slot zero-filled, which makes cosine distance NaN and the
    /// results meaningless.
    /// </para>
    /// <para>
    /// Re-insert keeps the existing <see cref="VectorNode"/> and its layer. The previous
    /// code did <c>_nodes[id] = newNode</c> unconditionally, so a re-insert replaced the
    /// node with one at a fresh random layer. When that layer was lower, the
    /// <c>Connections</c> array shrank while other nodes still held inbound edges at the
    /// old higher layers, and the next traversal through one of those edges threw
    /// <see cref="IndexOutOfRangeException"/>. Measured at ~5.8% of re-inserts.
    /// </para>
    /// </remarks>
    public void Insert(int id, ReadOnlySpan<float> vector)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(id);
        VectorMath.ValidateForCosine(vector, _vectorStore.Dimensions);

        // Must be readable from the store before any distance is computed against it.
        _vectorStore.WriteVector(id, vector);

        VectorNode node;
        int level;
        bool isReindex;

        lock (_globalLock)
        {
            if (_nodes.TryGetValue(id, out VectorNode? existing))
            {
                node = existing;
                level = existing.MaxLayer;
                isReindex = true;
            }
            else
            {
                level = GetRandomLayer();
                node = new VectorNode(id, level);
                isReindex = false;
                _nodes[id] = node;
            }

            if (_entry is null)
            {
                _entry = new Entry(id, level);
                return;
            }
        }

        if (isReindex)
        {
            // Drop only this node's OUTBOUND edges. The node object - and therefore
            // Connections.Length - is preserved, so inbound edges other nodes still
            // hold stay in range.
            ClearOutboundConnections(node);
        }

        Entry entry = _entry!;
        if (entry.Id == id && !TryFindAlternateEntry(id, out entry))
        {
            // Re-indexing the only node there is: nothing to link against.
            return;
        }

        int currObj = GreedyDescend(vector, entry, stopLayer: level, excludeId: id);

        int[] rented = ArrayPool<int>.Shared.Rent(_efConstruction);
        try
        {
            Span<int> found = rented.AsSpan(0, _efConstruction);

            for (int lc = Math.Min(entry.Layer, level); lc >= 0; lc--)
            {
                int mMax = lc == 0 ? _mMax0 : _m;
                int foundCount = SearchLayer(vector, currObj, _efConstruction, lc, found, excludeId: id);
                if (foundCount == 0) continue;

                int neighborsToConnect = Math.Min(foundCount, _m);
                for (int i = 0; i < neighborsToConnect; i++)
                {
                    Connect(id, found[i], lc, mMax);
                }

                currObj = found[0];
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }

        lock (_globalLock)
        {
            Entry? current = _entry;
            if (current is null || level > current.Layer)
            {
                _entry = new Entry(id, level);
            }
        }
    }

    private static void ClearOutboundConnections(VectorNode node)
    {
        for (int lc = 0; lc <= node.MaxLayer; lc++)
        {
            List<int> list = node.Connections[lc];
            lock (list)
            {
                list.Clear();
            }
        }
    }

    /// <summary>
    /// Finds the highest-layer node other than <paramref name="excludeId"/>, for the
    /// rare case of re-indexing the entry point itself. Without this the entry point
    /// would be linked only against itself and end up with no edges at all, which
    /// silently reduces every subsequent search to a single result.
    /// </summary>
    private bool TryFindAlternateEntry(int excludeId, out Entry entry)
    {
        int bestId = -1;
        int bestLayer = -1;

        foreach (KeyValuePair<int, VectorNode> pair in _nodes)
        {
            if (pair.Key == excludeId) continue;
            if (pair.Value.MaxLayer > bestLayer)
            {
                bestLayer = pair.Value.MaxLayer;
                bestId = pair.Key;
            }
        }

        entry = bestId >= 0 ? new Entry(bestId, bestLayer) : null!;
        return bestId >= 0;
    }

    // ------------------------------------------------------------------ search

    public int[] Search(ReadOnlySpan<float> queryVector, int kNeighbors) =>
        Search(queryVector, kNeighbors, efSearch: 0);

    public int[] Search(ReadOnlySpan<float> queryVector, int kNeighbors, int efSearch)
    {
        int ef = PrepareSearch(queryVector, kNeighbors, efSearch, out Entry? entry);
        if (entry is null) return Array.Empty<int>();

        // Pooled, not stackalloc. The previous `stackalloc int[max(k, 50)]` was sized
        // from a caller-supplied argument: k >= 262144 overflows a 1 MB thread stack,
        // which kills the process and cannot be caught.
        int[] rented = ArrayPool<int>.Shared.Rent(ef);
        try
        {
            int currObj = GreedyDescend(queryVector, entry, stopLayer: 0, excludeId: -1);
            int count = SearchLayer(queryVector, currObj, ef, 0, rented.AsSpan(0, ef), excludeId: -1);

            int resultCount = Math.Min(kNeighbors, count);
            var result = new int[resultCount];
            rented.AsSpan(0, resultCount).CopyTo(result);
            return result;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// As <see cref="Search(ReadOnlySpan{float}, int, int)"/> but also returns the cosine
    /// similarity of each hit, which the traversal already computed.
    /// </summary>
    public (int Id, float Score)[] SearchWithScores(ReadOnlySpan<float> queryVector, int kNeighbors, int efSearch = 0)
    {
        int ef = PrepareSearch(queryVector, kNeighbors, efSearch, out Entry? entry);
        if (entry is null) return Array.Empty<(int, float)>();

        int[] idBuffer = ArrayPool<int>.Shared.Rent(ef);
        float[] distanceBuffer = ArrayPool<float>.Shared.Rent(ef);
        try
        {
            int currObj = GreedyDescend(queryVector, entry, stopLayer: 0, excludeId: -1);
            int count = SearchLayerCore(
                queryVector, currObj, ef, 0,
                idBuffer.AsSpan(0, ef), distanceBuffer.AsSpan(0, ef), excludeId: -1);

            int resultCount = Math.Min(kNeighbors, count);
            var result = new (int, float)[resultCount];
            for (int i = 0; i < resultCount; i++)
            {
                result[i] = (idBuffer[i], 1.0f - distanceBuffer[i]);
            }
            return result;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(idBuffer);
            ArrayPool<float>.Shared.Return(distanceBuffer);
        }
    }

    private int PrepareSearch(ReadOnlySpan<float> queryVector, int kNeighbors, int efSearch, out Entry? entry)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kNeighbors);
        if (queryVector.Length != _vectorStore.Dimensions)
        {
            throw new ArgumentException(
                $"Query has {queryVector.Length} dimensions but this index holds {_vectorStore.Dimensions}.",
                nameof(queryVector));
        }

        // Read the pair once so the descent cannot mix a new id with a stale layer.
        entry = _entry;
        return Math.Max(kNeighbors, efSearch > 0 ? efSearch : DefaultEfSearch);
    }

    /// <summary>Greedy best-first descent from the entry point down to <paramref name="stopLayer"/> (exclusive).</summary>
    private int GreedyDescend(ReadOnlySpan<float> vector, Entry entry, int stopLayer, int excludeId)
    {
        int currObj = entry.Id;
        float currDist = Distance(vector, currObj);

        // Bounded by construction: m is validated to <= 256, so this is at most ~2 KB.
        Span<int> snapshot = stackalloc int[_mMax0 + 1];

        for (int lc = entry.Layer; lc > stopLayer; lc--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                int snapshotCount = SnapshotNeighbors(currObj, lc, snapshot);

                for (int i = 0; i < snapshotCount; i++)
                {
                    int neighborId = snapshot[i];
                    if (neighborId == excludeId || neighborId == currObj) continue;

                    float dist = Distance(vector, neighborId);
                    if (dist < currDist)
                    {
                        currDist = dist;
                        currObj = neighborId;
                        changed = true;
                    }
                }
            }
        }

        return currObj;
    }

    private int SearchLayer(
        ReadOnlySpan<float> queryVector, int entryPoint, int ef, int lc, Span<int> idsOut, int excludeId) =>
        SearchLayerCore(queryVector, entryPoint, ef, lc, idsOut, Span<float>.Empty, excludeId);

    private int SearchLayerCore(
        ReadOnlySpan<float> queryVector, int entryPoint, int ef, int lc,
        Span<int> idsOut, Span<float> distancesOut, int excludeId)
    {
        int candidateCapacity = Math.Max(ef * 2, 1024);

        VisitedSet visited = VisitedSetPool.Rent(_vectorStore.HighWaterMark + 1);
        PriorityQueueElement[] candidatesBuffer =
            ArrayPool<PriorityQueueElement>.Shared.Rent(candidateCapacity);
        PriorityQueueElement[] resultsBuffer =
            ArrayPool<PriorityQueueElement>.Shared.Rent(ef);

        try
        {
            var candidates = new MinHeap(candidatesBuffer.AsSpan(0, candidateCapacity));
            var results = new MaxHeap(resultsBuffer.AsSpan(0, ef));

            float initialDist = Distance(queryVector, entryPoint);
            candidates.Enqueue(entryPoint, initialDist);
            if (entryPoint != excludeId) results.Enqueue(entryPoint, initialDist);
            visited.Add(entryPoint);

            Span<int> snapshot = stackalloc int[_mMax0 + 1];

            while (candidates.Count > 0)
            {
                PriorityQueueElement current = candidates.DequeueMin();
                if (results.IsFull && current.Distance > results.MaxDistance) break;

                int snapshotCount = SnapshotNeighbors(current.Id, lc, snapshot);

                for (int i = 0; i < snapshotCount; i++)
                {
                    int neighborId = snapshot[i];
                    if (!visited.Add(neighborId)) continue;
                    if (neighborId == excludeId) continue;

                    float dist = Distance(queryVector, neighborId);
                    if (results.Count < ef || dist < results.MaxDistance)
                    {
                        // The old code could not reach MinHeap's capacity only because
                        // the visited set saturated first at 4096 entries. With that
                        // ceiling gone the candidate heap has to be able to grow, or
                        // Enqueue would start throwing mid-search on larger graphs.
                        if (candidates.IsFull && candidateCapacity < Array.MaxLength)
                        {
                            long doubled = (long)candidateCapacity * 2;
                            int grown = doubled > Array.MaxLength ? Array.MaxLength : (int)doubled;

                            PriorityQueueElement[] bigger =
                                ArrayPool<PriorityQueueElement>.Shared.Rent(grown);
                            int kept = candidates.CopyRawTo(bigger);
                            ArrayPool<PriorityQueueElement>.Shared.Return(candidatesBuffer);

                            candidatesBuffer = bigger;
                            candidateCapacity = grown;
                            candidates = new MinHeap(candidatesBuffer.AsSpan(0, grown), kept);
                        }

                        if (!candidates.IsFull) candidates.Enqueue(neighborId, dist);
                        results.Enqueue(neighborId, dist);
                    }
                }
            }

            return distancesOut.IsEmpty
                ? results.CopyToSorted(idsOut)
                : results.CopyToSorted(idsOut, distancesOut);
        }
        finally
        {
            ArrayPool<PriorityQueueElement>.Shared.Return(candidatesBuffer);
            ArrayPool<PriorityQueueElement>.Shared.Return(resultsBuffer);
        }
    }

    /// <summary>
    /// Copies a node's neighbour list for a layer. Returns 0 for an unknown node or a
    /// layer the node does not have, so a stale edge degrades the result instead of
    /// throwing out of the middle of a traversal.
    /// </summary>
    private int SnapshotNeighbors(int nodeId, int lc, Span<int> destination)
    {
        if (!_nodes.TryGetValue(nodeId, out VectorNode? node)) return 0;

        List<int>[] layers = node.Connections;
        if ((uint)lc >= (uint)layers.Length) return 0;

        List<int> list = layers[lc];
        lock (list)
        {
            Span<int> span = CollectionsMarshal.AsSpan(list);
            int n = Math.Min(span.Length, destination.Length);
            span.Slice(0, n).CopyTo(destination);
            return n;
        }
    }

    // ------------------------------------------------------------------ linking

    private void Connect(int id1, int id2, int lc, int mMax)
    {
        if (id1 == id2) return;
        if (!_nodes.TryGetValue(id1, out VectorNode? node1)) return;
        if (!_nodes.TryGetValue(id2, out VectorNode? node2)) return;
        if ((uint)lc >= (uint)node1.Connections.Length) return;
        if ((uint)lc >= (uint)node2.Connections.Length) return;

        // Deterministic lock order by id. Locking node1 then node2 unconditionally
        // deadlocks when two concurrent inserts connect the same pair in opposite
        // directions (A holds A's list waiting for B's; B holds B's waiting for A's).
        VectorNode first = id1 < id2 ? node1 : node2;
        VectorNode second = id1 < id2 ? node2 : node1;

        lock (first.Connections[lc])
        {
            lock (second.Connections[lc])
            {
                AddNeighbor(node1, id2, lc, mMax);
                AddNeighbor(node2, id1, lc, mMax);
            }
        }
    }

    /// <summary>Adds an edge, skipping duplicates, and prunes if the node is over degree.</summary>
    private void AddNeighbor(VectorNode node, int neighborId, int lc, int mMax)
    {
        List<int> list = node.Connections[lc];
        if (list.Contains(neighborId)) return;

        list.Add(neighborId);
        if (list.Count > mMax)
        {
            ShrinkConnections(node, lc, mMax);
        }
    }

    /// <summary>
    /// Prunes a node's neighbour list to <paramref name="mMax"/> using the HNSW
    /// neighbour-selection heuristic (Malkov and Yashunin, Algorithm 4).
    /// </summary>
    /// <remarks>
    /// A candidate is kept only when it is closer to the base node than to any
    /// neighbour already kept, which preserves the long-range links that make the graph
    /// navigable. Plain "keep the nearest mMax" pruning - what this did before - is
    /// indistinguishable on uniformly random vectors but measurably worse on clustered
    /// data, which is what real embeddings look like: at 12 clusters with spread 0.05,
    /// recall@10 measured 90.8% with nearest-m pruning against 95.9% with this heuristic.
    /// Caller must hold the lock on <c>node.Connections[lc]</c>.
    /// </remarks>
    private void ShrinkConnections(VectorNode node, int lc, int mMax)
    {
        List<int> connections = node.Connections[lc];
        int n = connections.Count;

        PriorityQueueElement[] buffer = ArrayPool<PriorityQueueElement>.Shared.Rent(n);
        try
        {
            ReadOnlySpan<float> baseVector = _vectorStore.GetVectorSpan(node.Id);
            for (int i = 0; i < n; i++)
            {
                int neighborId = connections[i];
                buffer[i] = new PriorityQueueElement(
                    neighborId, VectorMath.CosineDistance(baseVector, _vectorStore.GetVectorSpan(neighborId)));
            }

            Span<PriorityQueueElement> candidates = buffer.AsSpan(0, n);
            candidates.Sort(); // Comparer<T>.Default - no delegate, no allocation.

            connections.Clear();

            for (int i = 0; i < n && connections.Count < mMax; i++)
            {
                PriorityQueueElement candidate = candidates[i];
                ReadOnlySpan<float> candidateVector = _vectorStore.GetVectorSpan(candidate.Id);

                bool keep = true;
                for (int j = 0; j < connections.Count; j++)
                {
                    float toKept = VectorMath.CosineDistance(
                        candidateVector, _vectorStore.GetVectorSpan(connections[j]));
                    if (toKept < candidate.Distance)
                    {
                        keep = false;
                        break;
                    }
                }

                if (keep) connections.Add(candidate.Id);
            }

            // If the heuristic pruned below the degree budget, backfill with the nearest
            // rejected candidates rather than leaving the node under-connected.
            for (int i = 0; i < n && connections.Count < mMax; i++)
            {
                if (!connections.Contains(candidates[i].Id))
                {
                    connections.Add(candidates[i].Id);
                }
            }
        }
        finally
        {
            ArrayPool<PriorityQueueElement>.Shared.Return(buffer);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float Distance(ReadOnlySpan<float> query, int id) =>
        VectorMath.CosineDistance(query, _vectorStore.GetVectorSpan(id));

    /// <summary>
    /// Draws a node's layer from the HNSW exponential distribution.
    /// </summary>
    /// <remarks>
    /// Uses <see cref="Random.Shared"/> because <see cref="Random"/> instance methods are
    /// not thread-safe: concurrent calls on one instance can corrupt its state and make
    /// it return degenerate values. Also guards <c>NextDouble()</c> returning exactly 0,
    /// for which <c>-Math.Log(0)</c> is infinity and the cast to int is undefined
    /// (it yields int.MinValue, and <c>new List&lt;int&gt;[int.MinValue + 1]</c> throws).
    /// </remarks>
    private int GetRandomLayer()
    {
        double r = Random.Shared.NextDouble();
        if (r <= 0.0) r = double.Epsilon;

        int level = (int)Math.Floor(-Math.Log(r) * _mL);
        if (level < 0) level = 0;
        return level > MaxLevelCap ? MaxLevelCap : level;
    }

    // ------------------------------------------------------------------ persistence

    /// <summary>
    /// Writes the graph topology to <paramref name="path"/>.
    /// </summary>
    /// <remarks>
    /// Without this the graph lived only in memory: <c>vectors.bin</c> and
    /// <c>metadata.dat</c> survived a restart but the index did not, so on reload every
    /// stored vector was unreachable and searching returned nothing. Call with
    /// inserts quiesced - the node set is snapshotted, but concurrent linking can still
    /// change adjacency while it is being written.
    /// </remarks>
    public void SaveGraph(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        KeyValuePair<int, VectorNode>[] snapshot = _nodes.ToArray();
        Entry? entry = _entry;

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream);

        writer.Write(GraphMagic);
        writer.Write(GraphVersion);
        writer.Write(_m);
        writer.Write(_efConstruction);
        writer.Write(_vectorStore.Dimensions);
        writer.Write(entry?.Id ?? -1);
        writer.Write(entry?.Layer ?? -1);
        writer.Write(snapshot.Length);

        foreach (KeyValuePair<int, VectorNode> pair in snapshot)
        {
            VectorNode node = pair.Value;
            writer.Write(node.Id);
            writer.Write(node.MaxLayer);

            for (int lc = 0; lc <= node.MaxLayer; lc++)
            {
                List<int> list = node.Connections[lc];
                lock (list)
                {
                    writer.Write(list.Count);
                    for (int i = 0; i < list.Count; i++)
                    {
                        writer.Write(list[i]);
                    }
                }
            }
        }
    }

    /// <summary>Rebuilds an index from a file written by <see cref="SaveGraph"/>.</summary>
    public static HnswIndex LoadGraph(string path, PersistentVectorStore vectorStore)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(vectorStore);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var reader = new BinaryReader(stream);

        if (reader.ReadInt32() != GraphMagic)
        {
            throw new InvalidDataException($"'{path}' is not an HNSW graph file.");
        }

        int version = reader.ReadInt32();
        if (version != GraphVersion)
        {
            throw new InvalidDataException($"Graph file version {version} is not supported (expected {GraphVersion}).");
        }

        int m = reader.ReadInt32();
        int efConstruction = reader.ReadInt32();
        int dimensions = reader.ReadInt32();
        if (dimensions != vectorStore.Dimensions)
        {
            throw new InvalidDataException(
                $"Graph was built for {dimensions} dimensions but the store holds {vectorStore.Dimensions}.");
        }

        int entryId = reader.ReadInt32();
        int entryLayer = reader.ReadInt32();
        int nodeCount = reader.ReadInt32();
        if (nodeCount < 0) throw new InvalidDataException("Graph file declares a negative node count.");

        var index = new HnswIndex(vectorStore, m, efConstruction);

        for (int n = 0; n < nodeCount; n++)
        {
            int id = reader.ReadInt32();
            int maxLayer = reader.ReadInt32();
            if (id < 0 || maxLayer < 0 || maxLayer > MaxLevelCap)
            {
                throw new InvalidDataException($"Graph file holds an invalid node (id {id}, layer {maxLayer}).");
            }

            var node = new VectorNode(id, maxLayer);
            for (int lc = 0; lc <= maxLayer; lc++)
            {
                int count = reader.ReadInt32();
                if (count < 0) throw new InvalidDataException($"Node {id} declares a negative degree at layer {lc}.");

                List<int> list = node.Connections[lc];
                list.Capacity = count;
                for (int i = 0; i < count; i++)
                {
                    list.Add(reader.ReadInt32());
                }
            }

            index._nodes[id] = node;
        }

        index._entry = entryId >= 0 ? new Entry(entryId, entryLayer) : null;
        return index;
    }
}
