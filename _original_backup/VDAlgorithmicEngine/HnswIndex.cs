namespace VDAlgorithmicEngine;

using System;
using System.Buffers;
using System.Collections.Concurrent;

public class HnswIndex : IVectorIndex
{
    private readonly ConcurrentDictionary<int, VectorNode> _nodes = new();
    private readonly PersistentVectorStore _vectorStore;
    private int _entryPointId = -1;
    private int _maxLayer = -1;
    
    // HNSW parameters
    private readonly int _m;
    private readonly int _mMax0;
    private readonly int _efConstruction;
    private readonly double _mL;
    private readonly Random _random = new();
    private readonly object _globalLock = new(); 

    public HnswIndex(PersistentVectorStore vectorStore, int m = 16, int efConstruction = 100)
    {
        _vectorStore = vectorStore;
        _m = m;
        _mMax0 = m * 2;
        _efConstruction = efConstruction;
        _mL = 1.0 / Math.Log(m);
    }

    public void Insert(int id, ReadOnlySpan<float> vector)
    {
        int level = GetRandomLayer();
        var newNode = new VectorNode(id, level);
        
        lock (_globalLock)
        {
            if (_nodes.IsEmpty)
            {
                _entryPointId = id;
                _maxLayer = level;
                _nodes[id] = newNode;
                return;
            }
        }
        
        _nodes[id] = newNode;
        
        int currObj = _entryPointId;
        int maxL = _maxLayer;
        float currDist = VectorMath.CosineDistance(vector, _vectorStore.GetVectorSpan(currObj));

        Span<int> snapshot = stackalloc int[_mMax0 + 1];
        for (int lc = maxL; lc > level; lc--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                var neighbors = _nodes[currObj].Connections[lc];
                int snapshotCount = 0;
                lock (neighbors) 
                { 
                    var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(neighbors);
                    span.CopyTo(snapshot);
                    snapshotCount = span.Length;
                }

                for (int i = 0; i < snapshotCount; i++)
                {
                    int neighborId = snapshot[i];
                    float dist = VectorMath.CosineDistance(vector, _vectorStore.GetVectorSpan(neighborId));
                    if (dist < currDist)
                    {
                        currDist = dist;
                        currObj = neighborId;
                        changed = true;
                    }
                }
            }
        }

        Span<int> searchResultsOut = stackalloc int[_efConstruction];
        for (int lc = Math.Min(maxL, level); lc >= 0; lc--)
        {
            int mMax = lc == 0 ? _mMax0 : _m;
            int foundCount = SearchLayer(vector, currObj, _efConstruction, lc, searchResultsOut);
            
            int neighborsToConnect = Math.Min(foundCount, _m);
            for(int i = 0; i < neighborsToConnect; i++)
            {
                Connect(id, searchResultsOut[i], lc, mMax);
            }
            
            currObj = searchResultsOut[0]; 
        }

        lock (_globalLock)
        {
            if (level > _maxLayer)
            {
                _maxLayer = level;
                _entryPointId = id;
            }
        }
    }

    public int[] Search(ReadOnlySpan<float> queryVector, int kNeighbors)
    {
        if (_nodes.IsEmpty) return Array.Empty<int>();

        int currObj = _entryPointId;
        int maxL = _maxLayer;
        float currDist = VectorMath.CosineDistance(queryVector, _vectorStore.GetVectorSpan(currObj));

        Span<int> snapshot = stackalloc int[_mMax0 + 1];
        for (int lc = maxL; lc > 0; lc--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                var neighbors = _nodes[currObj].Connections[lc];
                int snapshotCount = 0;
                lock (neighbors) 
                { 
                    var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(neighbors);
                    span.CopyTo(snapshot);
                    snapshotCount = span.Length;
                }

                for (int i = 0; i < snapshotCount; i++)
                {
                    int neighborId = snapshot[i];
                    float dist = VectorMath.CosineDistance(queryVector, _vectorStore.GetVectorSpan(neighborId));
                    if (dist < currDist)
                    {
                        currDist = dist;
                        currObj = neighborId;
                        changed = true;
                    }
                }
            }
        }

        int efSearch = Math.Max(kNeighbors, 50);
        Span<int> topK = stackalloc int[efSearch];
        int count = SearchLayer(queryVector, currObj, efSearch, 0, topK);
        
        int resultCount = Math.Min(kNeighbors, count);
        var result = new int[resultCount];
        for (int i = 0; i < resultCount; i++)
        {
            result[i] = topK[i];
        }
        return result;
    }

    private void Connect(int id1, int id2, int lc, int mMax)
    {
        var node1 = _nodes[id1];
        var node2 = _nodes[id2];

        lock (node1.Connections[lc])
        {
            node1.Connections[lc].Add(id2);
            if (node1.Connections[lc].Count > mMax)
            {
                ShrinkConnections(node1, lc, mMax);
            }
        }

        lock (node2.Connections[lc])
        {
            node2.Connections[lc].Add(id1);
            if (node2.Connections[lc].Count > mMax)
            {
                ShrinkConnections(node2, lc, mMax);
            }
        }
    }

    private void ShrinkConnections(VectorNode node, int lc, int mMax)
    {
        var connections = node.Connections[lc];
        var candidates = new PriorityQueueElement[connections.Count];
        for (int i = 0; i < connections.Count; i++)
        {
            int nId = connections[i];
            float dist = VectorMath.CosineDistance(_vectorStore.GetVectorSpan(node.Id), _vectorStore.GetVectorSpan(nId));
            candidates[i] = new PriorityQueueElement(nId, dist);
        }
        Array.Sort(candidates);
        
        connections.Clear();
        for (int i = 0; i < mMax; i++)
        {
            connections.Add(candidates[i].Id);
        }
    }

    private int SearchLayer(ReadOnlySpan<float> queryVector, int entryPoint, int ef, int lc, Span<int> resultsOut)
    {
        int visitedSize = 8192;
        var visitedBuffer = ArrayPool<int>.Shared.Rent(visitedSize);
        var candidatesBuffer = ArrayPool<PriorityQueueElement>.Shared.Rent(visitedSize);
        var resultsBuffer = ArrayPool<PriorityQueueElement>.Shared.Rent(ef);
        
        try
        {
            var candidates = new MinHeap(candidatesBuffer.AsSpan(0, visitedSize));
            var results = new MaxHeap(resultsBuffer.AsSpan(0, ef));
            var visited = new FastHashSet(visitedBuffer.AsSpan(0, visitedSize));

            float initialDist = VectorMath.CosineDistance(queryVector, _vectorStore.GetVectorSpan(entryPoint));
            candidates.Enqueue(entryPoint, initialDist);
            results.Enqueue(entryPoint, initialDist);
            visited.Add(entryPoint);

            Span<int> snapshot = stackalloc int[_mMax0 + 1];
            while (candidates.Count > 0)
            {
                var current = candidates.DequeueMin();
                
                if (current.Distance > results.MaxDistance && results.IsFull)
                {
                    break;
                }

                var neighbors = _nodes[current.Id].Connections[lc];
                int snapshotCount = 0;
                lock (neighbors) 
                { 
                    var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(neighbors);
                    span.CopyTo(snapshot);
                    snapshotCount = span.Length;
                }

                for (int i = 0; i < snapshotCount; i++)
                {
                    int neighborId = snapshot[i];
                    if (visited.Add(neighborId))
                    {
                        float dist = VectorMath.CosineDistance(queryVector, _vectorStore.GetVectorSpan(neighborId));
                        
                        if (results.Count < ef || dist < results.MaxDistance)
                        {
                            candidates.Enqueue(neighborId, dist);
                            results.Enqueue(neighborId, dist);
                        }
                    }
                }
            }

            int count = results.Count;
            results.CopyToSorted(resultsOut);
            return count;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(visitedBuffer);
            ArrayPool<PriorityQueueElement>.Shared.Return(candidatesBuffer);
            ArrayPool<PriorityQueueElement>.Shared.Return(resultsBuffer);
        }
    }

    private int GetRandomLayer()
    {
        double r = _random.NextDouble();
        return (int)Math.Floor(-Math.Log(r) * _mL);
    }
}
