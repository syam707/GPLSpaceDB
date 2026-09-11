namespace VDAlgorithmicEngine;

using System.Collections.Generic;

/// <summary>
/// A single vector's position in the layered graph.
/// </summary>
/// <remarks>
/// <see cref="MaxLayer"/> is exposed because the index must be able to re-index an
/// existing id <em>without</em> reallocating <see cref="Connections"/>. Replacing the
/// node object on re-insert is what allowed <c>Connections</c> to shrink while other
/// nodes still held inbound edges at higher layers, producing an
/// <see cref="System.IndexOutOfRangeException"/> during traversal.
/// </remarks>
public sealed class VectorNode
{
    public VectorNode(int id, int maxLayer)
    {
        Id = id;
        MaxLayer = maxLayer;
        Connections = InitializeConnections(maxLayer);
    }

    public int Id { get; }

    /// <summary>Highest layer this node participates in. <see cref="Connections"/> has <c>MaxLayer + 1</c> entries.</summary>
    public int MaxLayer { get; }

    /// <summary>Connections per layer. Index 0 is the base layer, index <see cref="MaxLayer"/> the top.</summary>
    public List<int>[] Connections { get; }

    private static List<int>[] InitializeConnections(int maxLayer)
    {
        var connections = new List<int>[maxLayer + 1];
        for (int i = 0; i <= maxLayer; i++)
        {
            connections[i] = new List<int>();
        }
        return connections;
    }
}
