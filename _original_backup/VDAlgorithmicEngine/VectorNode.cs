namespace VDAlgorithmicEngine;

using System.Collections.Generic;

public class VectorNode(int id, int maxLayer)
{
    public int Id { get; } = id;
    
    // Connections per layer. Index 0 is the base layer, index maxLayer is the top layer.
    public List<int>[] Connections { get; } = InitializeConnections(maxLayer);

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
