namespace VDAlgorithmicEngine;

using System;

public interface IVectorIndex
{
    void Insert(int id, ReadOnlySpan<float> vector);
    int[] Search(ReadOnlySpan<float> queryVector, int kNeighbors);
}
