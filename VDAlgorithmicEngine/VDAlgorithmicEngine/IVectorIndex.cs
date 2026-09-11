namespace VDAlgorithmicEngine;

using System;

public interface IVectorIndex
{
    /// <summary>Number of vectors currently in the index.</summary>
    int Count { get; }

    /// <summary>Dimensionality every vector in this index must have.</summary>
    int Dimensions { get; }

    /// <summary>
    /// Inserts a vector, or re-indexes it in place if <paramref name="id"/> is already
    /// present. The implementation persists <paramref name="vector"/> itself; callers
    /// must not write it to the backing store separately.
    /// </summary>
    void Insert(int id, ReadOnlySpan<float> vector);

    /// <summary>Returns the ids of the <paramref name="kNeighbors"/> nearest vectors, nearest first.</summary>
    int[] Search(ReadOnlySpan<float> queryVector, int kNeighbors);

    /// <summary>
    /// As <see cref="Search(ReadOnlySpan{float}, int)"/>, but with an explicit search
    /// breadth. Larger <paramref name="efSearch"/> trades latency for recall.
    /// </summary>
    int[] Search(ReadOnlySpan<float> queryVector, int kNeighbors, int efSearch);
}
