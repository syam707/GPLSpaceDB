namespace VDAlgorithmicEngine;

using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

public static class VectorMath
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CosineSimilarity(ReadOnlySpan<float> vector1, ReadOnlySpan<float> vector2)
    {
        return TensorPrimitives.CosineSimilarity(vector1, vector2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CosineDistance(ReadOnlySpan<float> vector1, ReadOnlySpan<float> vector2)
    {
        // Cosine distance is typically 1 - cosine similarity
        return 1.0f - TensorPrimitives.CosineSimilarity(vector1, vector2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float L2Distance(ReadOnlySpan<float> vector1, ReadOnlySpan<float> vector2)
    {
        return TensorPrimitives.Distance(vector1, vector2);
    }
}
