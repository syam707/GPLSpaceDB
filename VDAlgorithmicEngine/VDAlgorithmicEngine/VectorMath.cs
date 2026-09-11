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
        // Cosine distance is typically 1 - cosine similarity.
        return 1.0f - TensorPrimitives.CosineSimilarity(vector1, vector2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float L2Distance(ReadOnlySpan<float> vector1, ReadOnlySpan<float> vector2)
    {
        return TensorPrimitives.Distance(vector1, vector2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> vector1, ReadOnlySpan<float> vector2)
    {
        return TensorPrimitives.Dot(vector1, vector2);
    }

    /// <summary>Euclidean magnitude of <paramref name="vector"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Magnitude(ReadOnlySpan<float> vector)
    {
        return MathF.Sqrt(TensorPrimitives.Dot(vector, vector));
    }

    /// <summary>
    /// Rejects vectors for which cosine distance is not defined, before they can
    /// poison the index.
    /// </summary>
    /// <remarks>
    /// Cosine similarity divides by the product of both magnitudes, so a zero vector
    /// yields 0/0 = NaN and a non-finite component yields NaN. NaN then breaks every
    /// comparison in the traversal: <c>dist &lt; currDist</c> is always false, so the
    /// greedy descent can never move, and a NaN distance recorded for the entry point
    /// is returned as the top-ranked result. The memory-mapped vector file is
    /// zero-filled, so every id that was never written is exactly this case - which is
    /// what happened when <c>HnswIndex</c> was used without writing vectors first.
    /// </remarks>
    public static void ValidateForCosine(ReadOnlySpan<float> vector, int expectedDimensions, string paramName = "vector")
    {
        if (vector.Length != expectedDimensions)
        {
            throw new ArgumentException(
                $"Vector has {vector.Length} dimensions but this index expects {expectedDimensions}.", paramName);
        }

        float magnitude = Magnitude(vector);
        if (!float.IsFinite(magnitude) || magnitude == 0.0f)
        {
            throw new ArgumentException(
                "Vector must have a finite, non-zero magnitude; cosine distance is undefined otherwise " +
                "(a zero or NaN-bearing vector produces NaN distances that corrupt ranking).", paramName);
        }
    }
}
