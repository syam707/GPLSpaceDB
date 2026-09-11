namespace VDAlgorithmicEngine;

using System;

/// <summary>
/// Generation-stamped visited set used by <see cref="HnswIndex"/> graph traversal.
/// </summary>
/// <remarks>
/// This replaces the previous <c>FastHashSet</c>, which had two defects:
/// <list type="bullet">
/// <item><description>
/// <c>Add</c> returned <c>false</c> both for "already present" and for "buffer is
/// full". Because the caller treats <c>false</c> as "already visited", the search
/// silently stopped discovering new nodes once 4096 entries were reached - with no
/// error and no way to detect it. A build at m=32/efConstruction=400 hits that
/// ceiling roughly four million times.
/// </description></item>
/// <item><description>
/// Ids were stored as <c>value + 1</c> with 0 as the empty sentinel, so id -1
/// collided with "empty" and was never deduplicated.
/// </description></item>
/// </list>
/// A generation stamp removes the ceiling entirely, makes <c>Add</c> a single array
/// compare with no hashing and no modulo, and removes the 32 KB <c>Span.Clear()</c>
/// that the old set paid on every single <c>SearchLayer</c> call.
/// </remarks>
internal sealed class VisitedSet
{
    private int[] _stamps;
    private int _generation;

    public VisitedSet(int capacity)
    {
        _stamps = new int[Math.Clamp(capacity, 1024, Array.MaxLength)];
        _generation = 0;
    }

    /// <summary>Begins a new traversal. Must be called before the first <see cref="Add"/>.</summary>
    public void NewGeneration()
    {
        if (_generation == int.MaxValue)
        {
            Array.Clear(_stamps);
            _generation = 1;
        }
        else
        {
            _generation++;
        }
    }

    public void EnsureCapacity(int capacity)
    {
        if (capacity <= _stamps.Length) return;

        long grown = _stamps.Length;
        while (grown < capacity) grown *= 2;
        if (grown > Array.MaxLength) grown = Array.MaxLength;
        if (capacity > grown)
        {
            throw new ArgumentOutOfRangeException(
                nameof(capacity), capacity, "Visited set cannot address that many vector ids.");
        }

        Array.Resize(ref _stamps, (int)grown);
    }

    /// <summary>Returns <c>true</c> if <paramref name="id"/> had not been seen in this generation.</summary>
    public bool Add(int id)
    {
        if (id < 0) throw new ArgumentOutOfRangeException(nameof(id), id, "Vector ids must be non-negative.");
        if (id >= _stamps.Length) EnsureCapacity(id + 1);

        if (_stamps[id] == _generation) return false;
        _stamps[id] = _generation;
        return true;
    }
}

/// <summary>
/// Hands out one <see cref="VisitedSet"/> per thread, so the stamp array is allocated
/// once per thread rather than once per query.
/// </summary>
/// <remarks>
/// Deliberately a thread-static slot rather than a pool: <c>ConcurrentBag.Add</c>
/// allocates an internal node on every return, which would put an allocation back on a
/// traversal path that is otherwise allocation-free. No <c>Return</c> is needed, and
/// nothing has to be disposed.
///
/// Safe to share between indexes: a traversal never nests within a thread, and
/// <see cref="VisitedSet.EnsureCapacity"/> plus <see cref="VisitedSet.NewGeneration"/>
/// make each rental independent of the last.
/// </remarks>
internal static class VisitedSetPool
{
    [ThreadStatic]
    private static VisitedSet? _current;

    public static VisitedSet Rent(int capacity)
    {
        VisitedSet set = _current ??= new VisitedSet(capacity);
        set.EnsureCapacity(capacity);
        set.NewGeneration();
        return set;
    }
}
