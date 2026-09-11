namespace VDAlgorithmicEngine;

using System;
using System.Runtime.CompilerServices;

public readonly struct PriorityQueueElement(int id, float distance) : IComparable<PriorityQueueElement>
{
    public int Id { get; } = id;
    public float Distance { get; } = distance;

    /// <summary>
    /// Orders by ascending distance, with NaN sorted <em>last</em>.
    /// </summary>
    /// <remarks>
    /// <see cref="float.CompareTo(float)"/> ranks NaN below every real value, so a
    /// single NaN distance would sort to index 0 and be returned as the closest match.
    /// Vectors are validated on insert so NaN should never occur; this keeps a stray
    /// one from being reported as the best result if it ever does.
    /// </remarks>
    public int CompareTo(PriorityQueueElement other)
    {
        bool thisNaN = float.IsNaN(Distance);
        bool otherNaN = float.IsNaN(other.Distance);
        if (thisNaN || otherNaN)
        {
            if (thisNaN == otherNaN) return 0;
            return thisNaN ? 1 : -1;
        }
        return Distance.CompareTo(other.Distance);
    }
}

/// <summary>Bounded max-heap holding the <c>ef</c> smallest distances seen; the root is the worst kept.</summary>
public ref struct MaxHeap
{
    private Span<PriorityQueueElement> _elements;
    private int _count;

    public MaxHeap(Span<PriorityQueueElement> buffer)
    {
        _elements = buffer;
        _count = 0;
    }

    public int Count => _count;
    public bool IsFull => _count == _elements.Length;
    public float MaxDistance => _count > 0 ? _elements[0].Distance : float.MaxValue;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Enqueue(int id, float distance)
    {
        if (_count < _elements.Length)
        {
            _elements[_count] = new PriorityQueueElement(id, distance);
            SiftUp(_count);
            _count++;
        }
        else if (distance < _elements[0].Distance)
        {
            _elements[0] = new PriorityQueueElement(id, distance);
            SiftDown(0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public PriorityQueueElement DequeueMax()
    {
        if (_count == 0) throw new InvalidOperationException("Queue is empty.");
        var max = _elements[0];
        _count--;
        if (_count > 0)
        {
            _elements[0] = _elements[_count];
            SiftDown(0);
        }
        return max;
    }

    private void SiftUp(int index)
    {
        var element = _elements[index];
        while (index > 0)
        {
            int parentIndex = (index - 1) / 2;
            var parent = _elements[parentIndex];
            if (element.Distance <= parent.Distance) break;

            _elements[index] = parent;
            index = parentIndex;
        }
        _elements[index] = element;
    }

    private void SiftDown(int index)
    {
        var element = _elements[index];
        int half = _count / 2;
        while (index < half)
        {
            int leftChild = 2 * index + 1;
            int rightChild = leftChild + 1;
            int maxChild = leftChild;

            if (rightChild < _count && _elements[rightChild].Distance > _elements[leftChild].Distance)
            {
                maxChild = rightChild;
            }

            if (element.Distance >= _elements[maxChild].Distance) break;

            _elements[index] = _elements[maxChild];
            index = maxChild;
        }
        _elements[index] = element;
    }

    /// <summary>
    /// Drains the heap into <paramref name="destination"/> nearest-first and returns how
    /// many were written. Consumes the heap.
    /// </summary>
    /// <remarks>
    /// The previous implementation called <c>Span.Sort(Comparison&lt;T&gt;)</c>, which
    /// dispatches through a delegate for every comparison - unnecessary overhead on the
    /// hot path, and at odds with the "no delegates, no allocation" design of these
    /// heaps. Draining the heap from the top and filling backwards is allocation-free
    /// and delegate-free, and it discards the worst entries first when
    /// <paramref name="destination"/> is smaller than the heap.
    /// </remarks>
    public int CopyToSorted(Span<int> destination)
    {
        while (_count > destination.Length) DequeueMax();

        int written = _count;
        for (int i = written - 1; i >= 0; i--)
        {
            destination[i] = DequeueMax().Id;
        }
        return written;
    }

    /// <summary>
    /// As <see cref="CopyToSorted(Span{int})"/>, but also emits the distances so callers
    /// do not have to recompute them to report scores.
    /// </summary>
    public int CopyToSorted(Span<int> ids, Span<float> distances)
    {
        int limit = Math.Min(ids.Length, distances.Length);
        while (_count > limit) DequeueMax();

        int written = _count;
        for (int i = written - 1; i >= 0; i--)
        {
            var element = DequeueMax();
            ids[i] = element.Id;
            distances[i] = element.Distance;
        }
        return written;
    }
}

/// <summary>Min-heap of traversal candidates.</summary>
public ref struct MinHeap
{
    private Span<PriorityQueueElement> _elements;
    private int _count;

    public MinHeap(Span<PriorityQueueElement> buffer)
    {
        _elements = buffer;
        _count = 0;
    }

    /// <summary>
    /// Wraps a buffer whose first <paramref name="count"/> elements already satisfy the
    /// heap property. Used to move an existing heap into a larger buffer without
    /// re-heapifying, since a straight copy preserves the ordering.
    /// </summary>
    public MinHeap(Span<PriorityQueueElement> buffer, int count)
    {
        if ((uint)count > (uint)buffer.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Count exceeds buffer length.");
        }
        _elements = buffer;
        _count = count;
    }

    public int Count => _count;
    public bool IsFull => _count == _elements.Length;

    /// <summary>Copies the live elements out in heap order and returns how many.</summary>
    public int CopyRawTo(Span<PriorityQueueElement> destination)
    {
        _elements.Slice(0, _count).CopyTo(destination);
        return _count;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Enqueue(int id, float distance)
    {
        // Callers grow the backing buffer before it fills, so this is a guard rather
        // than a reachable path. It used to be reachable only because the visited-set
        // ceiling happened to cap enqueues below this capacity.
        if (_count >= _elements.Length) throw new InvalidOperationException("Queue is full.");
        _elements[_count] = new PriorityQueueElement(id, distance);
        SiftUp(_count);
        _count++;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public PriorityQueueElement DequeueMin()
    {
        if (_count == 0) throw new InvalidOperationException("Queue is empty.");
        var min = _elements[0];
        _count--;
        if (_count > 0)
        {
            _elements[0] = _elements[_count];
            SiftDown(0);
        }
        return min;
    }

    private void SiftUp(int index)
    {
        var element = _elements[index];
        while (index > 0)
        {
            int parentIndex = (index - 1) / 2;
            var parent = _elements[parentIndex];
            if (element.Distance >= parent.Distance) break;

            _elements[index] = parent;
            index = parentIndex;
        }
        _elements[index] = element;
    }

    private void SiftDown(int index)
    {
        var element = _elements[index];
        int half = _count / 2;
        while (index < half)
        {
            int leftChild = 2 * index + 1;
            int rightChild = leftChild + 1;
            int minChild = leftChild;

            if (rightChild < _count && _elements[rightChild].Distance < _elements[leftChild].Distance)
            {
                minChild = rightChild;
            }

            if (element.Distance <= _elements[minChild].Distance) break;

            _elements[index] = _elements[minChild];
            index = minChild;
        }
        _elements[index] = element;
    }
}
