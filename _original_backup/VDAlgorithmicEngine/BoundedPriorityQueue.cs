namespace VDAlgorithmicEngine;

using System;
using System.Runtime.CompilerServices;

public readonly struct PriorityQueueElement(int id, float distance) : IComparable<PriorityQueueElement>
{
    public int Id { get; } = id;
    public float Distance { get; } = distance;

    public int CompareTo(PriorityQueueElement other)
    {
        return Distance.CompareTo(other.Distance);
    }
}

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

    public float MaxDistance => _count > 0 ? _elements[0].Distance : float.MaxValue;

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

    public void CopyToSorted(Span<int> destination)
    {
        var activeSpan = _elements.Slice(0, _count);
        activeSpan.Sort((a, b) => a.Distance.CompareTo(b.Distance));
        for (int i = 0; i < _count && i < destination.Length; i++)
        {
            destination[i] = activeSpan[i].Id;
        }
    }
}

public ref struct MinHeap
{
    private Span<PriorityQueueElement> _elements;
    private int _count;

    public MinHeap(Span<PriorityQueueElement> buffer)
    {
        _elements = buffer;
        _count = 0;
    }

    public int Count => _count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Enqueue(int id, float distance)
    {
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
