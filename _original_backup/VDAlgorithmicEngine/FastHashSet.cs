namespace VDAlgorithmicEngine;

using System;

public ref struct FastHashSet
{
    private Span<int> _buffer;
    private int _count;

    public FastHashSet(Span<int> buffer)
    {
        _buffer = buffer;
        _buffer.Clear();
        _count = 0;
    }

    public bool Add(int value)
    {
        if (_count >= _buffer.Length / 2) return false;

        int key = value + 1; 
        uint hash = (uint)value * 2654435761u;
        int index = (int)(hash % (uint)_buffer.Length);

        while (_buffer[index] != 0)
        {
            if (_buffer[index] == key) return false;
            index = (index + 1) % _buffer.Length;
        }

        _buffer[index] = key;
        _count++;
        return true;
    }
}
