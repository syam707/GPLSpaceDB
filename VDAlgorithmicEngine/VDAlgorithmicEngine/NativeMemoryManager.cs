namespace VDAlgorithmicEngine;

using System;
using System.Buffers;
using System.Runtime.InteropServices;

public sealed unsafe class NativeMemoryManager<T> : MemoryManager<T> where T : unmanaged
{
    private readonly T* _pointer;
    private readonly int _length;
    private bool _disposed;

    public NativeMemoryManager(int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        _length = length;
        _pointer = (T*)NativeMemory.Alloc((nuint)length, (nuint)sizeof(T));
    }

    public override Span<T> GetSpan()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NativeMemoryManager<T>));
        return new Span<T>(_pointer, _length);
    }

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NativeMemoryManager<T>));
        if (elementIndex < 0 || elementIndex >= _length)
            throw new ArgumentOutOfRangeException(nameof(elementIndex));
        
        return new MemoryHandle(_pointer + elementIndex);
    }

    public override void Unpin() { }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (_pointer != null)
            {
                NativeMemory.Free(_pointer);
            }
            _disposed = true;
        }
    }
}
