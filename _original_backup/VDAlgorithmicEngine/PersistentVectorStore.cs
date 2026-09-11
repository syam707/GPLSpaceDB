namespace VDAlgorithmicEngine;

using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;

public sealed unsafe class PersistentVectorStore : IDisposable
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _basePointer = null;

    public int Dimensions { get => field; init => field = value; }
    public string FilePath { get => field; init => field = value; }
    
    private readonly long _capacity;
    private readonly int _vectorByteSize;

    public PersistentVectorStore(string filePath, int dimensions, int maxVectors = 10_000_000)
    {
        FilePath = filePath;
        Dimensions = dimensions;
        _vectorByteSize = dimensions * sizeof(float);
        _capacity = maxVectors;

        long fileSizeBytes = maxVectors * (long)_vectorByteSize;
        
        _mmf = MemoryMappedFile.CreateFromFile(
            filePath, 
            FileMode.OpenOrCreate, 
            null, 
            fileSizeBytes, 
            MemoryMappedFileAccess.ReadWrite);
            
        _accessor = _mmf.CreateViewAccessor(0, fileSizeBytes);
        _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _basePointer);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> GetVectorSpan(int vectorId)
    {
        if (_basePointer == null) throw new ObjectDisposedException(nameof(PersistentVectorStore));
        byte* ptr = _basePointer + (vectorId * (long)_vectorByteSize);
        return new ReadOnlySpan<float>(ptr, Dimensions);
    }

    public void WriteVector(int vectorId, ReadOnlySpan<float> vectorData)
    {
        if (vectorData.Length != Dimensions) throw new ArgumentException("Vector dimensions mismatch");
        if (_basePointer == null) throw new ObjectDisposedException(nameof(PersistentVectorStore));

        byte* ptr = _basePointer + (vectorId * (long)_vectorByteSize);
        var destSpan = new Span<float>(ptr, Dimensions);
        vectorData.CopyTo(destSpan);
    }

    public void Dispose()
    {
        if (_basePointer != null && _accessor != null)
        {
            _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            _basePointer = null;
        }

        _accessor?.Dispose();
        _mmf?.Dispose();
        
        _accessor = null;
        _mmf = null;
    }
}
