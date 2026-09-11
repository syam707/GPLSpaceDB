namespace VDAlgorithmicEngine;

using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;

/// <summary>
/// Fixed-stride vector storage backed by a memory-mapped file. A vector's id is its
/// slot index, so ids must be dense and in <c>[0, MaxVectors)</c>.
/// </summary>
public sealed unsafe class PersistentVectorStore : IDisposable
{
    private const uint FsctlSetSparse = 0x000900C4;

    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _basePointer = null;

    private readonly int _maxVectors;
    private readonly int _vectorByteSize;
    private int _highWaterMark;

    public int Dimensions { get; }
    public string FilePath { get; }

    /// <summary>Exclusive upper bound on valid vector ids.</summary>
    public int MaxVectors => _maxVectors;

    /// <summary>Total bytes the backing file is mapped at.</summary>
    public long MappedBytes { get; }

    /// <summary>One past the highest id ever written, i.e. the used prefix of the file.</summary>
    public int HighWaterMark => Volatile.Read(ref _highWaterMark);

    /// <param name="maxVectors">
    /// Slot count to reserve. The file is created at <c>maxVectors * dimensions * 4</c>
    /// bytes up front, so this is also the hard id ceiling. On Windows the file is
    /// flagged sparse before it is sized, so unwritten slots cost no physical disk;
    /// without that flag the default reserved 4.8 GiB of real disk to hold 1000 vectors.
    /// </param>
    public PersistentVectorStore(string filePath, int dimensions, int maxVectors = 10_000_000)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxVectors);

        FilePath = filePath;
        Dimensions = dimensions;
        _vectorByteSize = dimensions * sizeof(float);
        _maxVectors = maxVectors;

        long requestedBytes = maxVectors * (long)_vectorByteSize;

        var fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
        try
        {
            if (fileStream.Length < requestedBytes)
            {
                TrySetSparse(fileStream);
                fileStream.SetLength(requestedBytes);
            }

            // CreateFromFile rejects a capacity below the existing file length, so never
            // shrink the map for a file that was created with a larger maxVectors.
            MappedBytes = Math.Max(requestedBytes, fileStream.Length);

            _mmf = MemoryMappedFile.CreateFromFile(
                fileStream,
                mapName: null,
                capacity: MappedBytes,
                MemoryMappedFileAccess.ReadWrite,
                HandleInheritability.None,
                leaveOpen: false);
        }
        catch
        {
            fileStream.Dispose();
            throw;
        }

        _accessor = _mmf.CreateViewAccessor(0, MappedBytes, MemoryMappedFileAccess.ReadWrite);
        _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _basePointer);
    }

    /// <summary>
    /// Marks the file sparse so that setting a large length reserves address space
    /// rather than physical disk. Best effort: Windows only, and NTFS/ReFS only.
    /// Unix filesystems already allocate lazily.
    /// </summary>
    private static void TrySetSparse(FileStream fileStream)
    {
        if (!OperatingSystem.IsWindows()) return;

        try
        {
            DeviceIoControl(
                fileStream.SafeFileHandle, FsctlSetSparse,
                IntPtr.Zero, 0, IntPtr.Zero, 0, out _, IntPtr.Zero);
        }
        catch (DllNotFoundException)
        {
            // Not fatal - the file is simply allocated eagerly.
        }
        catch (EntryPointNotFoundException)
        {
        }
    }

    [DllImport("kernel32.dll", SetLastError = true, EntryPoint = "DeviceIoControl")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool DeviceIoControl(
        SafeFileHandle hDevice,
        uint dwIoControlCode,
        IntPtr lpInBuffer,
        uint nInBufferSize,
        IntPtr lpOutBuffer,
        uint nOutBufferSize,
        out uint lpBytesReturned,
        IntPtr lpOverlapped);

    /// <summary>Returns a view over the stored vector. The span aliases the mapping; do not retain it past a <see cref="Dispose"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> GetVectorSpan(int vectorId)
    {
        // Unsigned compare rejects negatives and overruns in one branch. Without this
        // check the pointer arithmetic below walked straight off either end of the
        // mapping: an id of MaxVectors wrote past the last slot, and a negative id
        // wrote *before* the base pointer. Both were silent memory corruption.
        if ((uint)vectorId >= (uint)_maxVectors) ThrowIdOutOfRange(vectorId);
        if (_basePointer == null) ThrowDisposed();

        byte* ptr = _basePointer + (vectorId * (long)_vectorByteSize);
        return new ReadOnlySpan<float>(ptr, Dimensions);
    }

    public void WriteVector(int vectorId, ReadOnlySpan<float> vectorData)
    {
        if ((uint)vectorId >= (uint)_maxVectors) ThrowIdOutOfRange(vectorId);
        if (vectorData.Length != Dimensions)
        {
            throw new ArgumentException(
                $"Vector has {vectorData.Length} dimensions but this store holds {Dimensions}.", nameof(vectorData));
        }
        if (_basePointer == null) ThrowDisposed();

        byte* ptr = _basePointer + (vectorId * (long)_vectorByteSize);
        vectorData.CopyTo(new Span<float>(ptr, Dimensions));

        AdvanceHighWaterMark(vectorId + 1);
    }

    private void AdvanceHighWaterMark(int candidate)
    {
        int observed = Volatile.Read(ref _highWaterMark);
        while (candidate > observed)
        {
            int prior = Interlocked.CompareExchange(ref _highWaterMark, candidate, observed);
            if (prior == observed) return;
            observed = prior;
        }
    }

    /// <summary>Pushes dirty mapped pages to the filesystem. Not called implicitly on every write.</summary>
    public void Flush() => _accessor?.Flush();

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowIdOutOfRange(int vectorId) =>
        throw new ArgumentOutOfRangeException(
            "vectorId", vectorId,
            $"Vector id must be in [0, {_maxVectors}). Ids are slot indices in '{FilePath}'; " +
            "construct the store with a larger maxVectors if you need a wider id range.");

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowDisposed() => throw new ObjectDisposedException(nameof(PersistentVectorStore));

    public void Dispose()
    {
        if (_basePointer != null && _accessor != null)
        {
            try
            {
                _accessor.Flush();
            }
            catch (ObjectDisposedException)
            {
            }

            _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            _basePointer = null;
        }

        _accessor?.Dispose();
        _mmf?.Dispose();

        _accessor = null;
        _mmf = null;
    }
}
