namespace VDAlgorithmicEngine;

using System;
using System.Collections.Concurrent;
using System.IO;

/// <summary>
/// Append-only id-to-string metadata log with an in-memory read cache.
/// </summary>
/// <remarks>
/// The log is never compacted, so updating the same id repeatedly grows the file
/// without bound; the last record for an id wins on reload.
/// </remarks>
public sealed class MetadataStore : IDisposable
{
    private readonly ConcurrentDictionary<int, string> _metadataCache = new();
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _writer;
    private readonly object _writeLock = new();
    private readonly bool _flushOnWrite;

    public string FilePath { get; }

    public int Count => _metadataCache.Count;

    /// <summary>True if a partially written trailing record was discarded when opening.</summary>
    public bool WasTruncatedOnLoad { get; }

    /// <param name="flushOnWrite">
    /// Flush after every record. Safe by default, but it is a syscall per insert and
    /// dominates bulk-load throughput - set false while loading, then call
    /// <see cref="Flush"/> once at the end.
    /// </param>
    public MetadataStore(string filePath, bool flushOnWrite = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        FilePath = filePath;
        _flushOnWrite = flushOnWrite;

        long validLength = 0;
        bool torn = false;

        if (File.Exists(filePath))
        {
            using (var readerStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var reader = new BinaryReader(readerStream))
            {
                try
                {
                    while (readerStream.Position < readerStream.Length)
                    {
                        int id = reader.ReadInt32();
                        string meta = reader.ReadString();
                        _metadataCache[id] = meta;
                        validLength = readerStream.Position;
                    }
                }
                catch (EndOfStreamException)
                {
                    // A crash between Write(int) and Write(string) leaves a partial
                    // record at the tail. Letting that exception escape the constructor -
                    // which is what happened before - made the store permanently
                    // unopenable, taking all the metadata with it. Drop the partial tail.
                    torn = true;
                }
                catch (IOException ex)
                {
                    throw new InvalidDataException(
                        $"'{filePath}' is corrupt at byte offset {validLength}; " +
                        $"{_metadataCache.Count} records were readable before that point.", ex);
                }
            }

            if (torn)
            {
                using var repair = new FileStream(filePath, FileMode.Open, FileAccess.Write, FileShare.None);
                repair.SetLength(validLength);
            }
        }

        WasTruncatedOnLoad = torn;

        _fileStream = new FileStream(filePath, FileMode.Append, FileAccess.Write, FileShare.Read);
        _writer = new BinaryWriter(_fileStream);
    }

    public void SetMetadata(int vectorId, string? metadata)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(vectorId);

        metadata ??= string.Empty;
        _metadataCache[vectorId] = metadata;

        lock (_writeLock)
        {
            _writer.Write(vectorId);
            _writer.Write(metadata);
            if (_flushOnWrite) _writer.Flush();
        }
    }

    public string GetMetadata(int vectorId) =>
        _metadataCache.TryGetValue(vectorId, out string? meta) ? meta : string.Empty;

    public bool TryGetMetadata(int vectorId, out string metadata)
    {
        if (_metadataCache.TryGetValue(vectorId, out string? found))
        {
            metadata = found;
            return true;
        }
        metadata = string.Empty;
        return false;
    }

    public void Flush()
    {
        lock (_writeLock)
        {
            _writer.Flush();
        }
    }

    public void Dispose()
    {
        try
        {
            Flush();
        }
        catch (ObjectDisposedException)
        {
        }

        _writer.Dispose();
        _fileStream.Dispose();
    }
}
