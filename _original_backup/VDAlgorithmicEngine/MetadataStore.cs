namespace VDAlgorithmicEngine;

using System;
using System.Collections.Concurrent;
using System.IO;

public class MetadataStore : IDisposable
{
    public string FilePath { get => field; init => field = value; }
    
    private readonly ConcurrentDictionary<int, string> _metadataCache = new();
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _writer;
    private readonly object _writeLock = new();

    public MetadataStore(string filePath)
    {
        FilePath = filePath;
        
        if (File.Exists(filePath))
        {
            using var readerStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var reader = new BinaryReader(readerStream);
            while (readerStream.Position < readerStream.Length)
            {
                int id = reader.ReadInt32();
                string meta = reader.ReadString();
                _metadataCache[id] = meta;
            }
        }

        _fileStream = new FileStream(filePath, FileMode.Append, FileAccess.Write, FileShare.Read);
        _writer = new BinaryWriter(_fileStream);
    }

    public void SetMetadata(int vectorId, string metadata)
    {
        metadata ??= string.Empty; // C# null-conditional assignment
        
        _metadataCache[vectorId] = metadata;
        
        lock (_writeLock)
        {
            _writer.Write(vectorId);
            _writer.Write(metadata);
            _writer.Flush();
        }
    }

    public string GetMetadata(int vectorId)
    {
        return _metadataCache.TryGetValue(vectorId, out string? meta) ? meta : string.Empty;
    }

    public void Dispose()
    {
        _writer?.Dispose();
        _fileStream?.Dispose();
    }
}
