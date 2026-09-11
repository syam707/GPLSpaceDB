# VDAlgorithmicEngine — code review

Reviewed at commit state of 2026-09-11. No .NET 10 SDK was reachable from the review
environment, so nothing here was verified by compiling. Correctness claims about the
algorithm were verified instead by porting `HnswIndex`'s exact construction and
`SearchLayer` semantics — including the bounded heaps and the visited set's saturation
behaviour — and running it against brute-force ground truth. The originals are kept
under `_original_backup/` so the whole change set can be diffed or reverted.

## What the engine does

Three projects sit in the tree. `VDAlgorithmicEngine` is the library and holds
everything real. `VDTest` is a console harness. `VectorDatabase` is an empty stub
containing only `Class1.cs`; it is referenced by the solution and nothing else.

The design is sound and the layering is clean. Vectors live at a fixed stride in a
memory-mapped file, so the garbage collector never scans them — `PersistentVectorStore`
hands out `ReadOnlySpan<float>` views directly over the mapping. On top of that sits an
HNSW graph whose nodes are `VectorNode` objects in a `ConcurrentDictionary`, each
holding a `List<int>` of neighbours per layer. Search is a greedy descent through the
upper layers followed by a best-first expansion at layer zero, using `ref struct` heaps
over `ArrayPool` buffers so the traversal itself does not allocate. Distances go through
`TensorPrimitives`, which is the right call — it dispatches to AVX2 or AVX-512 without
any intrinsics code of your own.

The ideas are all correct. The defects are in the details, and they cluster in three
places: the id model has no validation anywhere, the mutation paths were written as if
inserts only ever happen once per id, and several buffers were sized by magic numbers
that silently truncate rather than failing loudly.

## Confirmed defects

### 1. Re-inserting an existing id corrupts the graph

`Insert` did `_nodes[id] = newNode` unconditionally. Calling it again for an id that
already exists — the natural way to update an embedding — replaced the `VectorNode` with
a fresh one at a newly drawn random level. Every other node in the graph still held
inbound edges to that id at the *old* layers. When the new level was lower, the node's
`Connections` array shrank beneath those edges, and the next traversal that followed one
of them evaluated `_nodes[current.Id].Connections[lc]` with `lc` past the end of the
array.

Over 400 re-inserts at N=6000, 5.8% produced a shorter `Connections` array that still
had stale higher-layer inbound edges — each one a latent `IndexOutOfRangeException` in
`SearchLayer`. The remaining 94% silently orphaned the node's own edges instead. The
probability follows directly from the level distribution: with `mL = 1/ln(16)`, a node
sits above layer 0 with probability 1/16, and a redraw lands at layer 0 with probability
15/16.

Fixed by keeping the existing node and its level on re-insert, and clearing only its
outbound edges before relinking. `Connections.Length` never shrinks, so inbound edges
stay in range. The entry-point special case — re-indexing the entry point itself, which
would otherwise link the node only against itself and leave it with no edges at all — is
handled by falling back to the next highest-layer node.

### 2. The vector store never bounds-checked anything

`_capacity` was stored in the constructor and never read again. Both `GetVectorSpan` and
`WriteVector` computed `_basePointer + vectorId * _vectorByteSize` and constructed a
span over it with no validation. An id of `maxVectors` wrote 512 bytes past the end of a
5 GB mapping; a negative id wrote *before* the base pointer. Because the multiply was
already widened to `long` there was no overflow to catch it, and because the result is a
raw pointer there was no managed bounds check either. Any out-of-range integer reaching
`Insert` was silent memory corruption rather than an exception.

This is the most dangerous finding, because ids are slot indices: the moment anyone maps
a document id or a hash into this API instead of a dense counter, it corrupts memory
rather than complaining. Fixed with a single unsigned comparison on both paths, which
rejects negatives and overruns in one branch, plus an explicit `MaxVectors` property and
an error message that says what the id range actually is.

### 3. The visited set silently capped recall, and made `efConstruction` a no-op

`FastHashSet.Add` returned `false` when `_count >= _buffer.Length / 2`. The caller treats
`false` as "already visited", so once 4096 nodes had been seen, every genuinely new node
was reported as a duplicate and the expansion simply stopped — with no error, no
diagnostic, and no way to observe it from outside.

At the default tuning this is invisible: the peak visited count at N=6000 was 1699,
about 40% of the ceiling. It is reached under the tuning the README itself recommends
for higher accuracy:

| tuning | recall@10 | peak visited | refusals |
| --- | --- | --- | --- |
| `m=16, efConstruction=100` (default) | 100.0% | 1699 | 0 |
| `m=32, efConstruction=200` (README) | 100.0% | 3667 | 0 |
| `m=32, efConstruction=400` | 100.0% | 4096 | 4,033,079 |
| `m=48, efConstruction=500` | 100.0% | 4096 | 29,497,351 |

Recall did not drop on this data, so the practical effect is subtler than a wrong answer:
past roughly `efConstruction` 300 the parameter stops doing anything, the build burns
millions of no-op refusals, and the index you get is not the index the parameters
describe. On harder data the truncation costs recall directly.

Replaced with a generation-stamped visited set. `Add` is one array compare against a
generation counter, with no hashing, no modulo, no ceiling, and — because a new
generation is just an increment — none of the 32 KB `Span.Clear()` the old set paid on
*every* `SearchLayer` call, which at `efConstruction=100` is once per layer per insert.

The old set had a second, narrower bug: it stored ids as `value + 1` with 0 as the empty
sentinel, so id -1 hashed to the sentinel and was never deduplicated. The generation
stamp removes the sentinel concept entirely.

### 4. Zero vectors produce NaN, and NaN was ranked first

Cosine similarity divides by the product of both magnitudes, so a zero vector yields
0/0 = NaN and the distance becomes `1 - NaN = NaN`. The memory-mapped file is
zero-filled, which means *every id that was never written* is exactly this case.

NaN then breaks the traversal in two different ways. In the greedy descent, `dist <
currDist` is false for NaN, so if the entry point's distance is NaN the descent can never
improve and never moves. Worse, `SearchLayer` enqueues the entry point into the results
heap unconditionally, and `Single.CompareTo` ranks NaN *below* every real value — so the
old `CopyToSorted`, which sorted by `CompareTo`, placed that NaN at index 0 and returned
it as the closest match.

Fixed at the source: `Insert` now rejects any vector without a finite, non-zero
magnitude. As defence in depth, `PriorityQueueElement.CompareTo` now sorts NaN last, so a
stray one can never be reported as the best hit.

### 5. The README's example did not compile, and described the wrong data flow

The documented usage was `IVectorIndex index = new HnswIndex();`. There is no
parameterless constructor — it requires a `PersistentVectorStore`. The example also
implied `index.Insert(id, vector)` stores the vector, which it did not: only
`VectorEngine.Insert` wrote to the store. Following the README therefore built an index
over slots that were never written, which is finding 4 — every distance NaN, every
result meaningless.

This was a design problem, not just a doc problem: nothing in the type system made the
"write before you index" ordering visible, and the two ways in disagreed about who owned
the write. `HnswIndex.Insert` now performs the write itself, `VectorEngine.Insert` no
longer duplicates it, and the README was rewritten around what the code actually does.

### 6. `Connect` could deadlock

`Connect` took `lock (node1.Connections[lc])` and then `lock (node2.Connections[lc])` in
argument order. Two concurrent inserts linking the same pair in opposite directions — A
connecting to B while B connects to A — each hold the first lock and wait for the second.
Classic lock-order inversion, and reachable given that the class is built around
`ConcurrentDictionary` and per-list locks, so concurrent insert is clearly intended.
Fixed by always acquiring in ascending id order.

### 7. The entry point could be read torn

`_entryPointId` and `_maxLayer` were written together under `_globalLock` but read as two
independent fields with no lock. A reader could pair a freshly published entry-point id
with the previous max layer, or the reverse, and then index `Connections[lc]` past the
end of the array — a second, concurrency-only route to the same
`IndexOutOfRangeException` as finding 1. Both values are now published as a single
immutable record through one volatile reference, so every reader sees a consistent pair.

### 8. `Random` was shared across threads, and could draw an invalid level

`GetRandomLayer` called `_random.NextDouble()` on one shared `Random` instance.
`Random`'s instance methods are not thread-safe; concurrent calls can corrupt its
internal state and make it return degenerate values, which for this function means
degenerate graph levels. Now uses `Random.Shared`.

Two smaller issues in the same three lines. `NextDouble()` returns `[0, 1)` and can
return exactly 0, for which `-Math.Log(0)` is infinity and the cast to `int` is undefined
behaviour — it yields `int.MinValue`, and `new List<int>[int.MinValue + 1]` throws. And
the level was uncapped. Both are now guarded, with a `MaxLevelCap` of 32, which the
distribution will not reach in practice.

### 9. `stackalloc` was sized from a caller-supplied argument

`Search` did `stackalloc int[Math.Max(kNeighbors, 50)]`, and `Insert` did `stackalloc
int[_efConstruction]`. Both are caller-controlled, and a stack overflow cannot be caught
— it kills the process:

| k | stackalloc | outcome |
| --- | --- | --- |
| 1,000 | 4 KB | fine |
| 100,000 | 400 KB | risky on a 1 MB stack |
| 262,144 | 1 MB | stack overflow |
| 1,000,000 | 4 MB | stack overflow |

Both now rent from `ArrayPool`. The two remaining `stackalloc` sites are sized from
`_mMax0 + 1`, and `m` is validated to at most 256 in the constructor, so they are bounded
at roughly 2 KB by construction.

### 10. The graph was never persisted

`vectors.bin` and `metadata.dat` survived a restart. The graph did not — there was no
save, no load, and no record of which ids existed. Reopening gave an empty `_nodes`, so
`Search` returned nothing and every stored vector was unreachable. "Persistent" described
the file, not the system. Added `SaveGraph`/`LoadGraph` with a versioned binary format,
wired into `VectorEngine` so reopening the same paths reloads automatically, with
`LoadedFromDisk` to report whether it did.

### 11. Pruning had no diversity heuristic

`ShrinkConnections` kept the nearest `mMax` neighbours. Standard HNSW instead applies
Algorithm 4 from Malkov and Yashunin: keep a candidate only if it is closer to the base
node than to any neighbour already kept, which preserves the long-range links that make
the graph navigable.

This is worth isolating because it is the one finding where the obvious test would have
told you nothing. On uniformly random vectors the two are indistinguishable, which is
exactly what `VDTest` generated:

| data | nearest-m pruning | Algorithm 4 | delta |
| --- | --- | --- | --- |
| uniform gaussian | 100.0% | 99.9% | −0.1% |
| 40 clusters, spread 0.30 | 99.8% | 99.8% | +0.0% |
| 40 clusters, spread 0.10 | 95.5% | 99.7% | +4.2% |
| 12 clusters, spread 0.05 | 90.8% | 95.9% | +5.2% |

Real embeddings are clustered, so the 90.8% row is the realistic one. Now implemented,
with a backfill pass so the heuristic cannot leave a node under-connected. Note the
trade: it costs O(m²) distance computations per shrink, so inserts get slower.

### 12. The vector file consumed 4.8 GiB to hold 1000 vectors

`maxVectors` defaulted to 10,000,000, so at 128 dimensions the file was preallocated to
5,120,000,000 bytes. This was not hypothetical — `VDTest/vectors.bin` was on disk at
exactly that size, and `du` confirmed 4.8 GiB of *real* allocation, not a sparse file, to
hold 1000 vectors totalling 512 KB.

The file is now flagged sparse via `FSCTL_SET_SPARSE` before it is sized, so unwritten
slots reserve address space instead of disk, and `maxVectors` is documented as the hard
id ceiling it always was. `VDTest` now passes an explicit 20,000.

### 13. A crash during a metadata write bricked the store permanently

`MetadataStore`'s constructor read the log with `ReadInt32` followed by `ReadString` in a
bare loop. A crash between those two writes leaves a partial record, and the resulting
`EndOfStreamException` escaped the constructor — so the store could never be opened
again, taking all the metadata with it. The partial tail is now discarded, with
`WasTruncatedOnLoad` reporting that it happened; a genuine IO error is still surfaced, as
`InvalidDataException` naming the byte offset.

Separately, `SetMetadata` flushed on every record, which is a syscall per insert and
dominates bulk-load throughput. Still the default, but `flushOnWrite: false` plus one
`Flush()` at the end is now available.

### 14. Smaller items

`MaxHeap.CopyToSorted` called `Span.Sort(Comparison<T>)`, dispatching through a delegate
for every comparison on the hot path — at odds with the no-delegate design of the heaps
either side of it. It now drains the heap and fills backwards, which is delegate-free and
discards the worst entries first when the destination is smaller than the heap.

`ExecuteSearchWithMetadata` recomputed a cosine similarity per result because
`IVectorIndex.Search` returned bare ids and threw the distances away. `SearchWithScores`
now returns the distances the traversal already computed.

`efSearch` was hardcoded to `max(k, 50)` with no way to tune recall per query. It is now
a parameter.

The solution file omitted `VDTest`, so building the solution never built the tests. Added.
There was no `.gitignore`, meaning `bin/`, `obj/`, `.vs/` and a 4.8 GiB binary were all
sitting in the working tree ready to be committed. Added.

## Two things I suspected and was wrong about

Worth recording, because both look like bugs on a reading and are not.

`MinHeap.Enqueue` throws when full and the candidate heap was sized at a hardcoded 8192,
which reads like an overflow waiting to happen at scale. It cannot happen in the original
code: every enqueue is gated behind `visited.Add` returning true, and the visited set
refuses after 4096 entries, so enqueues are capped at 4097 — below the heap's capacity.
The two bugs cancelled. Measured peak heap depth was 292. This mattered for the fix,
though: *removing* the visited ceiling removes the accidental protection, so the candidate
heap now has to grow, and it does.

`Connect` appends without a duplicate check, which looks like it should accumulate
repeated edges. It does not, in normal operation — each ordered pair is connected at most
once per layer per insert. Measured across the whole graph: 0 duplicate slots out of
162,040. A dedup check was added anyway, because re-insert relinking can now revisit a
pair, but it was not fixing an existing bug.

## Recommended next, not done here

The largest remaining item is the node representation. `ConcurrentDictionary<int,
VectorNode>` with a `List<int>[]` per node means that at 10M vectors you have roughly 10M
`VectorNode` objects plus 11M `List<int>` instances plus their backing arrays — well over
30M GC-tracked objects. That sits oddly next to the design's central claim, which is that
the GC does not track your data: it is true of the vectors and false of the graph. A flat
CSR-style adjacency — one `int[]` at a fixed stride of `mMax0` plus an `int[]` of degrees
— would reduce that to a handful of objects and make traversal sequential instead of
pointer-chasing. It is invasive, so it belongs in its own change.

Beyond that: if embeddings arrive pre-normalised, cosine similarity reduces to a dot
product, and skipping both magnitude computations on every distance evaluation is the
single cheapest large win available on the hot path. There is no delete or tombstone
support, so vectors can only be added or replaced. The metadata log is never compacted.
`VDTest` is a console app rather than a test project, so there is nothing for `dotnet
test` or CI to run, and there is no BenchmarkDotNet project despite the performance
claims being central to the README. Concurrent inserts still build a slightly different
graph than serial ones — inherent to HNSW, but worth stating in the docs rather than
leaving for someone to discover. And the empty `VectorDatabase` project should probably
either become the public API surface or be deleted.
