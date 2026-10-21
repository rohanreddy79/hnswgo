# hnswgo

**The data structure under every vector database, written from the paper in readable, dependency-free Go: a Hierarchical Navigable Small World index with deletes, persistence, concurrent reads, an HTTP service and a recall-vs-QPS benchmark against exact search.**

[![ci](https://github.com/rohanreddy79/hnswgo/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanreddy79/hnswgo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why

Approximate nearest-neighbour search is what turns embeddings into a product: every RAG pipeline, recommender and semantic search sits on an ANN index, and HNSW (Malkov & Yashunin, 2016) is the algorithm most of them use. The interesting part is not the API but the graph: why a greedy walk over a multi-layer skip-list-like graph finds near-exact neighbours in microseconds, and why the *neighbour selection rule* used at build time matters more than any search-time knob. This repo implements the paper's Algorithms 1 to 5 in about 900 lines of Go with no dependencies, measures the choices that matter (heuristic vs simple selection, ef vs recall, single vs multi-thread throughput) and documents the trade-offs honestly.

## Features

- `hnsw.Index`: `Add`, `Search(q, k, ef)`, `Delete` (tombstone), `Len`, `Contains`, `Stats`, `Save`/`Load`.
- Metrics: squared L2, cosine (vectors normalised on insert so a query is one dot product) and negated dot product; 4-wide unrolled kernels.
- Build: level `floor(-ln U · mL)` with `mL = 1/ln M`, `Mmax0 = 2M`, beam width `efConstruction`, the paper's heuristic neighbour selection (Algorithm 4) with `ExtendCandidates` and `KeepPruned`, or simple selection for comparison.
- Search: greedy descent through the upper layers, then an `ef`-bounded best-first search on layer 0 with explicit candidate (min) and result (max) heaps.
- Zero-allocation traversal: a generation-counter visited set (`[]uint32`, cleared by incrementing a counter) and pooled searchers; one allocation per query, for the result slice.
- Concurrency: `sync.RWMutex`, many concurrent `Search`, serialised `Add`/`Delete`; tested under the race detector.
- Deletes are tombstones: skipped in results, still traversable, and new nodes can still link through them.
- Persistence: versioned little-endian binary format with a CRC-32 trailer and full structural validation; the level generator's state is saved so a loaded index keeps building deterministically.
- `hnswgo serve`: JSON API (`POST /vectors` upsert, `POST /search`, `DELETE /vectors/{id}`, `POST /save`, `GET /stats`) with atomic save and `--load`.
- `hnswgo bench`: seeded clustered Gaussian data, recall@k and QPS per `ef`, brute-force baseline, serialised and in-memory size.
- `bruteforce` package: exact k-NN and `Recall`, the ground truth for every recall figure here.

## Quickstart

Go 1.23+ and nothing else. From a checkout of this repository:

```sh
CGO_ENABLED=0 go build -o hnswgo ./cmd/hnswgo
./hnswgo bench --n 20000 --dim 64 --k 10 --queries 500        # ~11 s, ~60 MB
./hnswgo serve --addr :8080 --dim 4 &
curl -s -X POST localhost:8080/vectors -d '[{"id":1,"vector":[1,0,0,0]},{"id":2,"vector":[0,1,0,0]},{"id":3,"vector":[0.9,0.1,0,0]}]'
curl -s -X POST localhost:8080/search -d '{"vector":[1,0,0,0],"k":2,"ef":32}'
```

Expected output (trimmed; the full bench table is in [Benchmarks](#benchmarks)):

```
| ef | recall@10 | QPS (1 thread) | QPS (8 threads) | speedup vs brute force (1 thread) |
|---|---|---|---|---|
| 16 | 0.931 | 92307 | 464802 | 32x |
| 64 | 0.999 | 35689 | 176726 | 12x |
{"replaced":0,"upserted":3}
{"results":[{"id":1,"distance":0},{"id":3,"distance":0.020000005}]}
```

As a library (this is `hnsw/example_test.go`, so `go test` keeps it honest):

```go
idx, err := hnsw.New(hnsw.Config{Dim: 3, M: 16, EfConstruction: 200, Metric: hnsw.Cosine, Seed: 1})
idx.Add(1, []float32{1, 0, 0})
idx.Add(2, []float32{0, 1, 0})
idx.Add(3, []float32{0.9, 0.1, 0})
res := idx.Search([]float32{1, 0, 0}, 2, 32) // []hnsw.Result{{ID: 1, Distance: 0}, {ID: 3, Distance: 0.006}}
idx.Delete(3)                                // tombstoned: still routable, never returned
idx.Save(w)                                  // versioned binary; hnsw.Load(r) restores it
```

HTTP API summary: `POST /vectors` takes `[{"id": u64, "vector": [f32...]}]` and upserts (an existing id is deleted then re-added); `POST /search` takes `{"vector", "k" (default 10), "ef" (default 64)}`; `DELETE /vectors/{id}` returns 404 for unknown ids; `POST /save {"path"}` writes a temp file beside `path` and renames it into place; `GET /stats` returns the `hnsw.Stats` JSON. Errors are `{"error": "..."}` with 4xx/5xx. There is no auth: it is a local tool.

## How it works

```
             query q
                │
  layer 2   [E]─┼─────────────────────[ ]              ~n/M² nodes
             │  ▼ greedy: hop to the closest neighbour while one is closer
  layer 1   [E]─[a]──────[b]──────[ ]─────[ ]           ~n/M nodes, ≤ M edges each
                          │
                          ▼ continue from the local minimum, one layer down
  layer 0   [ ]─[ ]─[a]══[b]══[c]══[d]──[ ]─[ ]─[ ]     all n nodes, ≤ 2M edges each
                      ╰── beam search: expand the nearest candidate, keep the ef
                          best seen, stop when the nearest unexpanded candidate
                          is worse than the ef-th best; return the top k ──╯
```

Every node lives on layers 0..level, where level is geometric with ratio 1/M, so the upper layers are an exponentially sparser "express lane". A query descends greedily from the entry point `[E]` (the highest node) to layer 1, then runs the bounded beam search on layer 0. Insertion is the same descent followed by a beam search with width `efConstruction` on each layer the new node belongs to; the candidates found there are filtered by the selection heuristic into the node's edges, and back-edges are added to the neighbours, pruning any that exceed their degree cap.

| Path | Role |
|---|---|
| `hnsw/hnsw.go` | `Config`, `Add` (Algorithm 1), `Delete`, `Search` (Algorithm 5), `Stats`, locking |
| `hnsw/search.go` | searcher scratch state, generation-counter visited set, greedy descent, beam search (Algorithm 2) |
| `hnsw/select.go` | neighbour selection: heuristic (Algorithm 4) with `extendCandidates`/`keepPruned`, simple (Algorithm 3) |
| `hnsw/heap.go` | min-heap (candidates) and max-heap (results) over 8-byte `(distance, node)` items |
| `hnsw/metric.go` | L2 / cosine / dot, reference `Distance` plus the unrolled kernels |
| `hnsw/persist.go` | `Save`/`Load`: versioned little-endian format, CRC-32, validation |
| `bruteforce/` | exact scan and `Recall`; the measuring stick |
| `internal/synth/` | seeded Gaussian blobs and uniform data |
| `cmd/hnswgo/` | `serve` (JSON HTTP) and `bench` |

**The three parameters.** `M` is how many edges a node gets when inserted and the degree cap on the upper layers (layer 0 allows `2M`); it sets memory per node and the graph's connectivity, and 16 is the usual default. `efConstruction` is the beam width used while inserting; a wider beam finds better candidates for the heuristic to choose from, so it buys graph quality at build time and does not affect query cost directly. `ef` is the beam width at query time (raised to `k` if smaller); it is the runtime knob that trades throughput for recall, as the table below shows.

**Design decisions.**

1. *Memory layout: one flat `[]float32` plus per-node neighbour slices.* Vectors are stored contiguously and addressed by a `uint32` internal index, so a node's vector is a bounds-checked sub-slice and the adjacency lists hold 4-byte ids instead of 8-byte pointers or `uint64` ids. Per-node slices (`[][]uint32`, one per layer) keep the code idiomatic and let layer 0 reserve `2M` slots while upper layers reserve `M`. The alternative, hnswlib's single byte arena with fixed-size slots per node, is faster to walk but turns every access into pointer arithmetic; `[][]float32` was rejected because 20 000 separate vector allocations and 24-byte slice headers scatter the hot data.
2. *Generation-counter visited set.* The beam search must not expand a node twice. A `map[uint32]struct{}` per query would allocate on every search; zeroing a `[]bool` of length n per query is O(n). Instead each searcher keeps a `[]uint32` and a generation counter: "visited" means `visited[i] == gen`, and clearing is `gen++` (with a real clear on wrap-around). Searchers live in a `sync.Pool`, so the set is per-searcher, never shared, which is what keeps the race detector quiet and makes a query allocate exactly once (the result slice).
3. *Heuristic selection, measured rather than assumed.* Simple selection keeps the M nearest candidates. On clustered data those are all inside the new node's own cluster, so clusters become cliques with few bridges and the greedy descent lands in the wrong one. The heuristic keeps a candidate only if it is closer to the new node than to every neighbour already kept, which favours edges in distinct directions and builds the bridges. Same data, same `M`, same `efConstruction`: recall@10 at ef=64 is 0.999 with the heuristic and 0.889 without, at the same QPS. `extendCandidates` is applied only to the new node's own selection, not when pruning an existing node's list, because there the pool is already the node's full adjacency and widening it would cost O(M²) distances per insert for little gain.
4. *Tombstones instead of unlinking.* Edges are directed and stored only on the source node, so physically removing a node means finding and repairing every in-edge, and doing it badly disconnects the graph. A tombstone keeps the node as a routing waypoint, drops it from results, and (during construction) still lets new nodes link through it, so a region whose live nodes were all deleted stays reachable. The cost is memory and some wasted expansion until a rebuild; `Stats.Deleted` says when that is due.
5. *One writer, many readers.* The paper describes lock-free inserts with per-node locks; this index takes a plain `sync.RWMutex`. Searches run fully in parallel (8 threads give 4.9x the single-thread QPS) and each `Add` holds the write lock for about 70 µs, which is enough for 14 000 inserts/s and keeps the construction code free of ordering subtleties.
6. *A real file format.* `Save` writes a magic, a version, the config, the PCG level-generator state, the graph and the vectors, then a CRC-32. `Load` validates every count and index before trusting it, grows slices incrementally so a corrupt count fails on a short read instead of a huge allocation, and checks the CRC over exactly the bytes consumed (hashing beneath the `bufio.Reader` would include its read-ahead, which the empty-index test caught).

**What a production vector database adds on top.** Quantisation (scalar or product) to shrink vectors 4 to 32x and use int8 SIMD kernels; metadata filtering inside the traversal rather than post-filtering the top-k; sharding and replication with a segment per shard and merge-on-compaction; memory-mapped or disk-resident storage (DiskANN-style); background compaction to drop tombstones; and hand-written SIMD distance kernels, which are the single biggest per-query win this pure-Go version leaves on the table.

## Benchmarks

Apple M3, 8 GB RAM, macOS 27.0, CPU only, `go1.25.6 darwin/arm64`, `CGO_ENABLED=0`. The command below took 10.7 s wall clock and 56 MB peak RSS.

```sh
CGO_ENABLED=0 go run ./cmd/hnswgo bench --n 20000 --dim 64 --k 10 --queries 500
```

| Parameter | Value |
|---|---|
| Vectors | 20000 x 64 (50 Gaussian clusters, spread 0.50, seed 1) |
| Metric | l2 |
| M / efConstruction / selection | 16 / 200 / heuristic |
| Build time | 1.377s (14526 inserts/s) |
| Graph | 340639 edges, max level 4, level counts [18730 1194 70 5 1] |
| Index size | 6828052 bytes serialised, ~10252752 bytes in memory |
| Brute-force QPS (1 thread) | 2868 |
| Queries | 500, k = 10 |

| ef | recall@10 | QPS (1 thread) | QPS (8 threads) | speedup vs brute force (1 thread) |
|---|---|---|---|---|
| 16 | 0.931 | 92307 | 464802 | 32x |
| 32 | 0.988 | 59081 | 282682 | 21x |
| 64 | 0.999 | 35689 | 176726 | 12x |
| 128 | 1.000 | 19543 | 115324 | 7x |

Heuristic vs simple neighbour selection, same data and build parameters (`--select simple`):

| ef | recall@10, heuristic | recall@10, simple | QPS (1 thread), heuristic | QPS (1 thread), simple |
|---|---|---|---|---|
| 16 | 0.931 | 0.737 | 92307 | 81479 |
| 32 | 0.988 | 0.818 | 59081 | 52722 |
| 64 | 0.999 | 0.889 | 35689 | 34921 |
| 128 | 1.000 | 0.926 | 19543 | 23211 |

Micro-benchmarks (`CGO_ENABLED=0 go test -run xxx -bench . -benchmem ./hnsw/`, 10 000 x 64 index for search, 2 000 x 64 base for add):

```
BenchmarkSearch/ef=16-8    137617    8810 ns/op    160 B/op   1 allocs/op
BenchmarkSearch/ef=64-8     54339   22284 ns/op    160 B/op   1 allocs/op
BenchmarkAdd-8              15632   73878 ns/op   1996 B/op   2 allocs/op
```

Read the numbers for what they are: synthetic Gaussian clusters, not SIFT or GloVe, and a brute-force baseline that is itself plain Go. The ef-vs-recall curve and the heuristic-vs-simple gap are the robust findings; absolute QPS will differ on other data and hardware, and varies by roughly 10% between runs on this machine (recall does not: the build is seeded).

## Testing

```sh
CGO_ENABLED=0 go test -race ./...     # ~15 s on the machine above
```

- recall@10 >= 0.95 at ef=64 on 5 000 x 32 clustered vectors (actual: 1.000), and recall is monotone in ef;
- results are identical to brute force when ef covers the whole index, for all three metrics;
- deleted ids are never returned, `Len` shrinks, deleted ids can be re-added, and an index whose nodes were all deleted still accepts and finds new vectors;
- save/load gives identical results, identical bytes on re-save, and identical results after further inserts on both sides; bad magic, bad version, truncation and flipped bytes are all rejected;
- each metric kernel matches a float64 reference at every length 1..70; cosine ignores scale and handles the zero vector;
- heuristic selection drops shadowed candidates and `KeepPruned` tops up; `extendCandidates` reaches two-hop candidates; level draws match 1/M and 1/M² tails;
- concurrent `Add` + `Search` + `Delete` + `Save` under the race detector;
- every HTTP handler through `httptest`, including validation, atomic save and `--load`;
- `BenchmarkSearch` and `BenchmarkAdd`, plus a compiled `Example`.

CI (`.github/workflows/ci.yml`) runs `gofmt -l`, `go vet`, `go test -race` and `go build`.

## Roadmap / Limitations

- No SIMD: the distance kernels are plain Go with 4-wide unrolling. Assembly NEON/AVX kernels would cut per-distance cost several-fold and are the obvious next step.
- Deletes are tombstones and nothing reclaims them; a `Compact()` that rebuilds without tombstones is missing, so heavy churn costs memory and search work.
- Inserts are serialised by one write lock; the paper's fine-grained locking would let multi-core builds scale.
- No metadata filtering, quantisation or sharding; see "What a production vector database adds".
- `Save` is a full snapshot with no incremental log, and the HTTP server has no auth or TLS; it is a local tool, not a deployment.
- Benchmarks use synthetic data only; an ann-benchmarks (SIFT1M, GloVe) harness would make the recall numbers comparable with other libraries.

## License

MIT, see [LICENSE](LICENSE).
