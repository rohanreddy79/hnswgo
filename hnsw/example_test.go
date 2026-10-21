package hnsw_test

import (
	"bytes"
	"fmt"

	"github.com/rohanreddy79/hnswgo/hnsw"
)

// This is the README's library walkthrough; go test compiles and checks it.
func Example() {
	idx, err := hnsw.New(hnsw.Config{Dim: 3, M: 16, EfConstruction: 200, Metric: hnsw.Cosine, Seed: 1})
	if err != nil {
		panic(err)
	}
	vectors := map[uint64][]float32{
		1: {1, 0, 0},
		2: {0, 1, 0},
		3: {0.9, 0.1, 0}, // same direction as 1, scaled: cosine ignores length
	}
	for _, id := range []uint64{1, 2, 3} {
		if err := idx.Add(id, vectors[id]); err != nil {
			panic(err)
		}
	}

	// k = 2 nearest with a beam of ef = 32; results are nearest first.
	for _, r := range idx.Search([]float32{1, 0, 0}, 2, 32) {
		fmt.Printf("id=%d distance=%.3f\n", r.ID, r.Distance)
	}

	idx.Delete(3) // tombstoned: still routable, never returned
	fmt.Println("live:", idx.Len(), "nearest:", idx.Search([]float32{1, 0, 0}, 1, 32)[0].ID)

	// Save and Load round-trip the graph and the level generator state.
	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		panic(err)
	}
	size := buf.Len()
	loaded, err := hnsw.Load(&buf)
	if err != nil {
		panic(err)
	}
	fmt.Println("loaded:", loaded.Len(), "bytes:", size)

	// Output:
	// id=1 distance=0.000
	// id=3 distance=0.006
	// live: 2 nearest: 1
	// loaded: 2 bytes: 199
}
