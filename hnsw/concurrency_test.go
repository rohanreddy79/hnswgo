package hnsw_test

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/rohanreddy79/hnswgo/hnsw"
	"github.com/rohanreddy79/hnswgo/internal/synth"
)

// TestConcurrentAddSearchDelete is meant for the race detector: writers,
// readers and a deleter hammer one index, and the final count must add up.
func TestConcurrentAddSearchDelete(t *testing.T) {
	const writers, perWriter, readers = 4, 300, 4
	idx, err := hnsw.New(hnsw.Config{Dim: 16, M: 8, EfConstruction: 40, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	vecs := synth.Uniform(81, writers*perWriter, 16)
	queries := synth.Uniform(82, 32, 16)

	var wg sync.WaitGroup
	var done atomic.Bool
	var searches atomic.Int64
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			for i := 0; !done.Load(); i++ {
				q := queries[(i+r)%len(queries)]
				res := idx.Search(q, 5, 20)
				for j := 1; j < len(res); j++ {
					if res[j].Distance < res[j-1].Distance {
						t.Errorf("results out of order: %v", res)
						return
					}
				}
				searches.Add(1)
			}
		}(r)
	}
	var deletes atomic.Int64
	var ww sync.WaitGroup
	for w := 0; w < writers; w++ {
		ww.Add(1)
		go func(w int) {
			defer ww.Done()
			for i := 0; i < perWriter; i++ {
				id := uint64(w*perWriter + i)
				if err := idx.Add(id, vecs[id]); err != nil {
					t.Errorf("Add %d: %v", id, err)
					return
				}
				// Each writer tombstones a tenth of its own inserts.
				if i%10 == 9 && idx.Delete(id) {
					deletes.Add(1)
				}
			}
		}(w)
	}
	ww.Wait()
	done.Store(true)
	wg.Wait()

	want := writers*perWriter - int(deletes.Load())
	if idx.Len() != want {
		t.Fatalf("Len = %d, want %d", idx.Len(), want)
	}
	if searches.Load() == 0 {
		t.Fatal("readers never ran")
	}
	// After the dust settles every live vector is its own nearest neighbour.
	for id := uint64(0); id < writers*perWriter; id += 37 {
		if !idx.Contains(id) {
			continue
		}
		res := idx.Search(vecs[id], 1, 32)
		if len(res) != 1 || res[0].ID != id {
			t.Fatalf("self-query %d: got %v", id, res)
		}
	}
}

// Save must coexist with concurrent searches and finish while writers wait.
func TestConcurrentSaveAndSearch(t *testing.T) {
	vecs := synth.Uniform(91, 500, 8)
	idx, _ := build(t, hnsw.Config{Dim: 8, Seed: 1}, vecs)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, q := range vecs[:100] {
				idx.Search(q, 3, 10)
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := idx.Save(discard{}); err != nil {
			t.Error(err)
		}
	}()
	wg.Wait()
}

type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }
