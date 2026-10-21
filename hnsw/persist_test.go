package hnsw_test

import (
	"bytes"
	"errors"
	"slices"
	"testing"

	"github.com/rohanreddy79/hnswgo/hnsw"
	"github.com/rohanreddy79/hnswgo/internal/synth"
)

func savedIndex(t *testing.T) (*hnsw.Index, []byte) {
	t.Helper()
	vecs := synth.Uniform(71, 1000, 16)
	idx, _ := build(t, hnsw.Config{Dim: 16, M: 8, EfConstruction: 50, Metric: hnsw.Cosine, Seed: 5, KeepPruned: true}, vecs)
	for id := uint64(0); id < 1000; id += 7 {
		idx.Delete(id)
	}
	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}
	return idx, buf.Bytes()
}

func TestSaveLoadRoundTrip(t *testing.T) {
	orig, data := savedIndex(t)
	loaded, err := hnsw.Load(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Config() != orig.Config() {
		t.Fatalf("config: %+v vs %+v", loaded.Config(), orig.Config())
	}
	if loaded.Len() != orig.Len() {
		t.Fatalf("Len: %d vs %d", loaded.Len(), orig.Len())
	}
	if a, b := orig.Stats(), loaded.Stats(); !statsEqual(a, b) || a.Deleted != b.Deleted {
		t.Fatalf("stats differ: %+v vs %+v", a, b)
	}
	queries := synth.Uniform(72, 50, 16)
	for _, q := range queries {
		if a, b := orig.Search(q, 10, 40), loaded.Search(q, 10, 40); !slices.Equal(a, b) {
			t.Fatalf("results differ after load: %v vs %v", a, b)
		}
	}
	// The level generator state is restored, so continuing to build on
	// either side yields the same graph.
	for i, v := range synth.Uniform(73, 200, 16) {
		if err := orig.Add(uint64(5000+i), v); err != nil {
			t.Fatal(err)
		}
		if err := loaded.Add(uint64(5000+i), v); err != nil {
			t.Fatal(err)
		}
	}
	for _, q := range queries {
		if a, b := orig.Search(q, 10, 40), loaded.Search(q, 10, 40); !slices.Equal(a, b) {
			t.Fatalf("results diverge after post-load adds: %v vs %v", a, b)
		}
	}
	// A second save of the loaded index is byte-identical.
	var again bytes.Buffer
	if err := orig.Save(&again); err != nil {
		t.Fatal(err)
	}
	var again2 bytes.Buffer
	if err := loaded.Save(&again2); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(again.Bytes(), again2.Bytes()) {
		t.Fatal("re-saved bytes differ between original and loaded index")
	}
}

func TestLoadEmptyIndex(t *testing.T) {
	idx, err := hnsw.New(hnsw.Config{Dim: 3})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := hnsw.Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 0 || loaded.Search([]float32{1, 2, 3}, 1, 1) != nil {
		t.Fatal("loaded empty index is not empty")
	}
	if err := loaded.Add(1, []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
}

func TestLoadRejectsBadInput(t *testing.T) {
	_, data := savedIndex(t)

	t.Run("magic", func(t *testing.T) {
		bad := slices.Clone(data)
		bad[0] = 'X'
		if _, err := hnsw.Load(bytes.NewReader(bad)); !errors.Is(err, hnsw.ErrCorrupt) {
			t.Fatalf("got %v, want ErrCorrupt", err)
		}
	})
	t.Run("version", func(t *testing.T) {
		bad := slices.Clone(data)
		bad[4] = 99
		if _, err := hnsw.Load(bytes.NewReader(bad)); !errors.Is(err, hnsw.ErrVersion) {
			t.Fatalf("got %v, want ErrVersion", err)
		}
	})
	t.Run("truncated", func(t *testing.T) {
		for _, n := range []int{0, 3, 10, 40, 100, len(data) / 2, len(data) - 5, len(data) - 1} {
			if _, err := hnsw.Load(bytes.NewReader(data[:n])); err == nil {
				t.Fatalf("truncated to %d bytes: expected an error", n)
			}
		}
	})
	t.Run("flipped byte", func(t *testing.T) {
		for _, pos := range []int{60, 200, len(data) / 2, len(data) - 100, len(data) - 2} {
			bad := slices.Clone(data)
			bad[pos] ^= 0x5A
			if _, err := hnsw.Load(bytes.NewReader(bad)); err == nil {
				t.Fatalf("flipped byte %d: expected an error", pos)
			}
		}
	})
	t.Run("empty", func(t *testing.T) {
		if _, err := hnsw.Load(bytes.NewReader(nil)); err == nil {
			t.Fatal("expected an error for empty input")
		}
	})
}
