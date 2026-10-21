package hnsw_test

import (
	"strconv"
	"testing"

	"github.com/rohanreddy79/hnswgo/hnsw"
	"github.com/rohanreddy79/hnswgo/internal/synth"
)

func BenchmarkSearch(b *testing.B) {
	blobs := synth.NewBlobs(1, 64, 50, 0.5)
	base := blobs.Sample(10000)
	queries := blobs.Sample(256)
	idx, err := hnsw.New(hnsw.Config{Dim: 64, Seed: 1})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range base {
		if err := idx.Add(uint64(i), v); err != nil {
			b.Fatal(err)
		}
	}
	for _, ef := range []int{16, 64} {
		b.Run("ef="+strconv.Itoa(ef), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				idx.Search(queries[i%len(queries)], 10, ef)
			}
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	blobs := synth.NewBlobs(2, 64, 50, 0.5)
	base := blobs.Sample(2000)
	extra := blobs.Sample(b.N)
	idx, err := hnsw.New(hnsw.Config{Dim: 64, Seed: 1})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range base {
		if err := idx.Add(uint64(i), v); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := idx.Add(uint64(len(base)+i), extra[i]); err != nil {
			b.Fatal(err)
		}
	}
}
