package bruteforce_test

import (
	"errors"
	"slices"
	"sort"
	"testing"

	"github.com/rohanreddy79/hnswgo/bruteforce"
	"github.com/rohanreddy79/hnswgo/hnsw"
	"github.com/rohanreddy79/hnswgo/internal/synth"
)

func TestSearchMatchesFullSort(t *testing.T) {
	for _, metric := range []hnsw.Metric{hnsw.L2, hnsw.Cosine, hnsw.Dot} {
		t.Run(metric.String(), func(t *testing.T) {
			vecs := synth.Uniform(uint64(metric)+1, 500, 12)
			idx, err := bruteforce.New(12, metric)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range vecs {
				if err := idx.Add(uint64(i), v); err != nil {
					t.Fatal(err)
				}
			}
			if idx.Len() != 500 {
				t.Fatalf("Len = %d", idx.Len())
			}
			for _, q := range synth.Uniform(9, 20, 12) {
				want := make([]hnsw.Result, len(vecs))
				for i, v := range vecs {
					want[i] = hnsw.Result{ID: uint64(i), Distance: metric.Distance(q, v)}
				}
				sort.SliceStable(want, func(i, j int) bool { return want[i].Distance < want[j].Distance })
				for _, k := range []int{1, 7, 50} {
					if got := idx.Search(q, k); !slices.Equal(got, want[:k]) {
						t.Fatalf("k=%d: got %v want %v", k, got, want[:k])
					}
				}
			}
		})
	}
}

func TestEdgeCases(t *testing.T) {
	if _, err := bruteforce.New(0, hnsw.L2); !errors.Is(err, hnsw.ErrConfig) {
		t.Fatalf("New(0): %v", err)
	}
	idx, err := bruteforce.New(2, hnsw.L2)
	if err != nil {
		t.Fatal(err)
	}
	if got := idx.Search([]float32{0, 0}, 3); got != nil {
		t.Fatalf("empty: got %v", got)
	}
	if err := idx.Add(1, []float32{1}); !errors.Is(err, hnsw.ErrDimension) {
		t.Fatalf("bad dim: %v", err)
	}
	for i, v := range [][]float32{{1, 0}, {0, 2}, {3, 3}} {
		if err := idx.Add(uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}
	got := idx.Search([]float32{0, 0}, 10)
	if len(got) != 3 || got[0].ID != 0 || got[1].ID != 1 || got[2].ID != 2 {
		t.Fatalf("k > Len: got %v", got)
	}
	if got := idx.Search([]float32{0, 0}, 0); got != nil {
		t.Fatalf("k=0: got %v", got)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("wrong query dimension should panic")
		}
	}()
	idx.Search([]float32{0}, 1)
}

func TestRecall(t *testing.T) {
	r := func(ids ...uint64) []hnsw.Result {
		out := make([]hnsw.Result, len(ids))
		for i, id := range ids {
			out[i].ID = id
		}
		return out
	}
	cases := []struct {
		got, want []hnsw.Result
		recall    float64
	}{
		{r(1, 2, 3), r(1, 2, 3), 1},
		{r(3, 2, 1), r(1, 2, 3), 1},
		{r(1, 9, 8), r(1, 2, 3), 1.0 / 3},
		{r(), r(1, 2), 0},
		{r(1), r(), 1},
	}
	for _, c := range cases {
		if got := bruteforce.Recall(c.got, c.want); got != c.recall {
			t.Errorf("Recall(%v, %v) = %g, want %g", c.got, c.want, got, c.recall)
		}
	}
}
