package hnsw_test

import (
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/rohanreddy79/hnswgo/bruteforce"
	"github.com/rohanreddy79/hnswgo/hnsw"
	"github.com/rohanreddy79/hnswgo/internal/synth"
)

// build returns an HNSW index and a brute-force twin over the same vectors.
func build(t *testing.T, cfg hnsw.Config, vecs [][]float32) (*hnsw.Index, *bruteforce.Index) {
	t.Helper()
	idx, err := hnsw.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	exact, err := bruteforce.New(cfg.Dim, cfg.Metric)
	if err != nil {
		t.Fatalf("bruteforce.New: %v", err)
	}
	for i, v := range vecs {
		if err := idx.Add(uint64(i), v); err != nil {
			t.Fatalf("Add %d: %v", i, err)
		}
		if err := exact.Add(uint64(i), v); err != nil {
			t.Fatalf("bruteforce Add %d: %v", i, err)
		}
	}
	return idx, exact
}

func meanRecall(idx *hnsw.Index, exact *bruteforce.Index, queries [][]float32, k, ef int) float64 {
	var sum float64
	for _, q := range queries {
		sum += bruteforce.Recall(idx.Search(q, k, ef), exact.Search(q, k))
	}
	return sum / float64(len(queries))
}

func TestRecallOnClusteredData(t *testing.T) {
	blobs := synth.NewBlobs(1, 32, 20, 0.5)
	base := blobs.Sample(5000)
	queries := blobs.Sample(200)
	idx, exact := build(t, hnsw.Config{Dim: 32, M: 16, EfConstruction: 200, Seed: 7}, base)

	if r := meanRecall(idx, exact, queries, 10, 64); r < 0.95 {
		t.Fatalf("recall@10 at ef=64 = %.3f, want >= 0.95", r)
	}
	// Recall must not decrease with ef: a larger beam strictly explores more.
	lo := meanRecall(idx, exact, queries, 10, 10)
	hi := meanRecall(idx, exact, queries, 10, 256)
	if hi < lo {
		t.Fatalf("recall fell from %.3f at ef=10 to %.3f at ef=256", lo, hi)
	}
	if hi < 0.99 {
		t.Fatalf("recall@10 at ef=256 = %.3f, want >= 0.99", hi)
	}
}

func TestSearchIsExactWhenBeamCoversIndex(t *testing.T) {
	for _, metric := range []hnsw.Metric{hnsw.L2, hnsw.Cosine, hnsw.Dot} {
		t.Run(metric.String(), func(t *testing.T) {
			vecs := synth.Uniform(uint64(metric)+3, 300, 16)
			idx, exact := build(t, hnsw.Config{Dim: 16, M: 8, EfConstruction: 64, Metric: metric, Seed: 1}, vecs)
			queries := synth.Uniform(99, 50, 16)
			for _, q := range queries {
				got := idx.Search(q, 10, 300)
				want := exact.Search(q, 10)
				if !slices.Equal(ids(got), ids(want)) {
					t.Fatalf("metric %s: got %v want %v", metric, got, want)
				}
				for i := range got {
					if diff := got[i].Distance - want[i].Distance; diff > 1e-4 || diff < -1e-4 {
						t.Fatalf("distance %d: got %g want %g", i, got[i].Distance, want[i].Distance)
					}
				}
			}
		})
	}
}

func TestEmptyIndexSearchReturnsNil(t *testing.T) {
	idx, err := hnsw.New(hnsw.Config{Dim: 4})
	if err != nil {
		t.Fatal(err)
	}
	if got := idx.Search([]float32{1, 2, 3, 4}, 5, 10); got != nil {
		t.Fatalf("got %v, want nil", got)
	}
	if idx.Len() != 0 {
		t.Fatalf("Len = %d, want 0", idx.Len())
	}
}

func TestKLargerThanLen(t *testing.T) {
	vecs := synth.Uniform(5, 5, 8)
	idx, exact := build(t, hnsw.Config{Dim: 8, Seed: 1}, vecs)
	q := make([]float32, 8)
	got := idx.Search(q, 50, 10)
	if len(got) != 5 {
		t.Fatalf("got %d results, want 5", len(got))
	}
	if !slices.Equal(ids(got), ids(exact.Search(q, 50))) {
		t.Fatalf("order differs from exact: %v", got)
	}
	if got := idx.Search(q, 0, 10); got != nil {
		t.Fatalf("k=0 should return nil, got %v", got)
	}
}

func TestDuplicateIDRejected(t *testing.T) {
	idx, _ := build(t, hnsw.Config{Dim: 3}, [][]float32{{1, 0, 0}})
	err := idx.Add(0, []float32{0, 1, 0})
	if !errors.Is(err, hnsw.ErrDuplicateID) {
		t.Fatalf("got %v, want ErrDuplicateID", err)
	}
	if idx.Len() != 1 {
		t.Fatalf("Len = %d after rejected add, want 1", idx.Len())
	}
}

func TestDimensionMismatch(t *testing.T) {
	idx, _ := build(t, hnsw.Config{Dim: 3}, [][]float32{{1, 0, 0}})
	if err := idx.Add(1, []float32{1, 2}); !errors.Is(err, hnsw.ErrDimension) {
		t.Fatalf("Add: got %v, want ErrDimension", err)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("Search with wrong dimension should panic")
		}
	}()
	idx.Search([]float32{1, 2}, 1, 1)
}

func TestInvalidConfig(t *testing.T) {
	for _, cfg := range []hnsw.Config{
		{},
		{Dim: 4, M: 1},
		{Dim: 4, EfConstruction: -1},
		{Dim: 4, Metric: hnsw.Metric(9)},
		{Dim: 4, Select: hnsw.Selection(9)},
	} {
		if _, err := hnsw.New(cfg); !errors.Is(err, hnsw.ErrConfig) {
			t.Errorf("New(%+v): got %v, want ErrConfig", cfg, err)
		}
	}
	idx, err := hnsw.New(hnsw.Config{Dim: 4})
	if err != nil {
		t.Fatal(err)
	}
	if c := idx.Config(); c.M != 16 || c.EfConstruction != 200 {
		t.Fatalf("defaults not applied: %+v", c)
	}
}

func TestDeleteHidesVectorsAndShrinksLen(t *testing.T) {
	vecs := synth.Uniform(11, 400, 16)
	idx, exact := build(t, hnsw.Config{Dim: 16, Seed: 3}, vecs)
	rng := rand.New(rand.NewPCG(4, 5))
	deleted := map[uint64]bool{}
	for len(deleted) < 100 {
		id := uint64(rng.IntN(400))
		if deleted[id] {
			continue
		}
		if !idx.Delete(id) {
			t.Fatalf("Delete(%d) returned false for a live id", id)
		}
		deleted[id] = true
	}
	if idx.Len() != 300 {
		t.Fatalf("Len = %d, want 300", idx.Len())
	}
	for id := range deleted {
		if idx.Delete(id) {
			t.Fatalf("second Delete(%d) returned true", id)
		}
		if idx.Contains(id) {
			t.Fatalf("Contains(%d) true after delete", id)
		}
	}
	// A wide beam over a small index must return every live vector and no
	// tombstone, in exact order.
	for _, q := range synth.Uniform(12, 30, 16) {
		got := idx.Search(q, 400, 400)
		if len(got) != 300 {
			t.Fatalf("got %d results, want 300 live", len(got))
		}
		for _, r := range got {
			if deleted[r.ID] {
				t.Fatalf("deleted id %d returned", r.ID)
			}
		}
		want := exact.Search(q, 400)
		want = slices.DeleteFunc(want, func(r hnsw.Result) bool { return deleted[r.ID] })
		if !slices.Equal(ids(got), ids(want)) {
			t.Fatal("live results differ from exact search")
		}
	}
	// A deleted id can be inserted again and becomes searchable.
	if err := idx.Add(0, vecs[0]); deleted[0] && err != nil {
		t.Fatalf("re-add deleted id: %v", err)
	}
}

func TestDeleteEverythingThenAdd(t *testing.T) {
	vecs := synth.Uniform(21, 50, 8)
	idx, _ := build(t, hnsw.Config{Dim: 8, Seed: 1}, vecs)
	for i := range vecs {
		idx.Delete(uint64(i))
	}
	if got := idx.Search(vecs[0], 5, 50); got != nil {
		t.Fatalf("all deleted: got %v, want nil", got)
	}
	// New nodes must link through the tombstones and be found again.
	fresh := synth.Uniform(22, 50, 8)
	for i, v := range fresh {
		if err := idx.Add(uint64(1000+i), v); err != nil {
			t.Fatal(err)
		}
	}
	for i, v := range fresh {
		got := idx.Search(v, 1, 16)
		if len(got) != 1 || got[0].ID != uint64(1000+i) {
			t.Fatalf("query %d: got %v, want id %d", i, got, 1000+i)
		}
	}
}

func TestCosineIgnoresScale(t *testing.T) {
	vecs := synth.Uniform(31, 500, 16)
	scaled := make([][]float32, len(vecs))
	rng := rand.New(rand.NewPCG(1, 2))
	for i, v := range vecs {
		s := float32(rng.Float64()*9 + 0.1)
		scaled[i] = make([]float32, len(v))
		for j := range v {
			scaled[i][j] = v[j] * s
		}
	}
	a, _ := build(t, hnsw.Config{Dim: 16, Metric: hnsw.Cosine, Seed: 1}, vecs)
	b, _ := build(t, hnsw.Config{Dim: 16, Metric: hnsw.Cosine, Seed: 1}, scaled)
	for _, q := range synth.Uniform(32, 20, 16) {
		ra, rb := a.Search(q, 10, 500), b.Search(q, 10, 500)
		if !slices.Equal(ids(ra), ids(rb)) {
			t.Fatalf("scaling changed cosine neighbours: %v vs %v", ra, rb)
		}
		for i := range ra {
			if d := ra[i].Distance; d < -1e-6 || d > 2+1e-6 {
				t.Fatalf("cosine distance %g outside [0, 2]", d)
			}
		}
	}
	// A zero vector sits at distance 1 (undefined angle) rather than NaN.
	zero, _ := build(t, hnsw.Config{Dim: 2, Metric: hnsw.Cosine}, [][]float32{{0, 0}})
	if got := zero.Search([]float32{1, 0}, 1, 1); len(got) != 1 || got[0].Distance != 1 {
		t.Fatalf("zero vector: got %v", got)
	}
}

func TestSelectionVariantsAllSearchable(t *testing.T) {
	blobs := synth.NewBlobs(41, 16, 8, 0.4)
	base := blobs.Sample(1500)
	queries := blobs.Sample(100)
	for _, tc := range []struct {
		name string
		cfg  hnsw.Config
	}{
		{"simple", hnsw.Config{Dim: 16, Select: hnsw.SelectSimple}},
		{"heuristic", hnsw.Config{Dim: 16}},
		{"heuristic+extend", hnsw.Config{Dim: 16, ExtendCandidates: true}},
		{"heuristic+keepPruned", hnsw.Config{Dim: 16, KeepPruned: true}},
		{"heuristic+both", hnsw.Config{Dim: 16, ExtendCandidates: true, KeepPruned: true}},
		{"small M", hnsw.Config{Dim: 16, M: 4, EfConstruction: 32}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			idx, exact := build(t, tc.cfg, base)
			if r := meanRecall(idx, exact, queries, 10, 64); r < 0.9 {
				t.Fatalf("recall@10 = %.3f, want >= 0.9", r)
			}
		})
	}
}

func TestBuildIsDeterministic(t *testing.T) {
	vecs := synth.Uniform(51, 800, 12)
	a, _ := build(t, hnsw.Config{Dim: 12, Seed: 9}, vecs)
	b, _ := build(t, hnsw.Config{Dim: 12, Seed: 9}, vecs)
	if !statsEqual(a.Stats(), b.Stats()) {
		t.Fatal("same seed and data produced different graphs")
	}
	for _, q := range synth.Uniform(52, 20, 12) {
		if !slices.Equal(a.Search(q, 10, 20), b.Search(q, 10, 20)) {
			t.Fatal("same seed and data produced different results")
		}
	}
}

func TestStats(t *testing.T) {
	vecs := synth.Uniform(61, 1000, 8)
	idx, _ := build(t, hnsw.Config{Dim: 8, M: 8, Seed: 2}, vecs)
	idx.Delete(3)
	st := idx.Stats()
	if st.Nodes != 1000 || st.Live != 999 || st.Deleted != 1 {
		t.Fatalf("counts: %+v", st)
	}
	total := 0
	for _, c := range st.LevelCounts {
		total += c
	}
	if total != st.Nodes || len(st.LevelCounts) != st.MaxLevel+1 {
		t.Fatalf("level counts %v do not cover %d nodes", st.LevelCounts, st.Nodes)
	}
	if st.Edges == 0 || st.Bytes < int64(1000*8*4) {
		t.Fatalf("implausible edges/bytes: %+v", st)
	}
	if st.Metric != "l2" || st.Selection != "heuristic" || st.M != 8 {
		t.Fatalf("config echo: %+v", st)
	}
}

func ids(rs []hnsw.Result) []uint64 {
	out := make([]uint64, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

func statsEqual(a, b hnsw.Stats) bool {
	return a.Nodes == b.Nodes && a.Edges == b.Edges && a.MaxLevel == b.MaxLevel &&
		slices.Equal(a.LevelCounts, b.LevelCounts)
}
