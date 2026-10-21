package hnsw

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func naiveL2(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return s
}

func naiveDot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func naiveCosine(a, b []float32) float64 {
	return 1 - naiveDot(a, b)/math.Sqrt(naiveDot(a, a)*naiveDot(b, b))
}

func randVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.Float64()*4 - 2)
	}
	return v
}

// The unrolled kernels and the reference Distance must agree with a plain
// float64 loop at every length, including tails shorter than the unroll.
func TestMetricsMatchNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for dim := 1; dim <= 70; dim++ {
		for trial := 0; trial < 5; trial++ {
			a, b := randVec(rng, dim), randVec(rng, dim)
			check := func(name string, got float32, want float64) {
				t.Helper()
				if math.Abs(float64(got)-want) > 1e-3*(1+math.Abs(want)) {
					t.Fatalf("dim %d %s: got %g want %g", dim, name, got, want)
				}
			}
			check("l2", L2.Distance(a, b), naiveL2(a, b))
			check("dot", Dot.Distance(a, b), -naiveDot(a, b))
			check("cosine", Cosine.Distance(a, b), naiveCosine(a, b))

			// The index path: normalise copies, then use the unit kernel.
			an, bn := slices.Clone(a), slices.Clone(b)
			Cosine.prepare(an)
			Cosine.prepare(bn)
			check("cosine kernel", cosineUnit(an, bn), naiveCosine(a, b))
			check("dot kernel", negDot(a, b), -naiveDot(a, b))
			check("l2 kernel", l2Squared(a, b), naiveL2(a, b))
		}
	}
}

func TestPrepareCosine(t *testing.T) {
	v := []float32{3, 4}
	Cosine.prepare(v)
	if n := naiveDot(v, v); math.Abs(n-1) > 1e-6 {
		t.Fatalf("norm after prepare = %g, want 1", n)
	}
	zero := []float32{0, 0}
	Cosine.prepare(zero)
	if zero[0] != 0 || zero[1] != 0 || math.IsNaN(float64(zero[0])) {
		t.Fatalf("zero vector changed: %v", zero)
	}
	if Cosine.Distance(zero, v) != 1 {
		t.Fatalf("zero vector distance = %g, want 1", Cosine.Distance(zero, v))
	}
	l2 := []float32{3, 4}
	L2.prepare(l2)
	if l2[0] != 3 || l2[1] != 4 {
		t.Fatal("L2 prepare must not modify the vector")
	}
}

func TestParseRoundTrips(t *testing.T) {
	for _, m := range []Metric{L2, Cosine, Dot} {
		got, err := ParseMetric(m.String())
		if err != nil || got != m {
			t.Fatalf("ParseMetric(%q) = %v, %v", m.String(), got, err)
		}
	}
	if _, err := ParseMetric("manhattan"); err == nil {
		t.Fatal("expected error for unknown metric")
	}
	for _, s := range []Selection{SelectHeuristic, SelectSimple} {
		got, err := ParseSelection(s.String())
		if err != nil || got != s {
			t.Fatalf("ParseSelection(%q) = %v, %v", s.String(), got, err)
		}
	}
	if _, err := ParseSelection("random"); err == nil {
		t.Fatal("expected error for unknown selection")
	}
}

func TestHeapsOrder(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	var lo minHeap
	var hi maxHeap
	for i := 0; i < 500; i++ {
		it := item{dist: float32(rng.Float64()), idx: uint32(i)}
		lo.push(it)
		hi.push(it)
	}
	prev := float32(math.Inf(-1))
	for len(lo) > 0 {
		it := lo.pop()
		if it.dist < prev {
			t.Fatal("minHeap popped out of order")
		}
		prev = it.dist
	}
	prev = float32(math.Inf(1))
	for len(hi) > 0 {
		if hi.top().dist != hi[0].dist {
			t.Fatal("top must be the root")
		}
		it := hi.pop()
		if it.dist > prev {
			t.Fatal("maxHeap popped out of order")
		}
		prev = it.dist
	}
}

func TestVisitedSetGenerationWraps(t *testing.T) {
	s := &searcher{}
	s.reset(10)
	s.visit(3)
	s.gen = math.MaxUint32 // force the next reset to wrap
	s.reset(10)
	if s.gen != 1 {
		t.Fatalf("gen after wrap = %d, want 1", s.gen)
	}
	if s.visit(3) {
		t.Fatal("node 3 still marked visited after wrap")
	}
	if !s.visit(3) {
		t.Fatal("node 3 not marked after visit")
	}
	s.reset(20)
	if len(s.visited) != 20 {
		t.Fatalf("visited len = %d, want 20", len(s.visited))
	}
}

// Candidate b lies almost on the segment from q to a, so a "shadows" it: the
// heuristic must keep a and c (two directions) and drop b, unless KeepPruned
// tops the list up. Simple selection keeps the three nearest regardless.
func TestHeuristicSelection(t *testing.T) {
	pts := [][]float32{{1, 0}, {1.1, 0.1}, {0, 1.05}, {-5, -5}}
	q := []float32{0, 0}
	build := func(cfg Config) (*Index, []item) {
		x, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		for i, p := range pts {
			x.vecs = append(x.vecs, p...)
			x.nodes = append(x.nodes, node{id: uint64(i), neighbors: [][]uint32{{}}})
		}
		cands := make([]item, len(pts))
		for i, p := range pts {
			cands[i] = item{x.dist(q, p), uint32(i)}
		}
		sortItems(cands)
		return x, cands
	}
	s := &searcher{}
	cases := []struct {
		name string
		cfg  Config
		m    int
		want []uint32
	}{
		{"heuristic drops shadowed", Config{Dim: 2}, 3, []uint32{0, 2, 3}},
		{"heuristic m=2", Config{Dim: 2}, 2, []uint32{0, 2}},
		{"keepPruned tops up", Config{Dim: 2, KeepPruned: true}, 4, []uint32{0, 2, 3, 1}},
		{"simple keeps nearest", Config{Dim: 2, Select: SelectSimple}, 3, []uint32{0, 2, 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, cands := build(tc.cfg)
			got := x.selectNeighbors(s, q, cands, tc.m, 0, false, nil)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

// extendCandidates must pull in neighbours of the candidates that the beam
// search missed. Node 2 is only reachable through node 0's adjacency.
func TestExtendCandidates(t *testing.T) {
	x, err := New(Config{Dim: 2, ExtendCandidates: true})
	if err != nil {
		t.Fatal(err)
	}
	pts := [][]float32{{1, 0}, {0, 1}, {-1, 0}}
	for i, p := range pts {
		x.vecs = append(x.vecs, p...)
		x.nodes = append(x.nodes, node{id: uint64(i), neighbors: [][]uint32{{}}})
	}
	x.nodes[0].neighbors[0] = []uint32{2}
	q := []float32{0, 0}
	cands := []item{{1, 0}, {1, 1}}
	s := &searcher{}
	got := x.selectNeighbors(s, q, cands, 3, 0, true, nil)
	if !slices.Contains(got, 2) {
		t.Fatalf("extended candidate 2 not selected: %v", got)
	}
}

// Levels follow floor(-ln U * mL): P(level >= 1) = 1/M. Checking the first
// two tail probabilities pins both the formula and the constant.
func TestRandomLevelDistribution(t *testing.T) {
	x, err := New(Config{Dim: 1, M: 16, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	const n = 200000
	var ge1, ge2 int
	for i := 0; i < n; i++ {
		l := x.randomLevel()
		if l >= 1 {
			ge1++
		}
		if l >= 2 {
			ge2++
		}
		if l > maxLevel {
			t.Fatalf("level %d exceeds cap", l)
		}
	}
	p1, p2 := float64(ge1)/n, float64(ge2)/n
	if math.Abs(p1-1.0/16) > 0.004 {
		t.Fatalf("P(level>=1) = %.4f, want ~%.4f", p1, 1.0/16)
	}
	if math.Abs(p2-1.0/256) > 0.001 {
		t.Fatalf("P(level>=2) = %.4f, want ~%.4f", p2, 1.0/256)
	}
}
