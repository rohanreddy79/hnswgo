package main

import (
	"bytes"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/rohanreddy79/hnswgo/hnsw"
)

func TestBenchProducesTable(t *testing.T) {
	var out bytes.Buffer
	err := bench(benchOptions{
		N: 400, Dim: 8, K: 5, Queries: 20, M: 8, EfConstruction: 32,
		Clusters: 4, Spread: 0.3, Seed: 1, Efs: []int{8, 64}, Threads: 2,
		MinDuration: 10 * time.Millisecond,
	}, &out)
	if err != nil {
		t.Fatal(err)
	}
	s := out.String()
	for _, want := range []string{"| Build time |", "| Index size |", "| Brute-force QPS", "| ef | recall@5 |", "| 8 |", "| 64 |"} {
		if !strings.Contains(s, want) {
			t.Fatalf("output lacks %q:\n%s", want, s)
		}
	}
	// With ef covering a sixth of this tiny index recall must be near 1.
	m := regexp.MustCompile(`\| 64 \| ([0-9.]+) \|`).FindStringSubmatch(s)
	if m == nil {
		t.Fatalf("no ef=64 row:\n%s", s)
	}
	if r, _ := strconv.ParseFloat(m[1], 64); r < 0.95 {
		t.Fatalf("recall at ef=64 = %s, want >= 0.95:\n%s", m[1], s)
	}
}

func TestBenchRejectsBadOptions(t *testing.T) {
	var out bytes.Buffer
	if err := bench(benchOptions{N: 0, Queries: 1, K: 1, Threads: 1, Dim: 2, Clusters: 1}, &out); err == nil {
		t.Fatal("expected an error for n=0")
	}
	if err := bench(benchOptions{N: 10, Queries: 1, K: 1, Threads: 1, Dim: 2, Clusters: 1, M: 1}, &out); err == nil {
		t.Fatal("expected an error for M=1")
	}
	if err := bench(benchOptions{N: 10, Queries: 1, K: 1, Threads: 1, Dim: 2, Clusters: 1, Metric: hnsw.Metric(7)}, &out); err == nil {
		t.Fatal("expected an error for a bad metric")
	}
}

func TestMeasureQPSCountsEveryCall(t *testing.T) {
	var calls [5]int
	qps := measureQPS(0, 1, 5, func(i int) { calls[i]++ })
	for i, c := range calls {
		if c != 1 {
			t.Fatalf("query %d ran %d times, want 1 (minDur 0 = one pass)", i, c)
		}
	}
	if qps <= 0 {
		t.Fatalf("qps = %g", qps)
	}
}
