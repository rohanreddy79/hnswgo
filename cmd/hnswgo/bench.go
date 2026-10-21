// The benchmark: build on seeded clustered data, then report recall@k and
// throughput for several ef values against exact search.
package main

import (
	"flag"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rohanreddy79/hnswgo/bruteforce"
	"github.com/rohanreddy79/hnswgo/hnsw"
	"github.com/rohanreddy79/hnswgo/internal/synth"
)

type benchOptions struct {
	N, Dim, K, Queries int
	M, EfConstruction  int
	Clusters           int
	Spread             float64
	Seed               uint64
	Metric             hnsw.Metric
	Select             hnsw.Selection
	Efs                []int
	Threads            int
	// MinDuration is how long each timed loop runs at minimum; short query
	// sets are repeated until it elapses so QPS is not timer noise.
	MinDuration time.Duration
}

func runBench(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	o := benchOptions{MinDuration: time.Second}
	fs.IntVar(&o.N, "n", 20000, "number of base vectors")
	fs.IntVar(&o.Dim, "dim", 64, "vector dimension")
	fs.IntVar(&o.K, "k", 10, "neighbours per query")
	fs.IntVar(&o.Queries, "queries", 500, "number of queries")
	fs.IntVar(&o.M, "m", 16, "HNSW M")
	fs.IntVar(&o.EfConstruction, "ef-construction", 200, "beam width while inserting")
	fs.IntVar(&o.Clusters, "clusters", 50, "Gaussian clusters in the synthetic data")
	fs.Float64Var(&o.Spread, "spread", 0.5, "per-coordinate std-dev within a cluster")
	seed := fs.Uint64("seed", 1, "seed for data and level generation")
	metric := fs.String("metric", "l2", "distance metric: l2, cosine or dot")
	sel := fs.String("select", "heuristic", "neighbour selection: heuristic or simple")
	efs := fs.String("ef", "16,32,64,128", "comma-separated ef values to evaluate")
	fs.IntVar(&o.Threads, "threads", runtime.GOMAXPROCS(0), "goroutines for the parallel QPS column")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	o.Seed = *seed
	var err error
	if o.Metric, err = hnsw.ParseMetric(*metric); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	if o.Select, err = hnsw.ParseSelection(*sel); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	for _, s := range strings.Split(*efs, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || v <= 0 {
			fmt.Fprintf(stderr, "bench: invalid --ef value %q\n", s)
			return 2
		}
		o.Efs = append(o.Efs, v)
	}
	if err := bench(o, stdout); err != nil {
		fmt.Fprintln(stderr, "bench:", err)
		return 1
	}
	return 0
}

func bench(o benchOptions, w io.Writer) error {
	if o.N <= 0 || o.Queries <= 0 || o.K <= 0 || o.Threads <= 0 {
		return fmt.Errorf("n, queries, k and threads must be positive")
	}
	blobs := synth.NewBlobs(o.Seed, o.Dim, o.Clusters, o.Spread)
	base := blobs.Sample(o.N)
	queries := blobs.Sample(o.Queries)

	idx, err := hnsw.New(hnsw.Config{
		Dim: o.Dim, M: o.M, EfConstruction: o.EfConstruction,
		Metric: o.Metric, Seed: o.Seed, Select: o.Select,
	})
	if err != nil {
		return err
	}
	exact, err := bruteforce.New(o.Dim, o.Metric)
	if err != nil {
		return err
	}
	start := time.Now()
	for i, v := range base {
		if err := idx.Add(uint64(i), v); err != nil {
			return err
		}
	}
	buildTime := time.Since(start)
	for i, v := range base {
		if err := exact.Add(uint64(i), v); err != nil {
			return err
		}
	}
	cw := &countingWriter{w: io.Discard}
	if err := idx.Save(cw); err != nil {
		return err
	}
	st := idx.Stats()

	truth := make([][]hnsw.Result, len(queries))
	bruteQPS := measureQPS(o.MinDuration, 1, len(queries), func(i int) {
		r := exact.Search(queries[i], o.K)
		if truth[i] == nil {
			truth[i] = r
		}
	})

	fmt.Fprintf(w, "hnswgo bench: %s %s/%s, %d CPUs\n\n", runtime.Version(), runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Fprintf(w, "| Parameter | Value |\n|---|---|\n")
	fmt.Fprintf(w, "| Vectors | %d x %d (%d Gaussian clusters, spread %.2f, seed %d) |\n", o.N, o.Dim, o.Clusters, o.Spread, o.Seed)
	fmt.Fprintf(w, "| Metric | %s |\n", o.Metric)
	fmt.Fprintf(w, "| M / efConstruction / selection | %d / %d / %s |\n", o.M, o.EfConstruction, o.Select)
	fmt.Fprintf(w, "| Build time | %s (%.0f inserts/s) |\n", buildTime.Round(time.Millisecond), float64(o.N)/buildTime.Seconds())
	fmt.Fprintf(w, "| Graph | %d edges, max level %d, level counts %v |\n", st.Edges, st.MaxLevel, st.LevelCounts)
	fmt.Fprintf(w, "| Index size | %d bytes serialised, ~%d bytes in memory |\n", cw.n, st.Bytes)
	fmt.Fprintf(w, "| Brute-force QPS (1 thread) | %.0f |\n", bruteQPS)
	fmt.Fprintf(w, "| Queries | %d, k = %d |\n\n", o.Queries, o.K)

	fmt.Fprintf(w, "| ef | recall@%d | QPS (1 thread) | QPS (%d threads) | speedup vs brute force (1 thread) |\n|---|---|---|---|---|\n", o.K, o.Threads)
	for _, ef := range o.Efs {
		var sum float64
		for i, q := range queries {
			sum += bruteforce.Recall(idx.Search(q, o.K, ef), truth[i])
		}
		recall := sum / float64(len(queries))
		single := measureQPS(o.MinDuration, 1, len(queries), func(i int) { idx.Search(queries[i], o.K, ef) })
		multi := measureQPS(o.MinDuration, o.Threads, len(queries), func(i int) { idx.Search(queries[i], o.K, ef) })
		fmt.Fprintf(w, "| %d | %.3f | %.0f | %.0f | %.0fx |\n", ef, recall, single, multi, single/bruteQPS)
	}
	return nil
}

// measureQPS runs fn over query indices 0..n-1 round-robin from `threads`
// goroutines until at least minDur has elapsed (and at least one full
// pass), and returns completed calls per second.
func measureQPS(minDur time.Duration, threads, n int, fn func(i int)) float64 {
	var next atomic.Int64
	var done atomic.Int64
	deadline := time.Now().Add(minDur)
	var wg sync.WaitGroup
	start := time.Now()
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := next.Add(1) - 1
				if i >= int64(n) && time.Now().After(deadline) {
					return
				}
				fn(int(i % int64(n)))
				done.Add(1)
			}
		}()
	}
	wg.Wait()
	return float64(done.Load()) / time.Since(start).Seconds()
}
