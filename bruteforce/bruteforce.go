// Package bruteforce is exact k-nearest-neighbour search by full scan. It
// exists to measure the approximate index: its answers are the ground truth
// for recall, and its throughput is the baseline HNSW must beat.
package bruteforce

import (
	"fmt"
	"sync"

	"github.com/rohanreddy79/hnswgo/hnsw"
)

// Index stores vectors flat and scans all of them per query.
type Index struct {
	dim    int
	metric hnsw.Metric

	mu   sync.RWMutex
	ids  []uint64
	vecs []float32
}

// New creates an empty exact index.
func New(dim int, metric hnsw.Metric) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("%w: Dim must be positive", hnsw.ErrConfig)
	}
	return &Index{dim: dim, metric: metric}, nil
}

// Add appends a vector. Ids are not checked for uniqueness: this index is a
// measurement tool and mirrors whatever the caller fed the HNSW index.
func (x *Index) Add(id uint64, vec []float32) error {
	if len(vec) != x.dim {
		return fmt.Errorf("%w: got %d, want %d", hnsw.ErrDimension, len(vec), x.dim)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	x.ids = append(x.ids, id)
	x.vecs = append(x.vecs, vec...)
	return nil
}

// Len returns the number of stored vectors.
func (x *Index) Len() int {
	x.mu.RLock()
	defer x.mu.RUnlock()
	return len(x.ids)
}

// Search returns the k exact nearest vectors, nearest first. Ties are broken
// by insertion order so results are deterministic.
func (x *Index) Search(q []float32, k int) []hnsw.Result {
	if len(q) != x.dim {
		panic(fmt.Sprintf("bruteforce: Search: query has %d dimensions, index has %d", len(q), x.dim))
	}
	x.mu.RLock()
	defer x.mu.RUnlock()
	if k <= 0 || len(x.ids) == 0 {
		return nil
	}
	k = min(k, len(x.ids))
	// A sorted window of size k; insertion is O(k) per candidate that
	// beats the current worst, which for small k beats heap bookkeeping.
	top := make([]hnsw.Result, 0, k)
	for i, id := range x.ids {
		d := x.metric.Distance(q, x.vecs[i*x.dim:(i+1)*x.dim])
		if len(top) == k && d >= top[k-1].Distance {
			continue
		}
		pos := len(top)
		if len(top) < k {
			top = append(top, hnsw.Result{})
		}
		for pos > 0 && top[pos-1].Distance > d {
			if pos < len(top) {
				top[pos] = top[pos-1]
			}
			pos--
		}
		top[pos] = hnsw.Result{ID: id, Distance: d}
	}
	return top
}

// Recall returns |got ∩ want| / |want|: the fraction of the true neighbours
// the approximate search found. It is 1 when want is empty.
func Recall(got, want []hnsw.Result) float64 {
	if len(want) == 0 {
		return 1
	}
	set := make(map[uint64]struct{}, len(got))
	for _, r := range got {
		set[r.ID] = struct{}{}
	}
	hits := 0
	for _, r := range want {
		if _, ok := set[r.ID]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(want))
}
