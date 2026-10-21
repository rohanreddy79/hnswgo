// Index construction (Algorithm 1), query (Algorithm 5), deletion and the
// public API. Many goroutines may Search concurrently; Add and Delete are
// serialised behind a write lock.
package hnsw

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
)

// Errors returned by the index.
var (
	ErrDuplicateID = errors.New("hnsw: id already present")
	ErrDimension   = errors.New("hnsw: vector dimension mismatch")
	ErrConfig      = errors.New("hnsw: invalid config")
	ErrFull        = errors.New("hnsw: index is full")
)

const (
	defaultM              = 16
	defaultEfConstruction = 200
	// maxLevel caps the random level. With M >= 2 the probability of ever
	// drawing a level this high is below 2^-32; the cap just bounds slices.
	maxLevel = 32
	// maxNodes keeps internal indices in uint32, which halves adjacency
	// memory versus int. ^uint32(0) is reserved as a "none" marker.
	maxNodes = math.MaxUint32 - 1
)

// Config describes an index. Zero values take the documented defaults.
type Config struct {
	// Dim is the vector dimension. Required.
	Dim int
	// M is the number of edges a node gets on insertion and the degree cap
	// on layers above 0 (layer 0 allows 2M). Default 16.
	M int
	// EfConstruction is the beam width used while inserting. Default 200.
	EfConstruction int
	// Metric is the distance function. Default L2.
	Metric Metric
	// Seed fixes the level generator so builds are reproducible.
	Seed uint64
	// Select is the neighbour-selection rule. Default SelectHeuristic.
	Select Selection
	// ExtendCandidates widens the heuristic's candidate pool with the
	// candidates' own neighbours (paper: extendCandidates).
	ExtendCandidates bool
	// KeepPruned tops up a node's neighbour list with rejected candidates
	// when the heuristic kept fewer than M (paper: keepPrunedConnections).
	KeepPruned bool
}

func (c Config) withDefaults() (Config, error) {
	if c.Dim <= 0 {
		return c, fmt.Errorf("%w: Dim must be positive", ErrConfig)
	}
	if c.M == 0 {
		c.M = defaultM
	}
	if c.M < 2 {
		return c, fmt.Errorf("%w: M must be at least 2", ErrConfig)
	}
	if c.EfConstruction == 0 {
		c.EfConstruction = defaultEfConstruction
	}
	if c.EfConstruction < 1 {
		return c, fmt.Errorf("%w: EfConstruction must be positive", ErrConfig)
	}
	if !c.Metric.valid() {
		return c, fmt.Errorf("%w: unknown metric", ErrConfig)
	}
	if !c.Select.valid() {
		return c, fmt.Errorf("%w: unknown selection", ErrConfig)
	}
	return c, nil
}

// Result is one search hit. Distance is in the index's metric, so for L2 it
// is squared Euclidean distance and for Dot it is the negated inner product.
type Result struct {
	ID       uint64  `json:"id"`
	Distance float32 `json:"distance"`
}

// node is one vertex of the graph. Its vector lives in Index.vecs at the
// node's position, keeping the hot per-node data compact.
type node struct {
	id      uint64
	deleted bool
	// neighbors[l] is the adjacency list on layer l; len(neighbors) is the
	// node's level + 1.
	neighbors [][]uint32
}

// Index is an HNSW graph. It is safe for concurrent use.
type Index struct {
	cfg  Config
	mL   float64
	dist func(a, b []float32) float32

	mu       sync.RWMutex
	vecs     []float32 // node i occupies vecs[i*dim : (i+1)*dim]
	nodes    []node
	byID     map[uint64]uint32 // live ids only; tombstones are unmapped
	entry    uint32
	maxLevel int
	live     int
	src      *rand.PCG // kept separately: it is what Save/Load serialise
	rng      *rand.Rand

	pool sync.Pool
}

// New creates an empty index.
func New(cfg Config) (*Index, error) {
	cfg, err := cfg.withDefaults()
	if err != nil {
		return nil, err
	}
	x := &Index{
		cfg:  cfg,
		mL:   1 / math.Log(float64(cfg.M)),
		dist: cfg.Metric.kernel(),
		byID: make(map[uint64]uint32),
	}
	x.src = rand.NewPCG(cfg.Seed, cfg.Seed^0x9E3779B97F4A7C15)
	x.rng = rand.New(x.src)
	x.pool.New = func() any { return &searcher{} }
	return x, nil
}

// Config returns the index configuration with defaults applied.
func (x *Index) Config() Config { return x.cfg }

func (x *Index) vec(i uint32) []float32 {
	d := x.cfg.Dim
	return x.vecs[int(i)*d : int(i)*d+d : int(i)*d+d]
}

// mmax is the degree cap on layer l: 2M on the base layer, M above it.
func (x *Index) mmax(l int) int {
	if l == 0 {
		return 2 * x.cfg.M
	}
	return x.cfg.M
}

// randomLevel draws floor(-ln(U) * mL): a geometric distribution whose
// expected size shrinks by 1/M per layer, giving O(log n) layers.
func (x *Index) randomLevel() int {
	u := 1 - x.rng.Float64() // (0, 1], so the log is finite
	l := int(math.Floor(-math.Log(u) * x.mL))
	if l > maxLevel {
		l = maxLevel
	}
	return l
}

func (x *Index) acquire() *searcher { return x.pool.Get().(*searcher) }

func (x *Index) release(s *searcher) { x.pool.Put(s) }

// Len returns the number of live (non-deleted) vectors.
func (x *Index) Len() int {
	x.mu.RLock()
	defer x.mu.RUnlock()
	return x.live
}

// Contains reports whether id is present and not deleted.
func (x *Index) Contains(id uint64) bool {
	x.mu.RLock()
	defer x.mu.RUnlock()
	_, ok := x.byID[id]
	return ok
}

// Add inserts vec under id. It returns ErrDuplicateID if a live vector with
// that id exists; a deleted id may be reused. vec is copied (and normalised
// for Cosine), so the caller may reuse it.
func (x *Index) Add(id uint64, vec []float32) error {
	if len(vec) != x.cfg.Dim {
		return fmt.Errorf("%w: got %d, want %d", ErrDimension, len(vec), x.cfg.Dim)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if _, ok := x.byID[id]; ok {
		return fmt.Errorf("%w: %d", ErrDuplicateID, id)
	}
	if len(x.nodes) >= maxNodes {
		return ErrFull
	}

	level := x.randomLevel()
	idx := uint32(len(x.nodes))
	x.vecs = append(x.vecs, vec...)
	q := x.vec(idx)
	x.cfg.Metric.prepare(q)
	nbrs := make([][]uint32, level+1)
	for l := range nbrs {
		nbrs[l] = make([]uint32, 0, x.mmax(l))
	}
	x.nodes = append(x.nodes, node{id: id, neighbors: nbrs})
	x.byID[id] = idx
	x.live++

	if len(x.nodes) == 1 {
		x.entry, x.maxLevel = idx, level
		return nil
	}

	s := x.acquire()
	defer x.release(s)
	ep := x.entry
	for l := x.maxLevel; l > level; l-- {
		ep = x.greedy(q, ep, l)
	}
	for l := min(level, x.maxLevel); l >= 0; l-- {
		// Tombstones are included so a new node can still link into a
		// region whose live nodes were all deleted.
		x.searchLayer(s, q, ep, x.cfg.EfConstruction, l, true)
		ep = s.sorted[0].idx
		chosen := x.selectNeighbors(s, q, s.sorted, x.cfg.M, l, x.cfg.ExtendCandidates, nbrs[l][:0])
		nbrs[l] = chosen
		for _, nb := range chosen {
			x.connect(s, nb, idx, l)
		}
	}
	if level > x.maxLevel {
		x.entry, x.maxLevel = idx, level
	}
	return nil
}

// connect adds the back-edge e -> n on layer l and, if e now exceeds its
// degree cap, re-selects e's neighbours from the enlarged list.
func (x *Index) connect(s *searcher, e, n uint32, l int) {
	list := &x.nodes[e].neighbors[l]
	*list = append(*list, n)
	limit := x.mmax(l)
	if len(*list) <= limit {
		return
	}
	ev := x.vec(e)
	s.extra = s.extra[:0]
	for _, nb := range *list {
		s.extra = append(s.extra, item{x.dist(ev, x.vec(nb)), nb})
	}
	sortItems(s.extra)
	// extend is false here: the pool is already e's full adjacency, and
	// widening it would cost O(M^2) distances per insert for little gain.
	*list = x.selectNeighbors(s, ev, s.extra, limit, l, false, (*list)[:0])
}

// Delete tombstones id. The node stays in the graph (so searches can still
// route through it) but is never returned. It reports whether id was live.
func (x *Index) Delete(id uint64) bool {
	x.mu.Lock()
	defer x.mu.Unlock()
	idx, ok := x.byID[id]
	if !ok {
		return false
	}
	delete(x.byID, id)
	x.nodes[idx].deleted = true
	x.live--
	return true
}

// Search returns up to k nearest live vectors to q, nearest first. ef is the
// beam width on the base layer and is raised to k if smaller; larger ef
// trades speed for recall. It returns nil when the index is empty and
// panics if len(q) != Dim, which is a programming error like an
// out-of-range slice.
func (x *Index) Search(q []float32, k, ef int) []Result {
	if len(q) != x.cfg.Dim {
		panic(fmt.Sprintf("hnsw: Search: query has %d dimensions, index has %d", len(q), x.cfg.Dim))
	}
	if k <= 0 {
		return nil
	}
	if ef < k {
		ef = k
	}
	x.mu.RLock()
	defer x.mu.RUnlock()
	if x.live == 0 {
		return nil
	}
	s := x.acquire()
	defer x.release(s)
	if x.cfg.Metric == Cosine {
		s.query = append(s.query[:0], q...)
		x.cfg.Metric.prepare(s.query)
		q = s.query
	}
	ep := x.entry
	for l := x.maxLevel; l > 0; l-- {
		ep = x.greedy(q, ep, l)
	}
	x.searchLayer(s, q, ep, ef, 0, false)
	n := min(k, len(s.sorted))
	out := make([]Result, n)
	for i := range out {
		it := s.sorted[i]
		out[i] = Result{ID: x.nodes[it.idx].id, Distance: it.dist}
	}
	return out
}

// Stats is a snapshot of index shape and size.
type Stats struct {
	Dim            int    `json:"dim"`
	M              int    `json:"m"`
	EfConstruction int    `json:"ef_construction"`
	Metric         string `json:"metric"`
	Selection      string `json:"selection"`
	Nodes          int    `json:"nodes"`   // live + tombstoned
	Live           int    `json:"live"`    // what Len reports
	Deleted        int    `json:"deleted"` // tombstones still in the graph
	MaxLevel       int    `json:"max_level"`
	// LevelCounts[l] is the number of nodes whose top layer is l.
	LevelCounts []int `json:"level_counts"`
	// Edges is the total number of directed edges over all layers.
	Edges int `json:"edges"`
	// Bytes approximates heap memory owned by the index: vectors,
	// adjacency lists, node headers and the id map.
	Bytes int64 `json:"bytes"`
}

// Stats returns a snapshot of the index.
func (x *Index) Stats() Stats {
	x.mu.RLock()
	defer x.mu.RUnlock()
	st := Stats{
		Dim:            x.cfg.Dim,
		M:              x.cfg.M,
		EfConstruction: x.cfg.EfConstruction,
		Metric:         x.cfg.Metric.String(),
		Selection:      x.cfg.Select.String(),
		Nodes:          len(x.nodes),
		Live:           x.live,
		Deleted:        len(x.nodes) - x.live,
	}
	if len(x.nodes) > 0 {
		st.MaxLevel = x.maxLevel
		st.LevelCounts = make([]int, x.maxLevel+1)
	}
	const nodeHeader = 40 // id + deleted (padded) + slice header
	const sliceHeader = 24
	const mapEntry = 24 // key + value + bucket overhead, approximate
	bytes := int64(cap(x.vecs))*4 + int64(cap(x.nodes))*nodeHeader + int64(len(x.byID))*mapEntry
	for i := range x.nodes {
		n := &x.nodes[i]
		st.LevelCounts[len(n.neighbors)-1]++
		for _, l := range n.neighbors {
			st.Edges += len(l)
			bytes += sliceHeader + int64(cap(l))*4
		}
	}
	st.Bytes = bytes
	return st
}

func sortItems(items []item) {
	// Insertion sort: lists here are at most 2M+1 long, where this beats
	// the generic sort and allocates nothing.
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].dist < items[j-1].dist; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
}
