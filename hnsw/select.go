// Neighbour selection: which of the candidates found during insertion become
// graph edges. This decides the graph's navigability far more than ef does.
package hnsw

import "slices"

// Selection chooses the neighbour-selection rule used while building the graph.
type Selection uint8

const (
	// SelectHeuristic is Algorithm 4 of the paper: a candidate is kept only if
	// it is closer to the query than to every neighbour already kept. This
	// favours edges in distinct directions, so clustered data stays
	// connected across clusters instead of forming isolated cliques.
	SelectHeuristic Selection = iota
	// SelectSimple keeps the M nearest candidates (Algorithm 3). Cheaper to
	// build, measurably worse recall on clustered data; kept for comparison.
	SelectSimple
)

// String returns the lower-case name used by the CLI.
func (s Selection) String() string {
	switch s {
	case SelectHeuristic:
		return "heuristic"
	case SelectSimple:
		return "simple"
	default:
		return "selection(?)"
	}
}

// ParseSelection is the inverse of String.
func ParseSelection(s string) (Selection, error) {
	switch s {
	case "heuristic":
		return SelectHeuristic, nil
	case "simple":
		return SelectSimple, nil
	default:
		return 0, errInvalidSelection(s)
	}
}

type errInvalidSelection string

func (e errInvalidSelection) Error() string {
	return "hnsw: unknown selection " + string(e) + " (want heuristic or simple)"
}

func (s Selection) valid() bool { return s <= SelectSimple }

// selectNeighbors picks at most m nodes from cands (sorted nearest-first to
// q) and appends them to dst. extend adds the candidates' own layer-l
// neighbours to the pool first (extendCandidates in the paper); it is only
// used for the new node, where the pool came from a bounded search and may
// have missed good edges.
func (x *Index) selectNeighbors(s *searcher, q []float32, cands []item, m, l int, extend bool, dst []uint32) []uint32 {
	if x.cfg.Select == SelectSimple {
		for i := 0; i < len(cands) && i < m; i++ {
			dst = append(dst, cands[i].idx)
		}
		return dst
	}
	if extend {
		cands = x.extendCandidates(s, q, cands, l)
	}
	return x.heuristic(q, cands, m, dst)
}

// extendCandidates returns cands plus their layer-l neighbours, deduplicated
// through the visited set and re-sorted by distance to q.
func (x *Index) extendCandidates(s *searcher, q []float32, cands []item, l int) []item {
	s.reset(len(x.nodes))
	s.extra = append(s.extra[:0], cands...)
	for _, c := range cands {
		s.visit(c.idx)
	}
	for _, c := range cands {
		for _, nb := range x.nodes[c.idx].neighbors[l] {
			if s.visit(nb) {
				continue
			}
			s.extra = append(s.extra, item{x.dist(q, x.vec(nb)), nb})
		}
	}
	slices.SortFunc(s.extra, func(a, b item) int {
		if a.dist < b.dist {
			return -1
		}
		if a.dist > b.dist {
			return 1
		}
		return 0
	})
	return s.extra
}

// heuristic is the core of Algorithm 4. cands must be sorted nearest-first.
func (x *Index) heuristic(q []float32, cands []item, m int, dst []uint32) []uint32 {
	start := len(dst)
	var pruned []item
	for _, c := range cands {
		if len(dst)-start >= m {
			break
		}
		cv := x.vec(c.idx)
		good := true
		for _, r := range dst[start:] {
			// c is shadowed by r when r is closer to c than q is: an edge
			// q->c would be redundant because q->r->c already covers it.
			if x.dist(cv, x.vec(r)) < c.dist {
				good = false
				break
			}
		}
		if good {
			dst = append(dst, c.idx)
		} else if x.cfg.KeepPruned {
			pruned = append(pruned, c)
		}
	}
	// keepPrunedConnections: top up to m with the nearest rejected
	// candidates so low-degree nodes still get enough edges.
	for i := 0; i < len(pruned) && len(dst)-start < m; i++ {
		dst = append(dst, pruned[i].idx)
	}
	return dst
}
