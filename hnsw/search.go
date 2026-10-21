// Graph traversal: the greedy descent through the upper layers and the
// ef-bounded beam search of Algorithm 2. All scratch state lives in a
// searcher so a query allocates nothing after warm-up.
package hnsw

import "slices"

// searcher is the per-call scratch space. Searchers are pooled; each is used
// by one goroutine at a time, so the visited set needs no synchronisation.
type searcher struct {
	// visited[i] == gen means node i was seen during the current traversal.
	// Bumping gen clears the whole set in O(1) instead of zeroing the slice
	// or allocating a map per query.
	visited []uint32
	gen     uint32

	cands   minHeap
	results maxHeap
	sorted  []item    // results flattened and ordered nearest-first
	extra   []item    // scratch for extendCandidates
	query   []float32 // normalised copy of the caller's query (cosine only)
}

// reset prepares the visited set for a new traversal over n nodes.
func (s *searcher) reset(n int) {
	if len(s.visited) < n {
		s.visited = slices.Grow(s.visited, n-len(s.visited))[:n]
	}
	s.gen++
	if s.gen == 0 {
		// Wrapped around: stale entries could collide with gen 0, so clear.
		clear(s.visited)
		s.gen = 1
	}
	s.cands = s.cands[:0]
	s.results = s.results[:0]
}

// visit marks idx and reports whether it had already been seen.
func (s *searcher) visit(idx uint32) bool {
	if s.visited[idx] == s.gen {
		return true
	}
	s.visited[idx] = s.gen
	return false
}

// greedy walks layer l from ep to the local minimum nearest q, the upper-layer
// descent of Algorithm 1/5 with ef = 1. No visited set is needed: the walk
// only moves to strictly closer nodes, so it cannot revisit one.
func (x *Index) greedy(q []float32, ep uint32, l int) uint32 {
	cur := ep
	best := x.dist(q, x.vec(cur))
	for changed := true; changed; {
		changed = false
		for _, nb := range x.nodes[cur].neighbors[l] {
			if d := x.dist(q, x.vec(nb)); d < best {
				best, cur, changed = d, nb, true
			}
		}
	}
	return cur
}

// searchLayer is Algorithm 2: an ef-bounded best-first search on layer l
// starting at ep. On return s.sorted holds up to ef items nearest-first.
// Tombstoned nodes are always expanded (they keep the graph connected) and
// are returned only when includeDeleted is set, which construction uses so
// that new nodes can still link through them.
func (x *Index) searchLayer(s *searcher, q []float32, ep uint32, ef, l int, includeDeleted bool) {
	s.reset(len(x.nodes))
	s.visit(ep)
	d := x.dist(q, x.vec(ep))
	s.cands.push(item{d, ep})
	if includeDeleted || !x.nodes[ep].deleted {
		s.results.push(item{d, ep})
	}
	for len(s.cands) > 0 {
		c := s.cands.pop()
		// Stop once the nearest unexpanded candidate cannot improve a full
		// result set. A non-full set keeps expanding, which matters when
		// many nodes near the query are tombstoned.
		if len(s.results) >= ef && c.dist > s.results.top().dist {
			break
		}
		for _, nb := range x.nodes[c.idx].neighbors[l] {
			if s.visit(nb) {
				continue
			}
			d := x.dist(q, x.vec(nb))
			if len(s.results) < ef || d < s.results.top().dist {
				s.cands.push(item{d, nb})
				if includeDeleted || !x.nodes[nb].deleted {
					s.results.push(item{d, nb})
					if len(s.results) > ef {
						s.results.pop()
					}
				}
			}
		}
	}
	s.sorted = append(s.sorted[:0], s.results...)
	slices.SortFunc(s.sorted, func(a, b item) int {
		switch {
		case a.dist < b.dist:
			return -1
		case a.dist > b.dist:
			return 1
		case a.idx < b.idx:
			return -1
		case a.idx > b.idx:
			return 1
		}
		return 0
	})
}
