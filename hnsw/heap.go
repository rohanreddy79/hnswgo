// Binary heaps over (distance, node) pairs. Two concrete types instead of
// container/heap: the interface indirection and per-op allocation of
// container/heap show up clearly in the search hot loop, and an 8-byte item
// keeps both heaps cache-resident.
package hnsw

// item is a node reference tagged with its distance to the current query.
type item struct {
	dist float32
	idx  uint32
}

// minHeap pops the nearest item first; it is the candidate queue of Algorithm 2.
type minHeap []item

func (h *minHeap) push(it item) {
	*h = append(*h, it)
	s := *h
	i := len(s) - 1
	for i > 0 {
		p := (i - 1) / 2
		if s[p].dist <= s[i].dist {
			break
		}
		s[p], s[i] = s[i], s[p]
		i = p
	}
}

func (h *minHeap) pop() item {
	s := *h
	top := s[0]
	last := len(s) - 1
	s[0] = s[last]
	s = s[:last]
	i := 0
	for {
		l := 2*i + 1
		if l >= last {
			break
		}
		m := l
		if r := l + 1; r < last && s[r].dist < s[l].dist {
			m = r
		}
		if s[i].dist <= s[m].dist {
			break
		}
		s[i], s[m] = s[m], s[i]
		i = m
	}
	*h = s
	return top
}

// maxHeap keeps the furthest item on top; it is the bounded result set W of
// Algorithm 2, where the top is the current worst entry and the pruning bound.
type maxHeap []item

func (h *maxHeap) push(it item) {
	*h = append(*h, it)
	s := *h
	i := len(s) - 1
	for i > 0 {
		p := (i - 1) / 2
		if s[p].dist >= s[i].dist {
			break
		}
		s[p], s[i] = s[i], s[p]
		i = p
	}
}

func (h *maxHeap) pop() item {
	s := *h
	top := s[0]
	last := len(s) - 1
	s[0] = s[last]
	s = s[:last]
	i := 0
	for {
		l := 2*i + 1
		if l >= last {
			break
		}
		m := l
		if r := l + 1; r < last && s[r].dist > s[l].dist {
			m = r
		}
		if s[i].dist >= s[m].dist {
			break
		}
		s[i], s[m] = s[m], s[i]
		i = m
	}
	*h = s
	return top
}

// top returns the furthest item without removing it. The caller checks
// emptiness.
func (h maxHeap) top() item { return h[0] }
