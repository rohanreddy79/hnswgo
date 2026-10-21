// Package hnsw implements the Hierarchical Navigable Small World graph
// (Malkov & Yashunin, 2016) for approximate nearest-neighbour search over
// float32 vectors. This file holds the distance metrics.
package hnsw

import (
	"fmt"
	"math"
)

// Metric selects how distances between vectors are computed. Smaller is
// always "closer" so that one search routine serves every metric.
type Metric uint8

const (
	// L2 is squared Euclidean distance. The square root is skipped because it
	// does not change the ordering and costs a sqrt per comparison.
	L2 Metric = iota
	// Cosine is 1 - cos(a, b), in [0, 2]. Vectors are normalised to unit
	// length when inserted so each comparison is a single dot product.
	Cosine
	// Dot is the negated inner product -a·b, so higher similarity sorts first.
	Dot
)

// String returns the lower-case metric name used by the CLI and file format.
func (m Metric) String() string {
	switch m {
	case L2:
		return "l2"
	case Cosine:
		return "cosine"
	case Dot:
		return "dot"
	default:
		return fmt.Sprintf("metric(%d)", uint8(m))
	}
}

// ParseMetric is the inverse of String.
func ParseMetric(s string) (Metric, error) {
	switch s {
	case "l2":
		return L2, nil
	case "cosine":
		return Cosine, nil
	case "dot":
		return Dot, nil
	default:
		return 0, fmt.Errorf("hnsw: unknown metric %q (want l2, cosine or dot)", s)
	}
}

func (m Metric) valid() bool { return m <= Dot }

// Distance computes the metric between two raw (un-normalised) vectors of the
// same length. It is the reference definition: the index itself pre-normalises
// cosine vectors and uses the cheaper internal kernels, which agree with this
// function up to floating-point rounding.
func (m Metric) Distance(a, b []float32) float32 {
	switch m {
	case L2:
		return l2Squared(a, b)
	case Cosine:
		dot, na, nb := dotAndNorms(a, b)
		if na == 0 || nb == 0 {
			return 1
		}
		return 1 - dot/float32(math.Sqrt(float64(na)*float64(nb)))
	case Dot:
		return -dotProduct(a, b)
	default:
		panic("hnsw: invalid metric")
	}
}

// kernel returns the distance function applied to stored (prepared) vectors.
func (m Metric) kernel() func(a, b []float32) float32 {
	switch m {
	case L2:
		return l2Squared
	case Cosine:
		return cosineUnit
	case Dot:
		return negDot
	default:
		panic("hnsw: invalid metric")
	}
}

// prepare rewrites v in place into the form the kernel expects. Only cosine
// needs work: unit-length vectors make 1 - a·b the cosine distance. A zero
// vector cannot be normalised and is left as-is, so it sits at distance 1
// from everything.
func (m Metric) prepare(v []float32) {
	if m != Cosine {
		return
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

func cosineUnit(a, b []float32) float32 { return 1 - dotProduct(a, b) }

func negDot(a, b []float32) float32 { return -dotProduct(a, b) }

// The kernels below are unrolled four-wide with independent accumulators so
// the CPU can overlap the floating-point adds; the compiler does not
// auto-vectorise, but this alone roughly doubles throughput over a naive loop.

func dotProduct(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func l2Squared(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}

// dotAndNorms returns a·b, a·a and b·b in one pass for the reference cosine.
func dotAndNorms(a, b []float32) (dot, na, nb float32) {
	b = b[:len(a)]
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	return dot, na, nb
}
