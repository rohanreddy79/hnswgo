// Package synth generates seeded synthetic vectors for tests and benchmarks.
// Points are drawn from a mixture of isotropic Gaussian blobs: clustered data
// is the realistic case for embeddings and the hard case for graph search,
// because a greedy walk can settle in the wrong cluster.
package synth

import "math/rand/v2"

// Blobs is a fixed set of Gaussian cluster centres plus a seeded generator.
// Sample draws points from the mixture, so queries and base vectors drawn
// from the same Blobs share a distribution.
type Blobs struct {
	dim     int
	spread  float64
	centres [][]float32
	rng     *rand.Rand
}

// NewBlobs creates clusters centres with coordinates ~ N(0, 1). Each sampled
// point is a centre plus N(0, spread^2) noise per coordinate. With spread
// well below 1 the clusters are distinct; near 1 they overlap heavily.
func NewBlobs(seed uint64, dim, clusters int, spread float64) *Blobs {
	if dim <= 0 || clusters <= 0 {
		panic("synth: dim and clusters must be positive")
	}
	b := &Blobs{
		dim:     dim,
		spread:  spread,
		centres: make([][]float32, clusters),
		rng:     rand.New(rand.NewPCG(seed, seed^0xD1B54A32D192ED03)),
	}
	for i := range b.centres {
		c := make([]float32, dim)
		for j := range c {
			c[j] = float32(b.rng.NormFloat64())
		}
		b.centres[i] = c
	}
	return b
}

// Dim returns the vector dimension.
func (b *Blobs) Dim() int { return b.dim }

// Sample draws n points, choosing a cluster uniformly for each.
func (b *Blobs) Sample(n int) [][]float32 {
	out := make([][]float32, n)
	flat := make([]float32, n*b.dim) // one allocation keeps the data contiguous
	for i := range out {
		c := b.centres[b.rng.IntN(len(b.centres))]
		v := flat[i*b.dim : (i+1)*b.dim : (i+1)*b.dim]
		for j := range v {
			v[j] = c[j] + float32(b.rng.NormFloat64()*b.spread)
		}
		out[i] = v
	}
	return out
}

// Uniform draws n points with coordinates ~ U(-1, 1): structureless data
// used where clustering would only get in the way (metric tests, fuzzing).
func Uniform(seed uint64, n, dim int) [][]float32 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	out := make([][]float32, n)
	flat := make([]float32, n*dim)
	for i := range out {
		v := flat[i*dim : (i+1)*dim : (i+1)*dim]
		for j := range v {
			v[j] = float32(rng.Float64()*2 - 1)
		}
		out[i] = v
	}
	return out
}
