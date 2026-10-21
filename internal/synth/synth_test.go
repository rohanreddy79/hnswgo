package synth

import (
	"math"
	"slices"
	"testing"
)

func TestBlobsDeterministic(t *testing.T) {
	a := NewBlobs(5, 8, 3, 0.2).Sample(50)
	b := NewBlobs(5, 8, 3, 0.2).Sample(50)
	for i := range a {
		if !slices.Equal(a[i], b[i]) {
			t.Fatalf("sample %d differs between equal seeds", i)
		}
	}
	c := NewBlobs(6, 8, 3, 0.2).Sample(50)
	if slices.Equal(a[0], c[0]) {
		t.Fatal("different seeds produced the same first sample")
	}
}

func TestBlobsShape(t *testing.T) {
	b := NewBlobs(1, 16, 4, 0.1)
	if b.Dim() != 16 {
		t.Fatalf("Dim = %d", b.Dim())
	}
	pts := b.Sample(1000)
	if len(pts) != 1000 || len(pts[0]) != 16 {
		t.Fatalf("shape %d x %d", len(pts), len(pts[0]))
	}
	// With a tight spread every point sits near one of the four centres, so
	// the nearest centre is much closer than any other.
	for _, p := range pts {
		var best, second float64 = math.Inf(1), math.Inf(1)
		for _, c := range b.centres {
			var d float64
			for j := range p {
				diff := float64(p[j] - c[j])
				d += diff * diff
			}
			if d < best {
				best, second = d, best
			} else if d < second {
				second = d
			}
		}
		if best*4 > second {
			t.Fatalf("point is not clearly clustered: nearest %.2f, next %.2f", best, second)
		}
	}
}

func TestUniformRange(t *testing.T) {
	pts := Uniform(3, 200, 5)
	if len(pts) != 200 {
		t.Fatalf("len = %d", len(pts))
	}
	for _, p := range pts {
		for _, x := range p {
			if x < -1 || x >= 1 {
				t.Fatalf("value %g outside [-1, 1)", x)
			}
		}
	}
	if !slices.Equal(pts[7], Uniform(3, 200, 5)[7]) {
		t.Fatal("Uniform is not deterministic")
	}
}
