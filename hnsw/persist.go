// Save and Load: a versioned little-endian binary format with a CRC-32
// trailer. Layout (all integers little-endian):
//
//	magic "HNSW" | version u32 | dim u32 | M u32 | efConstruction u32
//	metric u8 | selection u8 | extendCandidates u8 | keepPruned u8 | seed u64
//	rngState: len u32 + bytes          (so Adds after Load draw the same levels)
//	nodes u64 | entry u32 | maxLevel u32 | live u64
//	per node: id u64 | deleted u8 | level u32 | per layer: count u32, idx u32...
//	vectors: nodes*dim float32 bits
//	crc32 u32 (IEEE) over everything above
package hnsw

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"math"
	"math/rand/v2"
)

var (
	ErrCorrupt = errors.New("hnsw: corrupt index data")
	ErrVersion = errors.New("hnsw: unsupported index file version")
)

var magic = [4]byte{'H', 'N', 'S', 'W'}

const formatVersion = 1

// Save writes the index to w. It holds the read lock, so searches continue
// while a snapshot is taken but writes wait.
func (x *Index) Save(w io.Writer) error {
	x.mu.RLock()
	defer x.mu.RUnlock()

	h := crc32.NewIEEE()
	bw := bufio.NewWriter(io.MultiWriter(w, h))
	e := encoder{w: bw}

	e.bytes(magic[:])
	e.u32(formatVersion)
	e.u32(uint32(x.cfg.Dim))
	e.u32(uint32(x.cfg.M))
	e.u32(uint32(x.cfg.EfConstruction))
	e.u8(uint8(x.cfg.Metric))
	e.u8(uint8(x.cfg.Select))
	e.bool(x.cfg.ExtendCandidates)
	e.bool(x.cfg.KeepPruned)
	e.u64(x.cfg.Seed)
	state, err := x.src.MarshalBinary()
	if err != nil {
		return fmt.Errorf("hnsw: save rng state: %w", err)
	}
	e.u32(uint32(len(state)))
	e.bytes(state)

	e.u64(uint64(len(x.nodes)))
	e.u32(x.entry)
	e.u32(uint32(x.maxLevel))
	e.u64(uint64(x.live))
	for i := range x.nodes {
		n := &x.nodes[i]
		e.u64(n.id)
		e.bool(n.deleted)
		e.u32(uint32(len(n.neighbors) - 1))
		for _, list := range n.neighbors {
			e.u32(uint32(len(list)))
			for _, nb := range list {
				e.u32(nb)
			}
		}
	}
	e.floats(x.vecs)
	if e.err != nil {
		return fmt.Errorf("hnsw: save: %w", e.err)
	}
	if err := bw.Flush(); err != nil {
		return fmt.Errorf("hnsw: save: %w", err)
	}
	// The checksum covers the buffered body, so flush before reading h.
	var sum [4]byte
	binary.LittleEndian.PutUint32(sum[:], h.Sum32())
	if _, err := w.Write(sum[:]); err != nil {
		return fmt.Errorf("hnsw: save: %w", err)
	}
	return nil
}

// Load reads an index written by Save. Every count and index is validated
// before use, and the trailing checksum must match, so a truncated or
// corrupted file yields an error rather than a malformed graph.
func Load(r io.Reader) (*Index, error) {
	d := decoder{r: bufio.NewReader(r), h: crc32.NewIEEE()}

	var m [4]byte
	d.bytes(m[:])
	if d.err == nil && m != magic {
		return nil, fmt.Errorf("%w: bad magic", ErrCorrupt)
	}
	if v := d.u32(); d.err == nil && v != formatVersion {
		return nil, fmt.Errorf("%w: got %d, want %d", ErrVersion, v, formatVersion)
	}
	cfg := Config{
		Dim:            int(d.u32()),
		M:              int(d.u32()),
		EfConstruction: int(d.u32()),
		Metric:         Metric(d.u8()),
		Select:         Selection(d.u8()),
	}
	cfg.ExtendCandidates = d.bool()
	cfg.KeepPruned = d.bool()
	cfg.Seed = d.u64()
	state := make([]byte, d.u32())
	d.bytes(state)
	if d.err != nil {
		return nil, fmt.Errorf("hnsw: load header: %w", d.err)
	}
	x, err := New(cfg)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	if err := x.src.UnmarshalBinary(state); err != nil {
		return nil, fmt.Errorf("%w: rng state: %v", ErrCorrupt, err)
	}

	count := d.u64()
	entry := d.u32()
	maxLvl := d.u32()
	live := d.u64()
	if d.err != nil {
		return nil, fmt.Errorf("hnsw: load: %w", d.err)
	}
	if count > maxNodes || maxLvl > maxLevel || live > count {
		return nil, fmt.Errorf("%w: implausible counts", ErrCorrupt)
	}
	// Grow nodes incrementally rather than trusting count up front: a
	// corrupt count then fails on a short read instead of a huge allocation.
	x.nodes = make([]node, 0, min(count, 1<<16))
	for i := uint64(0); i < count; i++ {
		n := node{id: d.u64(), deleted: d.bool()}
		lvl := d.u32()
		if d.err != nil {
			return nil, fmt.Errorf("hnsw: load node %d: %w", i, d.err)
		}
		if lvl > maxLvl {
			return nil, fmt.Errorf("%w: node %d level %d exceeds max %d", ErrCorrupt, i, lvl, maxLvl)
		}
		n.neighbors = make([][]uint32, lvl+1)
		for l := range n.neighbors {
			cnt := d.u32()
			if d.err != nil {
				return nil, fmt.Errorf("hnsw: load node %d: %w", i, d.err)
			}
			if int(cnt) > x.mmax(l) {
				return nil, fmt.Errorf("%w: node %d has %d neighbours on layer %d", ErrCorrupt, i, cnt, l)
			}
			list := make([]uint32, cnt, x.mmax(l))
			for j := range list {
				list[j] = d.u32()
				if uint64(list[j]) >= count {
					return nil, fmt.Errorf("%w: neighbour index out of range", ErrCorrupt)
				}
			}
			n.neighbors[l] = list
		}
		if !n.deleted {
			if _, dup := x.byID[n.id]; dup {
				return nil, fmt.Errorf("%w: duplicate live id %d", ErrCorrupt, n.id)
			}
			x.byID[n.id] = uint32(i)
		}
		x.nodes = append(x.nodes, n)
	}
	if uint64(len(x.byID)) != live {
		return nil, fmt.Errorf("%w: live count mismatch", ErrCorrupt)
	}
	if count > 0 && uint64(entry) >= count {
		return nil, fmt.Errorf("%w: entry point out of range", ErrCorrupt)
	}
	x.vecs = make([]float32, count*uint64(cfg.Dim))
	d.floats(x.vecs)
	if d.err != nil {
		return nil, fmt.Errorf("hnsw: load vectors: %w", d.err)
	}
	want := d.h.Sum32() // everything consumed so far; the trailer itself is read unhashed
	if got := d.trailer(); d.err != nil || got != want {
		return nil, fmt.Errorf("%w: checksum mismatch", ErrCorrupt)
	}
	x.entry = entry
	x.maxLevel = int(maxLvl)
	x.live = int(live)
	return x, nil
}

// encoder writes little-endian primitives with a sticky error.
type encoder struct {
	w   *bufio.Writer
	err error
	buf [8]byte
}

func (e *encoder) bytes(b []byte) {
	if e.err == nil {
		_, e.err = e.w.Write(b)
	}
}

func (e *encoder) u8(v uint8) {
	if e.err == nil {
		e.err = e.w.WriteByte(v)
	}
}

func (e *encoder) bool(v bool) {
	if v {
		e.u8(1)
	} else {
		e.u8(0)
	}
}

func (e *encoder) u32(v uint32) {
	binary.LittleEndian.PutUint32(e.buf[:4], v)
	e.bytes(e.buf[:4])
}

func (e *encoder) u64(v uint64) {
	binary.LittleEndian.PutUint64(e.buf[:8], v)
	e.bytes(e.buf[:8])
}

// floats writes raw IEEE-754 bits in 16 KiB chunks; binary.Write on a
// []float32 goes through reflection and is an order of magnitude slower.
func (e *encoder) floats(v []float32) {
	const chunk = 4096
	buf := make([]byte, 4*chunk)
	for len(v) > 0 && e.err == nil {
		n := min(len(v), chunk)
		for i, f := range v[:n] {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(f))
		}
		e.bytes(buf[:4*n])
		v = v[n:]
	}
}

// decoder mirrors encoder. It hashes exactly the bytes it consumes: hashing
// underneath the bufio.Reader would include whatever it read ahead, such as
// the checksum trailer itself.
type decoder struct {
	r   *bufio.Reader
	h   hash.Hash32
	err error
	buf [8]byte
}

func (d *decoder) bytes(b []byte) {
	if d.err != nil {
		return
	}
	if _, d.err = io.ReadFull(d.r, b); d.err == nil {
		d.h.Write(b)
	}
}

// trailer reads the checksum without feeding it to the hash.
func (d *decoder) trailer() uint32 {
	if d.err != nil {
		return 0
	}
	_, d.err = io.ReadFull(d.r, d.buf[:4])
	return binary.LittleEndian.Uint32(d.buf[:4])
}

func (d *decoder) u8() uint8 {
	d.bytes(d.buf[:1])
	return d.buf[0]
}

func (d *decoder) bool() bool { return d.u8() != 0 }

func (d *decoder) u32() uint32 {
	d.bytes(d.buf[:4])
	return binary.LittleEndian.Uint32(d.buf[:4])
}

func (d *decoder) u64() uint64 {
	d.bytes(d.buf[:8])
	return binary.LittleEndian.Uint64(d.buf[:8])
}

func (d *decoder) floats(v []float32) {
	const chunk = 4096
	buf := make([]byte, 4*chunk)
	for len(v) > 0 && d.err == nil {
		n := min(len(v), chunk)
		d.bytes(buf[:4*n])
		for i := range v[:n] {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
		v = v[n:]
	}
}

// Ensure the rng type keeps implementing the marshalling we rely on.
var _ interface {
	MarshalBinary() ([]byte, error)
	UnmarshalBinary([]byte) error
} = (*rand.PCG)(nil)
