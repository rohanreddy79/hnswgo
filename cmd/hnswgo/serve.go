// The HTTP service: a thin JSON layer over one hnsw.Index.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/rohanreddy79/hnswgo/hnsw"
)

// maxBodyBytes bounds a request body; 64 MiB is ~250k 64-dim vectors.
const maxBodyBytes = 64 << 20

func runServe(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", ":8080", "listen address")
	dim := fs.Int("dim", 64, "vector dimension (ignored with --load)")
	m := fs.Int("m", 16, "HNSW M: edges per node (ignored with --load)")
	efc := fs.Int("ef-construction", 200, "beam width while inserting (ignored with --load)")
	metric := fs.String("metric", "l2", "distance metric: l2, cosine or dot (ignored with --load)")
	load := fs.String("load", "", "load an index saved by POST /save instead of starting empty")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	logger := log.New(stderr, "hnswgo: ", log.LstdFlags)

	idx, err := openIndex(*load, *dim, *m, *efc, *metric)
	if err != nil {
		logger.Print(err)
		return 1
	}
	srv := &http.Server{
		Addr:              *addr,
		Handler:           newServer(idx),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      60 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errc := make(chan error, 1)
	go func() {
		st := idx.Stats()
		logger.Printf("listening on %s (dim=%d live=%d metric=%s)", *addr, st.Dim, st.Live, st.Metric)
		errc <- srv.ListenAndServe()
	}()
	select {
	case err := <-errc:
		logger.Print(err)
		return 1
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			logger.Printf("shutdown: %v", err)
			return 1
		}
		fmt.Fprintln(stdout, "hnswgo: stopped")
		return 0
	}
}

// openIndex loads a saved index or builds an empty one from the flags.
func openIndex(load string, dim, m, efc int, metricName string) (*hnsw.Index, error) {
	if load != "" {
		f, err := os.Open(load)
		if err != nil {
			return nil, fmt.Errorf("load index: %w", err)
		}
		defer f.Close()
		idx, err := hnsw.Load(f)
		if err != nil {
			return nil, fmt.Errorf("load index %s: %w", load, err)
		}
		return idx, nil
	}
	metric, err := hnsw.ParseMetric(metricName)
	if err != nil {
		return nil, err
	}
	return hnsw.New(hnsw.Config{Dim: dim, M: m, EfConstruction: efc, Metric: metric})
}

// server routes JSON requests to the index. Upserts take an extra mutex so
// a delete+add pair for one id can never interleave with another upsert of
// the same id; reads go straight to the index's own read lock.
type server struct {
	idx    *hnsw.Index
	writes sync.Mutex
	mux    *http.ServeMux
}

func newServer(idx *hnsw.Index) *server {
	s := &server{idx: idx, mux: http.NewServeMux()}
	s.mux.HandleFunc("POST /vectors", s.upsert)
	s.mux.HandleFunc("DELETE /vectors/{id}", s.delete)
	s.mux.HandleFunc("POST /search", s.search)
	s.mux.HandleFunc("POST /save", s.save)
	s.mux.HandleFunc("GET /stats", s.stats)
	return s
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

type vectorEntry struct {
	ID     uint64    `json:"id"`
	Vector []float32 `json:"vector"`
}

func (s *server) upsert(w http.ResponseWriter, r *http.Request) {
	var entries []vectorEntry
	if !decodeBody(w, r, &entries) {
		return
	}
	dim := s.idx.Config().Dim
	for i, e := range entries {
		if len(e.Vector) != dim {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("entry %d: vector has %d dimensions, index has %d", i, len(e.Vector), dim))
			return
		}
	}
	s.writes.Lock()
	defer s.writes.Unlock()
	replaced := 0
	for _, e := range entries {
		if s.idx.Delete(e.ID) {
			replaced++
		}
		if err := s.idx.Add(e.ID, e.Vector); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]int{"upserted": len(entries), "replaced": replaced})
}

func (s *server) delete(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseUint(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "id must be an unsigned integer")
		return
	}
	if !s.idx.Delete(id) {
		writeError(w, http.StatusNotFound, fmt.Sprintf("id %d not found", id))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": id})
}

type searchRequest struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
	Ef     int       `json:"ef"`
}

func (s *server) search(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if !decodeBody(w, r, &req) {
		return
	}
	if dim := s.idx.Config().Dim; len(req.Vector) != dim {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("vector has %d dimensions, index has %d", len(req.Vector), dim))
		return
	}
	if req.K <= 0 {
		req.K = 10
	}
	if req.Ef <= 0 {
		req.Ef = 64
	}
	results := s.idx.Search(req.Vector, req.K, req.Ef)
	if results == nil {
		results = []hnsw.Result{} // an empty JSON array, not null
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

type saveRequest struct {
	Path string `json:"path"`
}

func (s *server) save(w http.ResponseWriter, r *http.Request) {
	var req saveRequest
	if !decodeBody(w, r, &req) {
		return
	}
	if req.Path == "" {
		writeError(w, http.StatusBadRequest, "path is required")
		return
	}
	n, err := saveAtomic(s.idx, req.Path)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"path": req.Path, "bytes": n})
}

// saveAtomic writes to a temp file beside path and renames it into place, so
// a crash mid-write never leaves a truncated index at path.
func saveAtomic(idx *hnsw.Index, path string) (int64, error) {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".hnswgo-*.tmp")
	if err != nil {
		return 0, err
	}
	defer os.Remove(tmp.Name()) // no-op after a successful rename
	cw := &countingWriter{w: tmp}
	if err := idx.Save(cw); err != nil {
		tmp.Close()
		return 0, err
	}
	if err := tmp.Close(); err != nil {
		return 0, err
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return 0, err
	}
	return cw.n, nil
}

func (s *server) stats(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.idx.Stats())
}

func decodeBody(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, err.Error())
			return false
		}
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		// The status line is already sent; nothing better to do than log.
		log.Printf("hnswgo: write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}
