package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rohanreddy79/hnswgo/hnsw"
)

func newTestServer(t *testing.T) (*httptest.Server, *hnsw.Index) {
	t.Helper()
	idx, err := hnsw.New(hnsw.Config{Dim: 3, M: 4, EfConstruction: 16, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(newServer(idx))
	t.Cleanup(ts.Close)
	return ts, idx
}

func do(t *testing.T, ts *httptest.Server, method, path string, body any) (int, map[string]any) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if s, ok := body.(string); ok {
			buf.WriteString(s)
		} else if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req, err := http.NewRequest(method, ts.URL+path, &buf)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Fatalf("%s %s: content-type %q", method, path, ct)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("%s %s: bad JSON %q: %v", method, path, raw, err)
	}
	return resp.StatusCode, out
}

func TestUpsertSearchDelete(t *testing.T) {
	ts, idx := newTestServer(t)

	code, out := do(t, ts, "POST", "/vectors", []vectorEntry{
		{ID: 1, Vector: []float32{1, 0, 0}},
		{ID: 2, Vector: []float32{0, 1, 0}},
		{ID: 3, Vector: []float32{0, 0, 1}},
	})
	if code != 200 || out["upserted"] != 3.0 || out["replaced"] != 0.0 {
		t.Fatalf("upsert: %d %v", code, out)
	}
	if idx.Len() != 3 {
		t.Fatalf("Len = %d", idx.Len())
	}

	code, out = do(t, ts, "POST", "/search", searchRequest{Vector: []float32{0.9, 0.1, 0}, K: 2})
	if code != 200 {
		t.Fatalf("search: %d %v", code, out)
	}
	results := out["results"].([]any)
	if len(results) != 2 || results[0].(map[string]any)["id"] != 1.0 || results[1].(map[string]any)["id"] != 2.0 {
		t.Fatalf("search results: %v", results)
	}

	// Upserting an existing id replaces its vector.
	code, out = do(t, ts, "POST", "/vectors", []vectorEntry{{ID: 1, Vector: []float32{0, 0, -1}}})
	if code != 200 || out["replaced"] != 1.0 {
		t.Fatalf("replace: %d %v", code, out)
	}
	_, out = do(t, ts, "POST", "/search", searchRequest{Vector: []float32{0, 0, -1}, K: 1})
	if got := out["results"].([]any)[0].(map[string]any)["id"]; got != 1.0 {
		t.Fatalf("after replace nearest = %v, want 1", got)
	}
	if idx.Len() != 3 {
		t.Fatalf("Len after replace = %d, want 3", idx.Len())
	}

	code, out = do(t, ts, "DELETE", "/vectors/2", nil)
	if code != 200 || out["deleted"] != 2.0 {
		t.Fatalf("delete: %d %v", code, out)
	}
	if code, _ := do(t, ts, "DELETE", "/vectors/2", nil); code != 404 {
		t.Fatalf("second delete: %d, want 404", code)
	}
	if code, _ := do(t, ts, "DELETE", "/vectors/abc", nil); code != 400 {
		t.Fatalf("bad id: %d, want 400", code)
	}
	_, out = do(t, ts, "POST", "/search", searchRequest{Vector: []float32{0, 1, 0}, K: 10})
	for _, r := range out["results"].([]any) {
		if r.(map[string]any)["id"] == 2.0 {
			t.Fatal("deleted id 2 returned by search")
		}
	}
}

func TestSearchDefaultsAndEmpty(t *testing.T) {
	ts, _ := newTestServer(t)
	code, out := do(t, ts, "POST", "/search", searchRequest{Vector: []float32{1, 2, 3}})
	if code != 200 {
		t.Fatalf("empty search: %d %v", code, out)
	}
	if results, ok := out["results"].([]any); !ok || len(results) != 0 {
		t.Fatalf("empty index should give an empty array, got %v", out["results"])
	}
}

func TestValidation(t *testing.T) {
	ts, _ := newTestServer(t)
	cases := []struct {
		method, path string
		body         any
		want         int
	}{
		{"POST", "/vectors", []vectorEntry{{ID: 1, Vector: []float32{1, 2}}}, 400},
		{"POST", "/vectors", `{"not": "a list"}`, 400},
		{"POST", "/vectors", `[{"id": 1, "vector": [1,2,3], "extra": true}]`, 400},
		{"POST", "/search", searchRequest{Vector: []float32{1}}, 400},
		{"POST", "/search", `not json`, 400},
		{"POST", "/save", saveRequest{}, 400},
		{"GET", "/vectors", nil, 405},
		{"GET", "/nope", nil, 404},
	}
	for _, c := range cases {
		var buf bytes.Buffer
		if s, ok := c.body.(string); ok {
			buf.WriteString(s)
		} else if c.body != nil {
			json.NewEncoder(&buf).Encode(c.body)
		}
		req, _ := http.NewRequest(c.method, ts.URL+c.path, &buf)
		resp, err := ts.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != c.want {
			t.Errorf("%s %s: got %d, want %d", c.method, c.path, resp.StatusCode, c.want)
		}
	}
}

func TestSaveStatsAndReload(t *testing.T) {
	ts, _ := newTestServer(t)
	do(t, ts, "POST", "/vectors", []vectorEntry{{ID: 7, Vector: []float32{1, 1, 1}}, {ID: 8, Vector: []float32{-1, 0, 0}}})

	code, out := do(t, ts, "GET", "/stats", nil)
	if code != 200 || out["live"] != 2.0 || out["dim"] != 3.0 || out["metric"] != "l2" {
		t.Fatalf("stats: %d %v", code, out)
	}

	path := filepath.Join(t.TempDir(), "idx.hnsw")
	code, out = do(t, ts, "POST", "/save", saveRequest{Path: path})
	if code != 200 || out["bytes"].(float64) <= 0 {
		t.Fatalf("save: %d %v", code, out)
	}
	info, err := os.Stat(path)
	if err != nil || info.Size() != int64(out["bytes"].(float64)) {
		t.Fatalf("saved file: %v, size %d vs %v", err, info.Size(), out["bytes"])
	}
	if leftovers, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".hnswgo-*.tmp")); len(leftovers) != 0 {
		t.Fatalf("temp files left behind: %v", leftovers)
	}

	// The --load path of serve must bring the same vectors back.
	loaded, err := openIndex(path, 0, 0, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 2 {
		t.Fatalf("loaded Len = %d", loaded.Len())
	}
	if got := loaded.Search([]float32{1, 1, 1}, 1, 8); len(got) != 1 || got[0].ID != 7 {
		t.Fatalf("loaded search: %v", got)
	}

	if _, err := openIndex(filepath.Join(t.TempDir(), "missing"), 0, 0, 0, ""); err == nil {
		t.Fatal("expected an error for a missing index file")
	}
	if _, err := openIndex("", 4, 8, 50, "manhattan"); err == nil {
		t.Fatal("expected an error for an unknown metric")
	}
	fresh, err := openIndex("", 4, 8, 50, "cosine")
	if err != nil || fresh.Config().Metric != hnsw.Cosine || fresh.Config().Dim != 4 {
		t.Fatalf("fresh index: %v %+v", err, fresh.Config())
	}
}

func TestSaveFailsForBadDirectory(t *testing.T) {
	ts, _ := newTestServer(t)
	code, out := do(t, ts, "POST", "/save", saveRequest{Path: filepath.Join(t.TempDir(), "no", "such", "dir", "x")})
	if code != 500 || !strings.Contains(fmt.Sprint(out["error"]), "no such") {
		t.Fatalf("save to bad dir: %d %v", code, out)
	}
}

func TestRunUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(nil, &out, &errOut); code != 2 || !strings.Contains(errOut.String(), "usage") {
		t.Fatalf("no args: code %d, stderr %q", code, errOut.String())
	}
	errOut.Reset()
	if code := run([]string{"bogus"}, &out, &errOut); code != 2 || !strings.Contains(errOut.String(), "unknown command") {
		t.Fatalf("bogus: code %d, stderr %q", code, errOut.String())
	}
	out.Reset()
	if code := run([]string{"help"}, &out, &errOut); code != 0 || !strings.Contains(out.String(), "serve") {
		t.Fatalf("help: code %d, stdout %q", code, out.String())
	}
	errOut.Reset()
	if code := run([]string{"serve", "--metric", "nope", "--addr", "127.0.0.1:0"}, &out, &errOut); code != 1 {
		t.Fatalf("serve with bad metric: code %d, stderr %q", code, errOut.String())
	}
	if code := run([]string{"bench", "--ef", "x"}, &out, &errOut); code != 2 {
		t.Fatalf("bench with bad ef: code %d", code)
	}
	if code := run([]string{"bench", "--select", "nope"}, &out, &errOut); code != 2 {
		t.Fatalf("bench with bad select: code %d", code)
	}
}
