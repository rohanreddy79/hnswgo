// Command hnswgo exposes the hnsw package as an HTTP service and a
// recall-versus-throughput benchmark.
//
//	hnswgo serve --addr :8080 --dim 64 [--load index.hnsw]
//	hnswgo bench --n 20000 --dim 64 --k 10 --queries 500
package main

import (
	"fmt"
	"io"
	"os"
)

const usage = `usage: hnswgo <command> [flags]

commands:
  serve   run the JSON HTTP API (see "hnswgo serve -h")
  bench   build an index on synthetic clustered data and print recall vs QPS
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}
	switch args[0] {
	case "serve":
		return runServe(args[1:], stdout, stderr)
	case "bench":
		return runBench(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		fmt.Fprint(stdout, usage)
		return 0
	default:
		fmt.Fprintf(stderr, "hnswgo: unknown command %q\n\n%s", args[0], usage)
		return 2
	}
}
