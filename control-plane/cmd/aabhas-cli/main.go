package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type statusResponse struct {
	Status string `json:"status,omitempty"`
	Error  string `json:"error,omitempty"`
	Code   int    `json:"code,omitempty"`
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var exitCode int
	switch os.Args[1] {
	case "start":
		exitCode = runStart(os.Args[2:])
	case "stop":
		exitCode = runStop(os.Args[2:])
	case "help", "-h", "--help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "aabhas-cli: unknown command %q\n\n", os.Args[1])
		usage()
		exitCode = 2
	}
	if exitCode != 0 {
		os.Exit(exitCode)
	}
}

func runStart(args []string) int {
	flags := flag.NewFlagSet("start", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	address := flags.String("addr", envOr("AABHAS_LISTEN_ADDR", "127.0.0.1:8080"), "daemon address")
	modelPath := flags.String("model", "", "model file path")
	timeout := flags.Duration("timeout", 15*time.Second, "HTTP request timeout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if strings.TrimSpace(*modelPath) == "" || strings.IndexByte(*modelPath, 0) >= 0 {
		fmt.Fprintln(os.Stderr, "aabhas-cli: -model is required")
		return 2
	}
	return callDaemon(*address, "/api/v1/engine/start", map[string]string{"model_path": *modelPath}, *timeout)
}

func runStop(args []string) int {
	flags := flag.NewFlagSet("stop", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	address := flags.String("addr", envOr("AABHAS_LISTEN_ADDR", "127.0.0.1:8080"), "daemon address")
	timeout := flags.Duration("timeout", 15*time.Second, "HTTP request timeout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "aabhas-cli: stop accepts no positional arguments")
		return 2
	}
	return callDaemon(*address, "/api/v1/engine/stop", nil, *timeout)
}

func callDaemon(address, path string, payload any, timeout time.Duration) int {
	if timeout <= 0 {
		fmt.Fprintln(os.Stderr, "aabhas-cli: timeout must be positive")
		return 2
	}
	base := strings.TrimRight(strings.TrimSpace(address), "/")
	if !strings.HasPrefix(base, "http://") && !strings.HasPrefix(base, "https://") {
		base = "http://" + base
	}
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			fmt.Fprintf(os.Stderr, "aabhas-cli: encode request: %v\n", err)
			return 1
		}
		body = strings.NewReader(string(encoded))
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, base+path, body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "aabhas-cli: create request: %v\n", err)
		return 1
	}
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := (&http.Client{Timeout: timeout}).Do(request)
	if err != nil {
		fmt.Fprintf(os.Stderr, "aabhas-cli: request failed: %v\n", err)
		return 1
	}
	defer response.Body.Close()
	var result statusResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&result); err != nil && !errors.Is(err, io.EOF) {
		fmt.Fprintf(os.Stderr, "aabhas-cli: decode response: %v\n", err)
		return 1
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		if result.Error == "" {
			result.Error = response.Status
		}
		fmt.Fprintf(os.Stderr, "aabhas-cli: %s\n", result.Error)
		return 1
	}
	if result.Status != "" {
		fmt.Println(result.Status)
	} else {
		fmt.Println(response.Status)
	}
	return 0
}

func envOr(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func usage() {
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  aabhas-cli start -model PATH [-addr HOST:PORT] [-timeout DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli stop [-addr HOST:PORT] [-timeout DURATION]")
}
