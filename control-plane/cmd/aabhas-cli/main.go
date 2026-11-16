package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/Voxion-Labs/Aabhas/control-plane/internal/auth"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/license"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/orchestrator"
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
	case "status":
		exitCode = runStatus(os.Args[2:])
	case "health":
		exitCode = runHealth(os.Args[2:])
	case "hwcheck":
		exitCode = runHwCheck(os.Args[2:])
	case "benchmark":
		exitCode = runBenchmark(os.Args[2:])
	case "keygen":
		exitCode = runKeygen(os.Args[2:])
	case "license":
		exitCode = runLicense(os.Args[2:])
	case "verify-model":
		exitCode = runVerifyModel(os.Args[2:])
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

func runStatus(args []string) int {
	flags := flag.NewFlagSet("status", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	address := flags.String("addr", envOr("AABHAS_LISTEN_ADDR", "127.0.0.1:8080"), "daemon address")
	timeout := flags.Duration("timeout", 15*time.Second, "HTTP request timeout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "aabhas-cli: status accepts no positional arguments")
		return 2
	}
	return callDaemonMethod(http.MethodGet, *address, "/api/v1/engine/status", nil, *timeout)
}

func runHealth(args []string) int {
	flags := flag.NewFlagSet("health", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	address := flags.String("addr", envOr("AABHAS_LISTEN_ADDR", "127.0.0.1:8080"), "daemon address")
	timeout := flags.Duration("timeout", 3*time.Second, "HTTP request timeout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "aabhas-cli: health accepts no positional arguments")
		return 2
	}
	return callDaemonMethod(http.MethodGet, *address, "/healthz", nil, *timeout)
}

func runHwCheck(args []string) int {
	flags := flag.NewFlagSet("hwcheck", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	address := flags.String("addr", envOr("AABHAS_LISTEN_ADDR", "127.0.0.1:8080"), "daemon address")
	timeout := flags.Duration("timeout", 10*time.Second, "HTTP request timeout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	return callDaemonMethod(http.MethodGet, *address, "/api/v1/hardware/compatibility", nil, *timeout)
}

func runBenchmark(args []string) int {
	flags := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	address := flags.String("addr", envOr("AABHAS_LISTEN_ADDR", "127.0.0.1:8080"), "daemon address")
	timeout := flags.Duration("timeout", 60*time.Second, "HTTP request timeout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	return callDaemonMethod(http.MethodPost, *address, "/api/v1/benchmark", map[string]any{"type": "full"}, *timeout)
}

func callDaemon(address, path string, payload any, timeout time.Duration) int {
	return callDaemonMethod(http.MethodPost, address, path, payload, timeout)
}

func callDaemonMethod(method, address, path string, payload any, timeout time.Duration) int {
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
	request, err := http.NewRequestWithContext(ctx, method, base+path, body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "aabhas-cli: create request: %v\n", err)
		return 1
	}
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if token := envOr("AABHAS_API_TOKEN", ""); token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
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

func runKeygen(args []string) int {
	flags := flag.NewFlagSet("keygen", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	name := flags.String("name", "service-key", "token description/name")
	roleStr := flags.String("role", "operator", "token role (admin, operator, inference, auditor)")
	ttl := flags.Duration("ttl", 30*24*time.Hour, "token validity duration (0 for indefinite)")
	if err := flags.Parse(args); err != nil {
		return 2
	}

	role := auth.Role(*roleStr)
	tm := auth.NewTokenManager()
	raw, rec, err := tm.GenerateToken(*name, role, *ttl)
	if err != nil {
		fmt.Fprintf(os.Stderr, "aabhas-cli: %v\n", err)
		return 1
	}

	fmt.Printf("Generated Aabhas Cryptographic Token:\n")
	fmt.Printf("  Token ID:   %s\n", rec.ID)
	fmt.Printf("  Name:       %s\n", rec.Name)
	fmt.Printf("  Role:       %s\n", rec.Role)
	if !rec.ExpiresAt.IsZero() {
		fmt.Printf("  Expires:    %s\n", rec.ExpiresAt.Format(time.RFC3339))
	} else {
		fmt.Printf("  Expires:    never\n")
	}
	fmt.Printf("  Secret:     %s\n\n", raw)
	fmt.Printf("Export for CLI usage:\n")
	fmt.Printf("  export AABHAS_API_TOKEN=\"%s\"\n", raw)
	return 0
}

func runLicense(args []string) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: aabhas-cli license [generate|verify] [options]")
		return 2
	}
	switch args[0] {
	case "generate":
		flags := flag.NewFlagSet("license generate", flag.ContinueOnError)
		flags.SetOutput(os.Stderr)
		uuid := flags.String("uuid", "", "target hardware UUID")
		customer := flags.String("customer", "Voxion Enterprise Customer", "customer name")
		tierStr := flags.String("tier", "Enterprise", "license tier (Enterprise, Pro, Community)")
		outFile := flags.String("out", "", "output file path for JSON license")
		if err := flags.Parse(args[1:]); err != nil {
			return 2
		}
		if strings.TrimSpace(*uuid) == "" {
			fmt.Fprintln(os.Stderr, "aabhas-cli: -uuid is required")
			return 2
		}

		pub, privKey, err := license.GenerateKeyPair()
		if err != nil {
			fmt.Fprintf(os.Stderr, "aabhas-cli: generate license keys: %v\n", err)
			return 1
		}
		_ = pub

		payload := license.LicensePayload{
			LicensedHardwareUUID:    *uuid,
			CustomerName:            *customer,
			Tier:                    license.LicenseTier(*tierStr),
			MaxConcurrentModels:     8,
			MaxVramBytes:            64 * 1024 * 1024 * 1024,
			MaxContextLength:        8192,
			IssuedAt:                time.Now().UTC(),
			ExpiresAt:               time.Now().UTC().Add(365 * 24 * time.Hour),
			FeatureGpuAcceleration:  true,
			FeatureDirectNvmeStream: true,
		}
		signed, err := license.IssueLicense(payload, privKey)
		if err != nil {
			fmt.Fprintf(os.Stderr, "aabhas-cli: issue license: %v\n", err)
			return 1
		}

		jsonStr := signed.ToJSON()
		if *outFile != "" {
			if err := os.WriteFile(*outFile, []byte(jsonStr), 0644); err != nil {
				fmt.Fprintf(os.Stderr, "aabhas-cli: write license file: %v\n", err)
				return 1
			}
			fmt.Printf("License written to %s\n", *outFile)
		} else {
			fmt.Println(jsonStr)
		}
		return 0

	case "verify":
		flags := flag.NewFlagSet("license verify", flag.ContinueOnError)
		flags.SetOutput(os.Stderr)
		file := flags.String("file", "", "path to license JSON file")
		uuid := flags.String("uuid", "", "target hardware UUID (defaults to license UUID)")
		if err := flags.Parse(args[1:]); err != nil {
			return 2
		}
		if strings.TrimSpace(*file) == "" {
			fmt.Fprintln(os.Stderr, "aabhas-cli: -file is required")
			return 2
		}
		data, err := os.ReadFile(*file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "aabhas-cli: read license file: %v\n", err)
			return 1
		}
		var signed license.SignedLicense
		if err := json.Unmarshal(data, &signed); err != nil {
			fmt.Fprintf(os.Stderr, "aabhas-cli: parse license file: %v\n", err)
			return 1
		}
		targetUUID := *uuid
		if targetUUID == "" {
			targetUUID = signed.LicensedHardwareUUID
		}
		if err := license.ValidateLicense(&signed, targetUUID, nil); err != nil {
			fmt.Fprintf(os.Stderr, "License INVALID: %v\n", err)
			return 1
		}
		fmt.Printf("License VALID: Customer=%s, Tier=%s, BoundUUID=%s\n",
			signed.Payload.CustomerName, signed.Payload.Tier, signed.LicensedHardwareUUID)
		return 0

	default:
		fmt.Fprintf(os.Stderr, "aabhas-cli: unknown license subcommand %q\n", args[0])
		return 2
	}
}

func runVerifyModel(args []string) int {
	flags := flag.NewFlagSet("verify-model", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	modelPath := flags.String("model", "", "model file path")
	expected := flags.String("expected", "", "optional expected SHA-256 checksum")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if strings.TrimSpace(*modelPath) == "" {
		fmt.Fprintln(os.Stderr, "aabhas-cli: -model is required")
		return 2
	}

	checksum, size, err := orchestrator.ComputeFileSHA256(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "aabhas-cli: %v\n", err)
		return 1
	}

	fmt.Printf("Model SHA-256: %s (%d bytes)\n", checksum, size)
	if *expected != "" {
		if subtle.ConstantTimeCompare([]byte(checksum), []byte(strings.ToLower(*expected))) == 1 {
			fmt.Println("Integrity check: PASSED (Checksum matches)")
			return 0
		}
		fmt.Println("Integrity check: FAILED (Checksum mismatch)")
		return 1
	}
	return 0
}

func usage() {
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  aabhas-cli start -model PATH [-addr HOST:PORT] [-timeout DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli stop [-addr HOST:PORT] [-timeout DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli status [-addr HOST:PORT] [-timeout DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli health [-addr HOST:PORT] [-timeout DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli hwcheck [-addr HOST:PORT] [-timeout DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli benchmark [-addr HOST:PORT] [-timeout DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli keygen [-name NAME] [-role ROLE] [-ttl DURATION]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli license [generate|verify] [options]")
	fmt.Fprintln(os.Stderr, "  aabhas-cli verify-model -model PATH [-expected SHA256]")
}
