package orchestrator

import (
	"math"
	"testing"
)

func TestCalculatePercentile(t *testing.T) {
	if CalculatePercentile(nil, 50) != 0.0 {
		t.Fatal("expected 0 for nil slice")
	}

	samples := make([]float64, 100)
	for i := 0; i < 100; i++ {
		samples[i] = float64(i + 1)
	}

	p50 := CalculatePercentile(samples, 50.0)
	if math.Abs(p50-50.5) > 1e-4 {
		t.Fatalf("expected P50=50.5, got %f", p50)
	}

	p90 := CalculatePercentile(samples, 90.0)
	if math.Abs(p90-90.1) > 1e-4 {
		t.Fatalf("expected P90=90.1, got %f", p90)
	}
}

func TestRunSyntheticBenchmark(t *testing.T) {
	res := RunSyntheticBenchmark()

	if res.Memory.CopyGBPS <= 0.0 {
		t.Fatalf("expected positive copy bandwidth, got %f", res.Memory.CopyGBPS)
	}
	if len(res.Kernels) == 0 {
		t.Fatal("expected kernel benchmarks")
	}
	if res.Inference.PrefillTokensPerSec <= 0.0 {
		t.Fatalf("expected positive prefill tok/s, got %f", res.Inference.PrefillTokensPerSec)
	}
	if res.Inference.DecodeTokensPerSec <= 0.0 {
		t.Fatalf("expected positive decode tok/s, got %f", res.Inference.DecodeTokensPerSec)
	}
}
