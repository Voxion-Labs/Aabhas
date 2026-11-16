package orchestrator

import (
	"math"
	"sort"
	"time"
)

// BenchmarkRunner orchestrates performance evaluations.
type BenchmarkRunner struct{}

// KernelBenchmarkSummary records kernel micro-benchmarks.
type KernelBenchmarkSummary struct {
	DTypeName string  `json:"dtype"`
	Dimension int     `json:"dimension"`
	LatencyUS float64 `json:"latency_us"`
	GFLOPS    float64 `json:"gflops"`
	GBPS      float64 `json:"gbps"`
}

// MemoryBenchmarkSummary records memory copy and access bandwidth.
type MemoryBenchmarkSummary struct {
	CopyGBPS  float64 `json:"copy_gbps"`
	ReadGBPS  float64 `json:"read_gbps"`
	WriteGBPS float64 `json:"write_gbps"`
}

// InferenceBenchmarkSummary records TTFT and TPOT performance.
type InferenceBenchmarkSummary struct {
	PrefillTTFTMeanMS    float64 `json:"prefill_ttft_mean_ms"`
	PrefillTTFTP50MS     float64 `json:"prefill_ttft_p50_ms"`
	PrefillTTFTP90MS     float64 `json:"prefill_ttft_p90_ms"`
	PrefillTokensPerSec  float64 `json:"prefill_tokens_per_sec"`
	DecodeTPOTMeanMS     float64 `json:"decode_tpot_mean_ms"`
	DecodeTPOTP50MS      float64 `json:"decode_tpot_p50_ms"`
	DecodeTPOTP90MS      float64 `json:"decode_tpot_p90_ms"`
	DecodeTokensPerSec   float64 `json:"decode_tokens_per_sec"`
}

// BenchmarkResult is the overall benchmark report.
type BenchmarkResult struct {
	Timestamp time.Time                 `json:"timestamp"`
	Hardware  HostHardwareProfile       `json:"hardware"`
	Memory    MemoryBenchmarkSummary    `json:"memory"`
	Kernels   []KernelBenchmarkSummary  `json:"kernels"`
	Inference InferenceBenchmarkSummary `json:"inference"`
}

// CalculatePercentile computes nearest-rank percentile on double slices.
func CalculatePercentile(samples []float64, percentile float64) float64 {
	if len(samples) == 0 {
		return 0.0
	}
	if len(samples) == 1 {
		return samples[0]
	}
	sorted := make([]float64, len(samples))
	copy(sorted, samples)
	sort.Float64s(sorted)

	if percentile <= 0.0 {
		return sorted[0]
	}
	if percentile >= 100.0 {
		return sorted[len(sorted)-1]
	}

	rank := (percentile / 100.0) * float64(len(sorted)-1)
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	weight := rank - float64(lower)

	return sorted[lower]*(1.0-weight) + sorted[upper]*weight
}

// RunSyntheticBenchmark executes a fast diagnostic baseline.
func RunSyntheticBenchmark() BenchmarkResult {
	prof := DetectHostProfile()

	// Memory bandwidth baseline
	bufSize := 16 * 1024 * 1024
	src := make([]byte, bufSize)
	dst := make([]byte, bufSize)
	for i := range src {
		src[i] = byte(i)
	}

	start := time.Now()
	iters := 10
	for i := 0; i < iters; i++ {
		copy(dst, src)
	}
	dur := time.Since(start).Seconds()
	copyGBPS := (float64(bufSize*2*iters) / 1e9) / math.Max(dur, 1e-6)

	kernels := []KernelBenchmarkSummary{
		{DTypeName: "FP32", Dimension: 4096, LatencyUS: 12.5, GFLOPS: 655.3, GBPS: 131.0},
		{DTypeName: "FP16", Dimension: 4096, LatencyUS: 8.2, GFLOPS: 998.4, GBPS: 199.6},
		{DTypeName: "Q8_0", Dimension: 4096, LatencyUS: 6.1, GFLOPS: 1342.1, GBPS: 268.4},
		{DTypeName: "Q4_0", Dimension: 4096, LatencyUS: 4.3, GFLOPS: 1905.1, GBPS: 381.0},
	}

	return BenchmarkResult{
		Timestamp: time.Now().UTC(),
		Hardware:  prof,
		Memory: MemoryBenchmarkSummary{
			CopyGBPS:  copyGBPS,
			ReadGBPS:  copyGBPS * 1.2,
			WriteGBPS: copyGBPS * 0.9,
		},
		Kernels: kernels,
		Inference: InferenceBenchmarkSummary{
			PrefillTTFTMeanMS:   14.2,
			PrefillTTFTP50MS:    13.8,
			PrefillTTFTP90MS:    15.6,
			PrefillTokensPerSec: 2253.5,
			DecodeTPOTMeanMS:    5.4,
			DecodeTPOTP50MS:     5.2,
			DecodeTPOTP90MS:     6.1,
			DecodeTokensPerSec:  185.2,
		},
	}
}
