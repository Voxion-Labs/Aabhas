package orchestrator

import (
	"testing"
)

func TestEstimateModelResourcesLLaMA7B(t *testing.T) {
	est := EstimateModelResources(32000, 4096, 32, 32, 32, 128, 11008, "Q4_0", false)

	if est.ParameterCount < 6_000_000_000 || est.ParameterCount > 7_500_000_000 {
		t.Fatalf("expected ~6.7B parameters, got %d", est.ParameterCount)
	}

	if est.WeightBytes < 3_000_000_000 || est.WeightBytes > 4_500_000_000 {
		t.Fatalf("expected ~3.8GB weight bytes for Q4_0, got %d", est.WeightBytes)
	}

	// 2 * 32 * 32 * 128 * 4 = 1,048,576 bytes per token
	if est.KVCacheBytesPerToken != 1048576 {
		t.Fatalf("expected 1048576 bytes/token KV cache, got %d", est.KVCacheBytesPerToken)
	}

	estF16 := EstimateModelResources(32000, 4096, 32, 32, 32, 128, 11008, "F16", false)
	if estF16.WeightBytes < 12_000_000_000 || estF16.WeightBytes > 15_000_000_000 {
		t.Fatalf("expected ~13.4GB weight bytes for F16, got %d", estF16.WeightBytes)
	}
}

func TestCompatibilityEvaluationTiers(t *testing.T) {
	est := EstimateModelResources(32000, 4096, 32, 32, 32, 128, 11008, "Q4_0", false)

	// 1. Full GPU
	t.Run("FullGPU", func(t *testing.T) {
		prof := DetectHostProfile()
		prof.HasDiscreteGPU = true
		prof.TotalVRAMBytes = 24 * 1024 * 1024 * 1024
		prof.FreeVRAMBytes = 20 * 1024 * 1024 * 1024

		eval := HardwareEvaluator{Profile: prof}
		res := eval.Evaluate(est, 2048, 1)
		if !res.CanRun || res.RecommendedTier != TierFullGpu {
			t.Fatalf("expected FullGpu tier, got %v (canRun=%v)", res.RecommendedTier, res.CanRun)
		}
	})

	// 2. Hybrid GPU
	t.Run("HybridGPU", func(t *testing.T) {
		prof := DetectHostProfile()
		prof.HasDiscreteGPU = true
		prof.TotalVRAMBytes = 4 * 1024 * 1024 * 1024
		prof.FreeVRAMBytes = 2 * 1024 * 1024 * 1024

		eval := HardwareEvaluator{Profile: prof}
		res := eval.Evaluate(est, 2048, 1)
		if !res.CanRun || res.RecommendedTier != TierHybridGpuOffload {
			t.Fatalf("expected HybridGpuOffload tier, got %v", res.RecommendedTier)
		}
	})

	// 3. CPU RAM
	t.Run("CpuRam", func(t *testing.T) {
		prof := DetectHostProfile()
		prof.HasDiscreteGPU = false
		prof.AvailableRAMBytes = 32 * 1024 * 1024 * 1024

		eval := HardwareEvaluator{Profile: prof}
		res := eval.Evaluate(est, 2048, 1)
		if !res.CanRun || res.RecommendedTier != TierCpuRam {
			t.Fatalf("expected CpuRam tier, got %v", res.RecommendedTier)
		}
	})

	// 4. NVMe Out-of-Core
	t.Run("NvmeOutofCore", func(t *testing.T) {
		prof := DetectHostProfile()
		prof.HasDiscreteGPU = false
		prof.AvailableRAMBytes = 3 * 1024 * 1024 * 1024 // 3 GB free < 5.8 GB total
		prof.StorageType = "NVMe"
		prof.StorageFreeBytes = 100 * 1024 * 1024 * 1024

		eval := HardwareEvaluator{Profile: prof}
		res := eval.Evaluate(est, 1024, 1)
		if !res.CanRun || res.RecommendedTier != TierNvmeOutofCore {
			t.Fatalf("expected NvmeOutofCore tier, got %v", res.RecommendedTier)
		}
		if res.MaxRecommendedBatchSize != 1 {
			t.Fatalf("expected batch size 1 for out-of-core, got %d", res.MaxRecommendedBatchSize)
		}
	})

	// 5. Incompatible
	t.Run("Incompatible", func(t *testing.T) {
		prof := DetectHostProfile()
		prof.HasDiscreteGPU = false
		prof.AvailableRAMBytes = 512 * 1024 * 1024
		prof.StorageFreeBytes = 100 * 1024 * 1024

		eval := HardwareEvaluator{Profile: prof}
		res := eval.Evaluate(est, 2048, 1)
		if res.CanRun || res.RecommendedTier != TierIncompatible {
			t.Fatalf("expected Incompatible tier, got %v (canRun=%v)", res.RecommendedTier, res.CanRun)
		}
	})
}

func TestCompatibilityToJSON(t *testing.T) {
	assess := CompatibilityAssessment{
		CanRun:                      true,
		RecommendedTier:             TierCpuRam,
		PrimaryBottleneck:           BottleneckHostBandwidth,
		CompatibilityScore:          75,
		MaxRecommendedBatchSize:     4,
		MaxRecommendedContextLength: 2048,
		Diagnostics:                 []string{"OK"},
	}

	jsonStr, err := assess.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}
	if len(jsonStr) == 0 {
		t.Fatal("expected non-empty json")
	}
}
