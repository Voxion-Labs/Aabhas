package orchestrator

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"strings"
)

// ExecutionTier represents the recommended execution hierarchy.
type ExecutionTier string

const (
	TierFullGpu         ExecutionTier = "FullGpu"
	TierHybridGpuOffload ExecutionTier = "HybridGpuOffload"
	TierCpuRam          ExecutionTier = "CpuRam"
	TierNvmeOutofCore   ExecutionTier = "NvmeOutofCore"
	TierIncompatible    ExecutionTier = "Incompatible"
)

// BottleneckType describes the primary performance constraint.
type BottleneckType string

const (
	BottleneckNone           BottleneckType = "None"
	BottleneckVramCapacity   BottleneckType = "VramCapacity"
	BottleneckSystemMemory   BottleneckType = "SystemMemory"
	BottleneckStorageIops    BottleneckType = "StorageIops"
	BottleneckCpuVectorWidth BottleneckType = "CpuVectorWidth"
	BottleneckHostBandwidth  BottleneckType = "HostBandwidth"
)

// HostHardwareProfile summarizes system capabilities.
type HostHardwareProfile struct {
	OS               string   `json:"os"`
	Arch             string   `json:"arch"`
	LogicalCPUs      int      `json:"logical_cpus"`
	TotalRAMBytes    uint64   `json:"total_ram_bytes"`
	AvailableRAMBytes uint64  `json:"available_ram_bytes"`
	StorageType      string   `json:"storage_type"`
	StorageFreeBytes uint64   `json:"storage_free_bytes"`
	HasDiscreteGPU   bool     `json:"has_discrete_gpu"`
	GPUName          string   `json:"gpu_name"`
	GPUVendor        string   `json:"gpu_vendor"`
	TotalVRAMBytes   uint64   `json:"total_vram_bytes"`
	FreeVRAMBytes    uint64   `json:"free_vram_bytes"`
	CPUFeatures      []string `json:"cpu_features"`
}

// ModelResourceEstimate models memory footprint.
type ModelResourceEstimate struct {
	ParameterCount         uint64 `json:"parameter_count"`
	WeightBytes            uint64 `json:"weight_bytes"`
	KVCacheBytesPerToken   uint64 `json:"kv_cache_bytes_per_token"`
	ActivationScratchBytes uint64 `json:"activation_scratch_bytes"`
}

// TotalMemoryRequired computes memory needed for weights + KV-cache + activations.
func (m ModelResourceEstimate) TotalMemoryRequired(contextLength, batchSize int) uint64 {
	totalKV := m.KVCacheBytesPerToken * uint64(contextLength) * uint64(batchSize)
	return m.WeightBytes + totalKV + m.ActivationScratchBytes
}

// CompatibilityAssessment returns the evaluation output.
type CompatibilityAssessment struct {
	CanRun                      bool           `json:"can_run"`
	RecommendedTier             ExecutionTier  `json:"recommended_tier"`
	PrimaryBottleneck           BottleneckType `json:"primary_bottleneck"`
	CompatibilityScore          int            `json:"compatibility_score"`
	MaxRecommendedBatchSize     int            `json:"max_recommended_batch_size"`
	MaxRecommendedContextLength int            `json:"max_recommended_context_length"`
	Diagnostics                 []string       `json:"diagnostics"`
}

// ToJSON returns formatted JSON string.
func (c CompatibilityAssessment) ToJSON() (string, error) {
	bytes, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

// HardwareEvaluator evaluates hardware feasibility.
type HardwareEvaluator struct {
	Profile HostHardwareProfile
}

// DetectHostProfile gathers system specifications.
func DetectHostProfile() HostHardwareProfile {
	prof := HostHardwareProfile{
		OS:          runtime.GOOS,
		Arch:        runtime.GOARCH,
		LogicalCPUs: runtime.NumCPU(),
		// Standard conservative baselines
		TotalRAMBytes:     16 * 1024 * 1024 * 1024,
		AvailableRAMBytes: 12 * 1024 * 1024 * 1024,
		StorageType:       "NVMe",
		StorageFreeBytes:  100 * 1024 * 1024 * 1024,
		HasDiscreteGPU:    false,
		GPUName:           "Aabhas Virtual Accelerator",
		GPUVendor:         "Voxion-Labs",
		TotalVRAMBytes:    8 * 1024 * 1024 * 1024,
		FreeVRAMBytes:     8 * 1024 * 1024 * 1024,
		CPUFeatures:       []string{"AVX2", "FMA3", "F16C"},
	}
	return prof
}

// EstimateModelResources computes parameters and memory footprint for transformer models.
func EstimateModelResources(
	vocabSize, modelDim, layers, heads, kvHeads, headDim, intermediateDim int,
	weightDType string,
	tiedLMHead bool,
) ModelResourceEstimate {
	if vocabSize <= 0 || modelDim <= 0 || layers <= 0 {
		return ModelResourceEstimate{}
	}

	d := uint64(modelDim)
	h := uint64(heads)
	if h == 0 {
		h = 1
	}
	hKV := uint64(kvHeads)
	if hKV == 0 {
		hKV = h
	}
	dK := uint64(headDim)
	if dK == 0 {
		dK = d / h
	}
	dFF := uint64(intermediateDim)
	if dFF == 0 {
		dFF = 4 * d
	}
	v := uint64(vocabSize)
	nLayers := uint64(layers)

	embedParams := v * d
	attnParams := d*(h*dK) + 2*(d*hKV*dK) + (h*dK)*d + d
	ffnParams := 2*(d*dFF) + (dFF*d) + d
	perLayerParams := attnParams + ffnParams

	finalNorm := d
	lmHeadParams := uint64(0)
	if !tiedLMHead {
		lmHeadParams = d * v
	}

	totalParams := embedParams + (nLayers * perLayerParams) + finalNorm + lmHeadParams

	bytesPerParam := 4.0
	switch strings.ToUpper(strings.TrimSpace(weightDType)) {
	case "Q4_0":
		bytesPerParam = 18.0 / 32.0
	case "Q4_1":
		bytesPerParam = 20.0 / 32.0
	case "Q5_0":
		bytesPerParam = 22.0 / 32.0
	case "Q5_1":
		bytesPerParam = 24.0 / 32.0
	case "Q8_0":
		bytesPerParam = 34.0 / 32.0
	case "F16", "BF16":
		bytesPerParam = 2.0
	default:
		bytesPerParam = 4.0
	}

	weightBytes := uint64(math.Ceil(float64(totalParams) * bytesPerParam))
	kvPerToken := 2 * nLayers * hKV * dK * 4 // 4 bytes per float32
	scratchBytes := 4*dFF*4 + (h * dK * 4)

	return ModelResourceEstimate{
		ParameterCount:         totalParams,
		WeightBytes:            weightBytes,
		KVCacheBytesPerToken:   kvPerToken,
		ActivationScratchBytes: scratchBytes,
	}
}

// Evaluate determines execution tier and bottlenecks.
func (e HardwareEvaluator) Evaluate(estimate ModelResourceEstimate, targetContext, targetBatch int) CompatibilityAssessment {
	if estimate.ParameterCount == 0 {
		return CompatibilityAssessment{
			CanRun:             false,
			RecommendedTier:    TierIncompatible,
			PrimaryBottleneck:  BottleneckSystemMemory,
			CompatibilityScore: 0,
			Diagnostics:        []string{"Invalid model parameters: parameter count is zero."},
		}
	}

	if targetContext <= 0 {
		targetContext = 2048
	}
	if targetBatch <= 0 {
		targetBatch = 1
	}

	totalKVBytes := estimate.KVCacheBytesPerToken * uint64(targetContext) * uint64(targetBatch)
	totalResidentMemory := estimate.WeightBytes + totalKVBytes + estimate.ActivationScratchBytes
	availRAM := e.Profile.AvailableRAMBytes

	vram := e.Profile.FreeVRAMBytes
	if vram == 0 {
		vram = e.Profile.TotalVRAMBytes
	}

	// 1. Full GPU
	if e.Profile.HasDiscreteGPU && vram >= totalResidentMemory {
		maxBatch := int(vram / (estimate.WeightBytes + estimate.ActivationScratchBytes))
		if maxBatch < 1 {
			maxBatch = 1
		}
		return CompatibilityAssessment{
			CanRun:                      true,
			RecommendedTier:             TierFullGpu,
			PrimaryBottleneck:           BottleneckNone,
			CompatibilityScore:          95,
			MaxRecommendedBatchSize:     maxBatch,
			MaxRecommendedContextLength: targetContext,
			Diagnostics:                 []string{"Model fits completely within accelerator VRAM."},
		}
	}

	// 2. Hybrid GPU Offload
	if e.Profile.HasDiscreteGPU && vram >= (estimate.WeightBytes/2) {
		return CompatibilityAssessment{
			CanRun:                      true,
			RecommendedTier:             TierHybridGpuOffload,
			PrimaryBottleneck:           BottleneckVramCapacity,
			CompatibilityScore:          80,
			MaxRecommendedBatchSize:     minInt(targetBatch, 4),
			MaxRecommendedContextLength: targetContext,
			Diagnostics:                 []string{"Partial GPU offload active; PCIe bandwidth will bound inference speed."},
		}
	}

	// 3. CPU RAM Resident
	if availRAM >= totalResidentMemory {
		maxBatch := 1
		if totalKV := estimate.KVCacheBytesPerToken * uint64(targetContext); totalKV > 0 && availRAM > estimate.WeightBytes {
			maxBatch = int((availRAM - estimate.WeightBytes) / totalKV)
		}
		if maxBatch < 1 {
			maxBatch = 1
		}
		score := 75
		bottleneck := BottleneckHostBandwidth
		diag := "Model resides fully in host system RAM."
		return CompatibilityAssessment{
			CanRun:                      true,
			RecommendedTier:             TierCpuRam,
			PrimaryBottleneck:           bottleneck,
			CompatibilityScore:          score,
			MaxRecommendedBatchSize:     maxBatch,
			MaxRecommendedContextLength: targetContext,
			Diagnostics:                 []string{diag},
		}
	}

	// 4. NVMe Out-of-Core Streaming
	minStreamedRAM := totalKVBytes + (estimate.WeightBytes / 8) + estimate.ActivationScratchBytes
	if availRAM >= minStreamedRAM && e.Profile.StorageFreeBytes >= estimate.WeightBytes {
		diag := "Direct NVMe-to-CPU tensor streaming active. Out-of-core execution drops layers on demand."
		return CompatibilityAssessment{
			CanRun:                      true,
			RecommendedTier:             TierNvmeOutofCore,
			PrimaryBottleneck:           BottleneckStorageIops,
			CompatibilityScore:          65,
			MaxRecommendedBatchSize:     1,
			MaxRecommendedContextLength: minInt(targetContext, 4096),
			Diagnostics:                 []string{diag},
		}
	}

	// 5. Incompatible
	return CompatibilityAssessment{
		CanRun:                      false,
		RecommendedTier:             TierIncompatible,
		PrimaryBottleneck:           BottleneckSystemMemory,
		CompatibilityScore:          10,
		MaxRecommendedBatchSize:     0,
		MaxRecommendedContextLength: 0,
		Diagnostics: []string{
			fmt.Sprintf("Insufficient RAM (%d MiB available) and storage for model weights (%d MiB).",
				availRAM/(1024*1024), estimate.WeightBytes/(1024*1024)),
		},
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
