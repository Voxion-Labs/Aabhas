package api

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Voxion-Labs/Aabhas/control-plane/internal/auth"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/license"
)

func TestRoleBasedAccessControlOnProtectedEndpoints(t *testing.T) {
	server := NewServer("127.0.0.1:0", log.New(io.Discard, "", 0))

	// 1. Generate Admin, Operator, Inference, and Auditor tokens
	adminToken, _, err := server.TokenManager().GenerateToken("admin-svc", auth.RoleAdmin, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate admin token: %v", err)
	}
	opToken, _, err := server.TokenManager().GenerateToken("operator-svc", auth.RoleOperator, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate op token: %v", err)
	}
	infToken, _, err := server.TokenManager().GenerateToken("inference-svc", auth.RoleInference, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate inf token: %v", err)
	}
	audToken, _, err := server.TokenManager().GenerateToken("auditor-svc", auth.RoleAuditor, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate aud token: %v", err)
	}

	// 2. Unauthenticated request to /api/v1/engine/start -> 401
	{
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/v1/engine/start", strings.NewReader(`{"model_path":"/tmp/test.bin"}`))
		req.Header.Set("Content-Type", "application/json")
		server.httpServer.Handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("unauthenticated start code = %d, want %d", rec.Code, http.StatusUnauthorized)
		}
	}

	// 3. Inference token trying /api/v1/engine/start -> 403 Forbidden
	{
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/v1/engine/start", strings.NewReader(`{"model_path":"/tmp/test.bin"}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+infToken)
		server.httpServer.Handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Fatalf("inference token calling start code = %d, want %d", rec.Code, http.StatusForbidden)
		}
	}

	// 4. Auditor token accessing /metrics -> 200 OK
	{
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
		req.Header.Set("Authorization", "Bearer "+audToken)
		server.httpServer.Handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("auditor token calling metrics code = %d, want %d", rec.Code, http.StatusOK)
		}
	}

	// 5. Operator token accessing /api/v1/hardware/compatibility -> 200 OK
	{
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/v1/hardware/compatibility", nil)
		req.Header.Set("Authorization", "Bearer "+opToken)
		server.httpServer.Handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("operator token calling hw compatibility code = %d, want %d", rec.Code, http.StatusOK)
		}
	}

	// 6. Admin token listing tokens via /api/v1/auth/tokens -> 200 OK
	{
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/v1/auth/tokens", nil)
		req.Header.Set("Authorization", "Bearer "+adminToken)
		server.httpServer.Handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("admin listing tokens code = %d, want %d", rec.Code, http.StatusOK)
		}
		var listResp tokenListResponse
		if err := json.NewDecoder(rec.Body).Decode(&listResp); err != nil {
			t.Fatalf("decode token list: %v", err)
		}
		if len(listResp.Tokens) < 4 {
			t.Fatalf("expected at least 4 tokens, got %d", len(listResp.Tokens))
		}
	}

	// 7. Non-admin trying to list tokens -> 403 Forbidden
	{
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/v1/auth/tokens", nil)
		req.Header.Set("Authorization", "Bearer "+opToken)
		server.httpServer.Handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Fatalf("non-admin listing tokens code = %d, want %d", rec.Code, http.StatusForbidden)
		}
	}
}

func TestLicenseAndModelVerificationEndpoints(t *testing.T) {
	server := NewServer("127.0.0.1:0", log.New(io.Discard, "", 0))
	adminToken, _, err := server.TokenManager().GenerateToken("test-admin", auth.RoleAdmin, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate token: %v", err)
	}

	// 1. Test /api/v1/license/verify endpoint
	pubKey, privKey, err := license.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	hwUUID := "B1C2D3E4-F5A6-7890-1234-567890ABCDEF"
	signed, err := license.IssueLicense(license.LicensePayload{
		LicensedHardwareUUID: hwUUID,
		CustomerName:         "Enterprise Node A",
		Tier:                 license.TierEnterprise,
		ExpiresAt:            time.Now().UTC().Add(30 * 24 * time.Hour),
	}, privKey)
	if err != nil {
		t.Fatalf("issue license: %v", err)
	}

	verifyBody, _ := json.Marshal(licenseVerifyRequest{
		License:            *signed,
		TargetHardwareUUID: hwUUID,
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/license/verify", bytes.NewReader(verifyBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+adminToken)
	server.httpServer.Handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("license verify code = %d, want %d", rec.Code, http.StatusOK)
	}
	var licResp licenseVerifyResponse
	if err := json.NewDecoder(rec.Body).Decode(&licResp); err != nil {
		t.Fatalf("decode license verify resp: %v", err)
	}
	if !licResp.Valid || licResp.CustomerName != "Enterprise Node A" {
		t.Fatalf("unexpected license verify resp: %+v", licResp)
	}

	_ = pubKey

	// 2. Test /api/v1/models/verify endpoint
	tempDir := t.TempDir()
	modelFile := filepath.Join(tempDir, "test-model.bin")
	modelContent := []byte("AABHAS-VERIFIED-MODEL-CONTENT-BLOCK")
	if err := os.WriteFile(modelFile, modelContent, 0644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	modelVerifyBody, _ := json.Marshal(modelVerifyRequest{
		ModelPath: modelFile,
	})
	recM := httptest.NewRecorder()
	reqM := httptest.NewRequest(http.MethodPost, "/api/v1/models/verify", bytes.NewReader(modelVerifyBody))
	reqM.Header.Set("Content-Type", "application/json")
	reqM.Header.Set("Authorization", "Bearer "+adminToken)
	server.httpServer.Handler.ServeHTTP(recM, reqM)

	if recM.Code != http.StatusOK {
		t.Fatalf("model verify code = %d, want %d", recM.Code, http.StatusOK)
	}
	var modelResp modelVerifyResponse
	if err := json.NewDecoder(recM.Body).Decode(&modelResp); err != nil {
		t.Fatalf("decode model verify resp: %v", err)
	}
	if !modelResp.Valid || modelResp.ComputedSHA256 == "" || modelResp.FileSizeBytes != int64(len(modelContent)) {
		t.Fatalf("unexpected model verify resp: %+v", modelResp)
	}
}
