// Package api exposes the control-plane HTTP API.
package api

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Voxion-Labs/Aabhas/control-plane/internal/auth"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/cgo_bridge"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/license"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/observability"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/orchestrator"
)

const maxRequestBodyBytes = 1 << 20

type startRequest struct {
	ModelPath string `json:"model_path"`
}

type modelRegisterRequest struct {
	ModelPath string `json:"model_path"`
}

type response struct {
	Status string `json:"status,omitempty"`
	Error  string `json:"error,omitempty"`
	Code   int    `json:"code,omitempty"`
}

type engineStatusResponse struct {
	Status         string `json:"status"`
	ModelPath      string `json:"model_path,omitempty"`
	Generation     uint64 `json:"generation"`
	Starts         uint64 `json:"starts"`
	Stops          uint64 `json:"stops"`
	RejectedStarts uint64 `json:"rejected_starts"`
}

type modelsResponse struct {
	Models []orchestrator.ModelRecord `json:"models"`
}

type tokenGenerateRequest struct {
	Name       string    `json:"name"`
	Role       auth.Role `json:"role"`
	TTLSeconds int64     `json:"ttl_seconds,omitempty"`
}

type tokenRevokeRequest struct {
	TokenID string `json:"token_id"`
}

type tokenResponse struct {
	Token  string            `json:"token,omitempty"`
	Record *auth.TokenRecord `json:"record,omitempty"`
}

type tokenListResponse struct {
	Tokens []auth.TokenRecord `json:"tokens"`
}

type licenseVerifyRequest struct {
	License            license.SignedLicense `json:"license"`
	TargetHardwareUUID string                `json:"target_hardware_uuid,omitempty"`
}

type licenseVerifyResponse struct {
	Valid          bool   `json:"valid"`
	CustomerName   string `json:"customer_name,omitempty"`
	Tier           string `json:"tier,omitempty"`
	HardwareBound  string `json:"hardware_bound,omitempty"`
	ExpirationDate string `json:"expiration_date,omitempty"`
	Error          string `json:"error,omitempty"`
}

type modelVerifyRequest struct {
	ModelPath      string `json:"model_path"`
	ExpectedSHA256 string `json:"expected_sha256,omitempty"`
}

type modelVerifyResponse struct {
	Valid          bool   `json:"valid"`
	ModelPath      string `json:"model_path"`
	ComputedSHA256 string `json:"computed_sha256"`
	FileSizeBytes  int64  `json:"file_size_bytes"`
	Error          string `json:"error,omitempty"`
}

type identityContextKey struct{}

// Server owns the HTTP server and its engine lifecycle handlers.
type Server struct {
	httpServer  *http.Server
	logger      *log.Logger
	controller  *orchestrator.ExecutionController
	registry    *orchestrator.ModelRegistry
	metrics     *observability.Metrics
	tokens      *auth.TokenManager
	lifecycleMu sync.Mutex
	requestSeq  atomic.Uint64
}

// NewServer constructs a bounded-timeout REST server on addr.
func NewServer(addr string, logger *log.Logger) *Server {
	if logger == nil {
		logger = log.Default()
	}

	server := &Server{
		logger:     logger,
		controller: orchestrator.NewExecutionController(),
		registry:   orchestrator.NewModelRegistry(),
		metrics:    observability.NewMetrics(),
		tokens:     auth.NewTokenManager(),
	}
	if envToken := strings.TrimSpace(os.Getenv("AABHAS_API_TOKEN")); envToken != "" {
		_ = server.tokens.RegisterStaticAdminToken(envToken, "env-admin")
	}

	protectedMux := http.NewServeMux()
	protectedMux.HandleFunc("/api/v1/engine/start", server.requireRole(auth.RoleAdmin, server.handleStart))
	protectedMux.HandleFunc("/api/v1/engine/stop", server.requireRole(auth.RoleAdmin, server.handleStop))
	protectedMux.HandleFunc("/api/v1/engine/status", server.requireRole(auth.RoleAuditor, server.handleStatus))
	protectedMux.HandleFunc("/api/v1/models", server.requireRole(auth.RoleInference, server.handleModels))
	protectedMux.HandleFunc("/api/v1/hardware/compatibility", server.requireRole(auth.RoleOperator, server.handleHardwareCompatibility))
	protectedMux.HandleFunc("/api/v1/benchmark", server.requireRole(auth.RoleOperator, server.handleBenchmark))
	protectedMux.HandleFunc("/api/v1/auth/tokens", server.requireRole(auth.RoleAdmin, server.handleTokens))
	protectedMux.HandleFunc("/api/v1/license/verify", server.requireRole(auth.RoleOperator, server.handleLicenseVerify))
	protectedMux.HandleFunc("/api/v1/models/verify", server.requireRole(auth.RoleOperator, server.handleModelVerify))
	protectedMux.HandleFunc("/metrics", server.requireRole(auth.RoleAuditor, server.handleMetrics))

	publicMux := http.NewServeMux()
	publicMux.HandleFunc("/healthz", server.handleHealth)
	publicMux.HandleFunc("/readyz", server.handleReady)
	protected := server.authenticate(protectedMux)
	rootHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz", "/readyz":
			publicMux.ServeHTTP(w, r)
		default:
			protected.ServeHTTP(w, r)
		}
	})
	server.httpServer = &http.Server{
		Addr:              addr,
		Handler:           server.instrument(rootHandler),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	return server
}

type observedResponseWriter struct {
	http.ResponseWriter
	status int
}

func (w *observedResponseWriter) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *observedResponseWriter) Write(body []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(body)
}

func (s *Server) instrument(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := fmt.Sprintf("req-%d", s.requestSeq.Add(1))
		w.Header().Set("X-Request-ID", requestID)
		observed := &observedResponseWriter{ResponseWriter: w}
		started := time.Now()
		next.ServeHTTP(observed, r)
		status := observed.status
		if status == 0 {
			status = http.StatusOK
		}
		duration := time.Since(started)
		s.metrics.ObserveRequest(r.Method, r.URL.Path, status, duration)
		s.logger.Printf("http request_id=%s method=%s path=%s status=%d duration_ms=%d", requestID, r.Method, r.URL.Path, status, duration.Milliseconds())
	})
}

func (s *Server) authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// If no tokens configured, development mode allows all requests
		if len(s.tokens.ListTokens()) == 0 {
			next.ServeHTTP(w, r)
			return
		}

		rawToken := extractBearerOrKey(r)
		if rawToken == "" {
			w.Header().Set("WWW-Authenticate", `Bearer realm="aabhas"`)
			writeJSON(w, http.StatusUnauthorized, response{Error: "bearer or api key authentication required"})
			return
		}

		ident, err := s.tokens.ValidateToken(rawToken)
		if err != nil {
			w.Header().Set("WWW-Authenticate", `Bearer realm="aabhas"`)
			writeJSON(w, http.StatusUnauthorized, response{Error: err.Error()})
			return
		}

		ctx := context.WithValue(r.Context(), identityContextKey{}, ident)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *Server) requireRole(role auth.Role, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(s.tokens.ListTokens()) == 0 {
			next(w, r)
			return
		}

		ident, ok := r.Context().Value(identityContextKey{}).(*auth.Identity)
		if !ok || ident == nil {
			w.Header().Set("WWW-Authenticate", `Bearer realm="aabhas"`)
			writeJSON(w, http.StatusUnauthorized, response{Error: "bearer authentication required"})
			return
		}

		if !ident.HasRole(role) {
			writeJSON(w, http.StatusForbidden, response{Error: fmt.Sprintf("permission denied: role %s required", role)})
			return
		}

		next(w, r)
	}
}

func extractBearerOrKey(r *http.Request) string {
	const prefix = "Bearer "
	authHeader := r.Header.Get("Authorization")
	if len(authHeader) > len(prefix) && strings.EqualFold(authHeader[:len(prefix)], prefix) {
		return strings.TrimSpace(authHeader[len(prefix):])
	}
	if key := r.Header.Get("X-Aabhas-Key"); key != "" {
		return strings.TrimSpace(key)
	}
	return ""
}

// ListenAndServe starts accepting HTTP requests.
func (s *Server) ListenAndServe() error {
	s.logger.Printf("REST server listening on %s", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

// Shutdown stops accepting requests and waits for in-flight handlers.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func (s *Server) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()

	var request startRequest
	if !decodeJSON(w, r, &request) {
		s.metrics.IncEngineStartFailure()
		return
	}
	if strings.TrimSpace(request.ModelPath) == "" || strings.IndexByte(request.ModelPath, 0) >= 0 {
		s.metrics.IncEngineStartFailure()
		writeJSON(w, http.StatusBadRequest, response{Error: "model_path is required"})
		return
	}
	if err := r.Context().Err(); err != nil {
		s.metrics.IncEngineStartFailure()
		writeJSON(w, http.StatusRequestTimeout, response{Error: "request canceled"})
		return
	}

	lease, err := s.controller.Acquire(request.ModelPath)
	if err != nil {
		s.metrics.IncEngineStartFailure()
		writeControlError(w, err)
		return
	}

	if err := cgo_bridge.Start(request.ModelPath); err != nil {
		s.metrics.IncEngineStartFailure()
		_ = lease.Cancel()
		s.logger.Printf("engine start rejected: %v", err)
		writeEngineError(w, err)
		return
	}
	if err := r.Context().Err(); err != nil {
		s.metrics.IncEngineStartFailure()
		_ = cgo_bridge.Stop()
		_ = lease.Cancel()
		writeJSON(w, http.StatusRequestTimeout, response{Error: "request canceled"})
		return
	}
	s.metrics.IncEngineStart()
	s.logger.Printf("engine started for model path %q", request.ModelPath)
	writeJSON(w, http.StatusOK, response{Status: "started"})
}

func (s *Server) handleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()

	if err := cgo_bridge.Stop(); err != nil {
		s.metrics.IncEngineStopFailure()
		s.logger.Printf("engine stop failed: %v", err)
		writeEngineError(w, err)
		return
	}
	_ = s.controller.Release()
	s.metrics.IncEngineStop()
	s.logger.Printf("engine stopped")
	writeJSON(w, http.StatusOK, response{Status: "stopped"})
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	stats := s.controller.Stats()
	status := "stopped"
	if stats.Active {
		status = "running"
	}
	writeJSON(w, http.StatusOK, engineStatusResponse{
		Status:         status,
		ModelPath:      stats.ModelPath,
		Generation:     stats.Generation,
		Starts:         stats.Starts,
		Stops:          stats.Stops,
		RejectedStarts: stats.RejectedStarts,
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	writeJSON(w, http.StatusOK, response{Status: "ok"})
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	writeJSON(w, http.StatusOK, response{Status: "ready"})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, modelsResponse{Models: s.registry.List()})
	case http.MethodPost:
		var request modelRegisterRequest
		if !decodeJSON(w, r, &request) {
			return
		}
		if strings.TrimSpace(request.ModelPath) == "" || strings.IndexByte(request.ModelPath, 0) >= 0 {
			writeJSON(w, http.StatusBadRequest, response{Error: "model_path is required"})
			return
		}
		record, err := s.registry.Register(request.ModelPath)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, response{Error: err.Error()})
			return
		}
		s.metrics.IncModelRegistration()
		writeJSON(w, http.StatusCreated, struct {
			Model orchestrator.ModelRecord `json:"model"`
		}{Model: record})
	default:
		w.Header().Set("Allow", "GET, POST")
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
	}
}

func (s *Server) handleHardwareCompatibility(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	prof := orchestrator.DetectHostProfile()
	eval := orchestrator.HardwareEvaluator{Profile: prof}
	// Evaluate baseline LLaMA-7B Q4_0
	est := orchestrator.EstimateModelResources(32000, 4096, 32, 32, 32, 128, 11008, "Q4_0", false)
	assessment := eval.Evaluate(est, 2048, 1)

	writeJSON(w, http.StatusOK, struct {
		Hardware   orchestrator.HostHardwareProfile     `json:"hardware"`
		Assessment orchestrator.CompatibilityAssessment `json:"assessment"`
	}{
		Hardware:   prof,
		Assessment: assessment,
	})
}

func (s *Server) handleBenchmark(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	result := orchestrator.RunSyntheticBenchmark()
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	stats := s.controller.Stats()
	if err := s.metrics.Render(w, stats.Active); err != nil {
		return
	}
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) bool {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBodyBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeJSON(w, http.StatusBadRequest, response{Error: "invalid JSON request"})
		return false
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		writeJSON(w, http.StatusBadRequest, response{Error: "request must contain one JSON object"})
		return false
	}
	return true
}

func writeControlError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, orchestrator.ErrEngineBusy):
		writeJSON(w, http.StatusConflict, response{Error: err.Error()})
	case errors.Is(err, orchestrator.ErrEmptyModelPath):
		writeJSON(w, http.StatusBadRequest, response{Error: err.Error()})
	default:
		writeJSON(w, http.StatusServiceUnavailable, response{Error: err.Error()})
	}
}

func writeEngineError(w http.ResponseWriter, err error) {
	status := http.StatusServiceUnavailable
	var nativeError cgo_bridge.StatusError
	if errors.As(err, &nativeError) {
		switch nativeError.Code {
		case int(CgoStatusAlreadyStarted):
			status = http.StatusConflict
		case int(CgoStatusInvalidArgument):
			status = http.StatusBadRequest
		case int(CgoStatusRamThrottled):
			status = http.StatusTooManyRequests
		case int(CgoStatusSsdReserve):
			status = http.StatusInsufficientStorage
		case int(CgoStatusNotStarted):
			status = http.StatusConflict
		case int(CgoStatusLicenseRejected):
			status = http.StatusForbidden
		}
		writeJSON(w, status, response{Error: err.Error(), Code: nativeError.Code})
		return
	}
	writeJSON(w, status, response{Error: err.Error()})
}

// These aliases keep the HTTP package independent of cgo-generated C names.
const (
	CgoStatusInvalidArgument = -1
	CgoStatusAlreadyStarted  = -2
	CgoStatusRamThrottled    = -3
	CgoStatusSsdReserve      = -4
	CgoStatusNotStarted      = -7
	CgoStatusLicenseRejected = -8
)

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("X-Aabhas-API-Version", "v1")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func (s *Server) handleTokens(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		var req tokenGenerateRequest
		if !decodeJSON(w, r, &req) {
			return
		}
		raw, record, err := s.tokens.GenerateToken(req.Name, req.Role, time.Duration(req.TTLSeconds)*time.Second)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, response{Error: err.Error()})
			return
		}
		writeJSON(w, http.StatusCreated, tokenResponse{Token: raw, Record: record})

	case http.MethodGet:
		tokens := s.tokens.ListTokens()
		sanitized := make([]auth.TokenRecord, len(tokens))
		for i, t := range tokens {
			sanitized[i] = t
			sanitized[i].TokenHash = "[redacted]"
		}
		writeJSON(w, http.StatusOK, tokenListResponse{Tokens: sanitized})

	case http.MethodDelete:
		var req tokenRevokeRequest
		if !decodeJSON(w, r, &req) {
			return
		}
		if err := s.tokens.Revoke(req.TokenID); err != nil {
			writeJSON(w, http.StatusNotFound, response{Error: err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, response{Status: "token revoked"})

	default:
		w.Header().Set("Allow", "GET, POST, DELETE")
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
	}
}

func (s *Server) handleLicenseVerify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	var req licenseVerifyRequest
	if !decodeJSON(w, r, &req) {
		return
	}

	targetUUID := req.TargetHardwareUUID
	if targetUUID == "" {
		targetUUID = os.Getenv("AABHAS_HARDWARE_UUID")
	}
	if targetUUID == "" {
		targetUUID = req.License.LicensedHardwareUUID
	}

	err := license.ValidateLicense(&req.License, targetUUID, nil)
	resp := licenseVerifyResponse{
		Valid:          err == nil,
		CustomerName:   req.License.Payload.CustomerName,
		Tier:           string(req.License.Payload.Tier),
		HardwareBound:  req.License.LicensedHardwareUUID,
		ExpirationDate: req.License.Payload.ExpiresAt.Format(time.RFC3339),
	}
	if err != nil {
		resp.Error = err.Error()
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleModelVerify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
		return
	}
	var req modelVerifyRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.ModelPath) == "" {
		writeJSON(w, http.StatusBadRequest, response{Error: "model_path is required"})
		return
	}

	sha, size, err := orchestrator.ComputeFileSHA256(req.ModelPath)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, modelVerifyResponse{
			Valid:     false,
			ModelPath: req.ModelPath,
			Error:     err.Error(),
		})
		return
	}

	valid := true
	if req.ExpectedSHA256 != "" {
		valid = subtle.ConstantTimeCompare([]byte(sha), []byte(strings.ToLower(req.ExpectedSHA256))) == 1
	}

	resp := modelVerifyResponse{
		Valid:          valid,
		ModelPath:      req.ModelPath,
		ComputedSHA256: sha,
		FileSizeBytes:  size,
	}
	if !valid {
		resp.Error = "computed sha256 does not match expected sha256"
	}
	writeJSON(w, http.StatusOK, resp)
}

// TokenManager returns the active token manager for programmatic administration.
func (s *Server) TokenManager() *auth.TokenManager {
	return s.tokens
}
