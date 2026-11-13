// Package api exposes the control-plane HTTP API.
package api

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/Voxion-Labs/Aabhas/control-plane/internal/cgo_bridge"
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

// Server owns the HTTP server and its engine lifecycle handlers.
type Server struct {
	httpServer  *http.Server
	logger      *log.Logger
	controller  *orchestrator.ExecutionController
	registry    *orchestrator.ModelRegistry
	lifecycleMu sync.Mutex
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
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/engine/start", server.handleStart)
	mux.HandleFunc("/api/v1/engine/stop", server.handleStop)
	mux.HandleFunc("/api/v1/engine/status", server.handleStatus)
	mux.HandleFunc("/api/v1/models", server.handleModels)
	server.httpServer = &http.Server{
		Addr:              addr,
		Handler:           requireBearerToken(mux, os.Getenv("AABHAS_API_TOKEN")),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	return server
}

func requireBearerToken(next http.Handler, configured string) http.Handler {
	token := strings.TrimSpace(configured)
	if token == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		const prefix = "Bearer "
		authorization := r.Header.Get("Authorization")
		if len(authorization) <= len(prefix) || !strings.EqualFold(authorization[:len(prefix)], prefix) ||
			subtle.ConstantTimeCompare([]byte(authorization[len(prefix):]), []byte(token)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="aabhas"`)
			writeJSON(w, http.StatusUnauthorized, response{Error: "bearer authentication required"})
			return
		}
		next.ServeHTTP(w, r)
	})
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
		return
	}
	if strings.TrimSpace(request.ModelPath) == "" || strings.IndexByte(request.ModelPath, 0) >= 0 {
		writeJSON(w, http.StatusBadRequest, response{Error: "model_path is required"})
		return
	}
	if err := r.Context().Err(); err != nil {
		writeJSON(w, http.StatusRequestTimeout, response{Error: "request canceled"})
		return
	}

	lease, err := s.controller.Acquire(request.ModelPath)
	if err != nil {
		writeControlError(w, err)
		return
	}

	if err := cgo_bridge.Start(request.ModelPath); err != nil {
		_ = lease.Cancel()
		s.logger.Printf("engine start rejected: %v", err)
		writeEngineError(w, err)
		return
	}
	if err := r.Context().Err(); err != nil {
		_ = cgo_bridge.Stop()
		_ = lease.Cancel()
		writeJSON(w, http.StatusRequestTimeout, response{Error: "request canceled"})
		return
	}
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
		s.logger.Printf("engine stop failed: %v", err)
		writeEngineError(w, err)
		return
	}
	_ = s.controller.Release()
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
		writeJSON(w, http.StatusCreated, struct {
			Model orchestrator.ModelRecord `json:"model"`
		}{Model: record})
	default:
		w.Header().Set("Allow", "GET, POST")
		writeJSON(w, http.StatusMethodNotAllowed, response{Error: "method not allowed"})
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
