// Package api exposes the control-plane HTTP API.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/Voxion-Labs/Aabhas/control-plane/internal/cgo_bridge"
)

const maxStartBodyBytes = 1 << 20

type startRequest struct {
	ModelPath string `json:"model_path"`
}

type response struct {
	Status string `json:"status,omitempty"`
	Error  string `json:"error,omitempty"`
	Code   int    `json:"code,omitempty"`
}

// Server owns the HTTP server and its engine lifecycle handlers.
type Server struct {
	httpServer *http.Server
	logger     *log.Logger
}

// NewServer constructs a bounded-timeout REST server on addr.
func NewServer(addr string, logger *log.Logger) *Server {
	if logger == nil {
		logger = log.Default()
	}

	server := &Server{logger: logger}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/engine/start", server.handleStart)
	mux.HandleFunc("/api/v1/engine/stop", server.handleStop)
	server.httpServer = &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	return server
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
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request startRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxStartBodyBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		writeJSON(w, http.StatusBadRequest, response{Error: "invalid JSON request"})
		return
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		writeJSON(w, http.StatusBadRequest, response{Error: "request must contain one JSON object"})
		return
	}
	if strings.TrimSpace(request.ModelPath) == "" || strings.IndexByte(request.ModelPath, 0) >= 0 {
		writeJSON(w, http.StatusBadRequest, response{Error: "model_path is required"})
		return
	}

	if err := cgo_bridge.Start(request.ModelPath); err != nil {
		s.logger.Printf("engine start rejected: %v", err)
		writeEngineError(w, err)
		return
	}
	s.logger.Printf("engine started for model path %q", request.ModelPath)
	writeJSON(w, http.StatusOK, response{Status: "started"})
}

func (s *Server) handleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := cgo_bridge.Stop(); err != nil {
		s.logger.Printf("engine stop failed: %v", err)
		writeEngineError(w, err)
		return
	}
	s.logger.Printf("engine stopped")
	writeJSON(w, http.StatusOK, response{Status: "stopped"})
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
)

func writeJSON(w http.ResponseWriter, status int, value response) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
