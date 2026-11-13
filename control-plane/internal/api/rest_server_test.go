package api

import (
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHealthEndpointsArePublicAndCorrelated(t *testing.T) {
	t.Setenv("AABHAS_API_TOKEN", "phase12-secret")
	server := NewServer("127.0.0.1:0", log.New(io.Discard, "", 0))

	for _, path := range []string{"/healthz", "/readyz"} {
		recorder := httptest.NewRecorder()
		server.httpServer.Handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))

		if recorder.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d, want %d", path, recorder.Code, http.StatusOK)
		}
		if !strings.HasPrefix(recorder.Header().Get("X-Request-ID"), "req-") {
			t.Fatalf("GET %s missing request ID", path)
		}
	}
}

func TestProtectedMetricsRequireConfiguredBearerToken(t *testing.T) {
	t.Setenv("AABHAS_API_TOKEN", "phase12-secret")
	server := NewServer("127.0.0.1:0", log.New(io.Discard, "", 0))

	unauthorized := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized metrics status = %d, want %d", unauthorized.Code, http.StatusUnauthorized)
	}

	authorizedRequest := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	authorizedRequest.Header.Set("Authorization", "Bearer phase12-secret")
	authorized := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(authorized, authorizedRequest)
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized metrics status = %d, want %d", authorized.Code, http.StatusOK)
	}
	if !strings.Contains(authorized.Body.String(), "aabhas_engine_active") {
		t.Fatal("authorized metrics response omitted engine gauge")
	}
}
