package observability

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestMetricsRenderUsesBoundedLabelsAndCumulativeBuckets(t *testing.T) {
	metrics := NewMetrics()
	metrics.ObserveRequest("GET", "/api/v1/engine/status", httpStatusOK, 20*time.Millisecond)
	metrics.ObserveRequest("DELETE", "/unbounded/user/path", 401, 2*time.Second)
	metrics.IncEngineStart()
	metrics.IncEngineStartFailure()
	metrics.IncEngineStop()
	metrics.IncEngineStopFailure()
	metrics.IncModelRegistration()

	var output bytes.Buffer
	if err := metrics.Render(&output, true); err != nil {
		t.Fatalf("Render() error = %v", err)
	}
	text := output.String()
	for _, expected := range []string{
		`aabhas_http_requests_total{method="GET",path="/api/v1/engine/status",status="200"} 1`,
		`aabhas_http_requests_total{method="OTHER",path="/other",status="401"} 1`,
		`aabhas_http_request_duration_seconds_bucket{le="0.05"} 1`,
		`aabhas_http_request_duration_seconds_bucket{le="+Inf"} 2`,
		`aabhas_auth_failures_total 1`,
		`aabhas_engine_active 1`,
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("metrics output missing %q:\n%s", expected, text)
		}
	}
}

func TestMetricsNilReceiverIsSafe(t *testing.T) {
	var metrics *Metrics
	metrics.ObserveRequest("GET", "/metrics", 200, time.Millisecond)
	metrics.IncEngineStart()
	if err := metrics.Render(&bytes.Buffer{}, false); err != nil {
		t.Fatalf("Render() error = %v", err)
	}
}

const httpStatusOK = 200
