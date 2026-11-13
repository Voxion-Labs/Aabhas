// Package observability provides low-cardinality, dependency-free telemetry.
package observability

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"
)

var requestBuckets = []float64{0.01, 0.05, 0.25, 1.0}

const requestBucketCount = 5

type requestKey struct {
	method string
	path   string
	status int
}

type histogram struct {
	buckets [requestBucketCount]uint64
	count   uint64
	sum     float64
}

// Metrics stores process-local counters for the control plane.
// Labels are normalized to a fixed route set to prevent cardinality growth.
type Metrics struct {
	mu sync.Mutex

	requests  map[requestKey]uint64
	durations histogram

	authFailures        uint64
	engineStarts        uint64
	engineStartFailures uint64
	engineStops         uint64
	engineStopFailures  uint64
	modelRegistrations  uint64
}

// NewMetrics constructs an empty metrics registry.
func NewMetrics() *Metrics {
	return &Metrics{requests: make(map[requestKey]uint64)}
}

// ObserveRequest records one completed HTTP request.
func (m *Metrics) ObserveRequest(method, path string, status int, duration time.Duration) {
	if m == nil {
		return
	}
	method = normalizeMethod(method)
	path = normalizePath(path)
	if status < 100 || status > 599 {
		status = 500
	}
	if duration < 0 {
		duration = 0
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.requests == nil {
		m.requests = make(map[requestKey]uint64)
	}
	m.requests[requestKey{method: method, path: path, status: status}]++
	seconds := duration.Seconds()
	m.durations.count++
	m.durations.sum += seconds
	for i, boundary := range requestBuckets {
		if seconds <= boundary {
			m.durations.buckets[i]++
		}
	}
	m.durations.buckets[len(requestBuckets)]++
	if status == 401 {
		m.authFailures++
	}
}

func (m *Metrics) IncEngineStart() {
	m.add(func() { m.engineStarts++ })
}

func (m *Metrics) IncEngineStartFailure() {
	m.add(func() { m.engineStartFailures++ })
}

func (m *Metrics) IncEngineStop() {
	m.add(func() { m.engineStops++ })
}

func (m *Metrics) IncEngineStopFailure() {
	m.add(func() { m.engineStopFailures++ })
}

func (m *Metrics) IncModelRegistration() {
	m.add(func() { m.modelRegistrations++ })
}

func (m *Metrics) add(update func()) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	update()
}

// Render writes Prometheus text exposition format.
func (m *Metrics) Render(w io.Writer, engineActive bool) error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	keys := make([]requestKey, 0, len(m.requests))
	for key := range m.requests {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].path != keys[j].path {
			return keys[i].path < keys[j].path
		}
		if keys[i].method != keys[j].method {
			return keys[i].method < keys[j].method
		}
		return keys[i].status < keys[j].status
	})

	write := func(format string, args ...any) error {
		_, err := fmt.Fprintf(w, format, args...)
		return err
	}
	if err := write("# HELP aabhas_http_requests_total Total HTTP requests handled by the control plane.\n# TYPE aabhas_http_requests_total counter\n"); err != nil {
		return err
	}
	for _, key := range keys {
		if err := write("aabhas_http_requests_total{method=\"%s\",path=\"%s\",status=\"%d\"} %d\n", escapeLabel(key.method), escapeLabel(key.path), key.status, m.requests[key]); err != nil {
			return err
		}
	}
	if err := write("# HELP aabhas_http_request_duration_seconds HTTP request duration in seconds.\n# TYPE aabhas_http_request_duration_seconds histogram\n"); err != nil {
		return err
	}
	for i, boundary := range requestBuckets {
		if err := write("aabhas_http_request_duration_seconds_bucket{le=\"%s\"} %d\n", formatFloat(boundary), m.durations.buckets[i]); err != nil {
			return err
		}
	}
	if err := write("aabhas_http_request_duration_seconds_bucket{le=\"+Inf\"} %d\naabhas_http_request_duration_seconds_sum %s\naabhas_http_request_duration_seconds_count %d\n", m.durations.buckets[len(requestBuckets)], formatFloat(m.durations.sum), m.durations.count); err != nil {
		return err
	}

	if err := write("# HELP aabhas_auth_failures_total HTTP authentication failures.\n# TYPE aabhas_auth_failures_total counter\naabhas_auth_failures_total %d\n", m.authFailures); err != nil {
		return err
	}
	if err := write("# HELP aabhas_engine_starts_total Successful engine starts.\n# TYPE aabhas_engine_starts_total counter\naabhas_engine_starts_total %d\n# HELP aabhas_engine_start_failures_total Failed engine starts.\n# TYPE aabhas_engine_start_failures_total counter\naabhas_engine_start_failures_total %d\n# HELP aabhas_engine_stops_total Successful engine stops.\n# TYPE aabhas_engine_stops_total counter\naabhas_engine_stops_total %d\n# HELP aabhas_engine_stop_failures_total Failed engine stops.\n# TYPE aabhas_engine_stop_failures_total counter\naabhas_engine_stop_failures_total %d\n# HELP aabhas_model_registrations_total Successful model registrations.\n# TYPE aabhas_model_registrations_total counter\naabhas_model_registrations_total %d\n", m.engineStarts, m.engineStartFailures, m.engineStops, m.engineStopFailures, m.modelRegistrations); err != nil {
		return err
	}
	active := 0
	if engineActive {
		active = 1
	}
	return write("# HELP aabhas_engine_active Whether the native engine is active.\n# TYPE aabhas_engine_active gauge\naabhas_engine_active %d\n", active)
}

func normalizeMethod(method string) string {
	switch method {
	case "GET", "POST":
		return method
	default:
		return "OTHER"
	}
}

func normalizePath(path string) string {
	switch path {
	case "/api/v1/engine/start", "/api/v1/engine/stop", "/api/v1/engine/status", "/api/v1/models", "/metrics":
		return path
	default:
		return "/other"
	}
}

func escapeLabel(value string) string {
	return strings.NewReplacer(`\\`, `\\\\`, `"`, `\\"`, "\n", `\n`).Replace(value)
}

func formatFloat(value float64) string {
	return fmt.Sprintf("%g", value)
}
