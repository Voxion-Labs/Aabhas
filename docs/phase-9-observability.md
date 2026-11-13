# Phase 9: Observability

Phase 9 adds operational visibility to the Go control plane without adding a
runtime dependency.

## Metrics

`GET /metrics` exposes Prometheus text format. The registry includes request
counts by method, normalized route, and status; request duration buckets;
authentication failures; engine start/stop outcomes; model registrations; and
the current engine-active gauge.

Request paths are normalized to the fixed API route set plus `/other`. This
prevents model paths, identifiers, or arbitrary URLs from creating unbounded
label cardinality. The metrics endpoint follows the same optional bearer
authentication policy as the API.

## Request correlation

Every HTTP response receives an `X-Request-ID` value such as `req-42`. The
server logs one structured key/value line containing only the request ID,
method, route, status, and duration. Request bodies, model paths, bearer
tokens, and native error payloads are not included in this access log.

## Scope and operation

The metrics registry is process-local and guarded by a mutex. It uses only the
Go standard library, so Phase 9 does not require downloading or installing a
package. Scrapers should treat the counters as resettable on process restart.
