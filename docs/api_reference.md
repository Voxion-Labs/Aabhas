# Aabhas Control-Plane API Reference

## Service

The Aabhas daemon exposes a loopback-oriented HTTP API for model lifecycle
control. The default listen address is `127.0.0.1:8080`; deployments may set
`AABHAS_LISTEN_ADDR` to an explicit bind address.

The API is intentionally small. It accepts control metadata only and never
transfers tensor payloads over HTTP.

Base URL:

```text
http://127.0.0.1:8080
```

Content type for requests with a body:

```text
application/json
```

If `AABHAS_API_TOKEN` is configured, every API request must include
`Authorization: Bearer <token>`. Keep the token in a deployment secret; it is
never returned by the API.

All responses include an `X-Request-ID` correlation header. The daemon logs
request ID, method, route, status, and duration only; request bodies and
credentials are not logged.

The unauthenticated `/healthz` and `/readyz` endpoints are reserved for
orchestrator probes. They return only process/readiness status and do not
expose model metadata or credentials.

## Start the engine

### Request

```http
POST /api/v1/engine/start HTTP/1.1
Host: 127.0.0.1:8080
Content-Type: application/json

{
  "model_path": "/models/example/model.bin"
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model_path` | string | yes | Absolute or deployment-local path to the model file. The path must not contain a NUL byte. |

The request body is limited to 1 MiB and unknown JSON fields are rejected.
Model paths are limited to 4096 bytes by the native boundary.

### Success response

Status: `200 OK`

```json
{
  "status": "started"
}
```

### Example

```bash
curl --fail-with-body --silent --show-error \
  -X POST 'http://127.0.0.1:8080/api/v1/engine/start' \
  -H 'Content-Type: application/json' \
  --data '{"model_path":"/models/example/model.bin"}'
```

## Stop the engine

### Request

```http
POST /api/v1/engine/stop HTTP/1.1
Host: 127.0.0.1:8080

```

The request has no JSON body. The daemon releases the active native engine
state and returns after the C ABI stop call completes.

### Success response

Status: `200 OK`

```json
{
  "status": "stopped"
}
```

### Example

```bash
curl --fail-with-body --silent --show-error \
  -X POST 'http://127.0.0.1:8080/api/v1/engine/stop'
```

## Engine status

### Request

```http
GET /api/v1/engine/status HTTP/1.1
Host: 127.0.0.1:8080
```

The response reports whether the single-model execution lease is running and
includes lifecycle counters. Responses carry `X-Aabhas-API-Version: v1`.

## Model registry

Register a readable, non-empty model file:

```http
POST /api/v1/models HTTP/1.1
Host: 127.0.0.1:8080
Content-Type: application/json

{"model_path":"/models/example/model.bin"}
```

`GET /api/v1/models` returns deterministic metadata records. The registry
stores paths and file metadata only; it never copies model contents.

## Metrics

```http
GET /metrics HTTP/1.1
Host: 127.0.0.1:8080
```

The endpoint returns Prometheus text exposition data for request counts and
latencies, authentication failures, lifecycle outcomes, model registrations,
and the current engine-active gauge. Route labels are normalized to a bounded
set, and the endpoint uses the same optional bearer authentication as the
control API.

## Error responses

Error responses use the following shape:

```json
{
  "error": "native engine returned status -2",
  "code": -2
}
```

`code` is present when the failure was returned by the native safety layer.

| HTTP status | Native code | Meaning |
| ---: | ---: | --- |
| `400 Bad Request` | `-1` | Invalid JSON or invalid model path. |
| `409 Conflict` | `-2` | A model is already active. |
| `400 Bad Request` | — | The request body was malformed or exceeded the configured limit. |
| `429 Too Many Requests` | `-3` | Host RAM is at or above the strict 70% admission threshold. |
| `507 Insufficient Storage` | `-4` | The model filesystem has less than 20 GiB available. |
| `503 Service Unavailable` | `-5` or `-6` | Native health check failure or recovered internal panic. |
| `403 Forbidden` | `-8` | Configured enterprise license or hardware binding was rejected. |
| `405 Method Not Allowed` | — | The endpoint requires `POST`. |

An idle stop is intentionally idempotent in the Go bridge and returns the same
`200` stopped response. Other error responses are safe to log as control
metadata. Model contents are not included in the response.

## CLI

The standard-library CLI uses the same endpoints:

```bash
./bin/aabhas-cli start \
  -addr 127.0.0.1:8080 \
  -model /models/example/model.bin

./bin/aabhas-cli stop -addr 127.0.0.1:8080

./bin/aabhas-cli status -addr 127.0.0.1:8080
```

The CLI returns a non-zero exit code for transport failures, malformed
responses, and non-2xx HTTP responses. `-timeout` accepts a Go duration such as
`5s` or `30s`.

## Operational contract

* Only one model may be active in a daemon process.
* Start and stop calls are serialized at the native boundary.
* The Rust safety layer performs RAM and filesystem admission checks before a
  start succeeds.
* SIGINT and SIGTERM cause the daemon to call the native stop function during
  shutdown.
* Without `AABHAS_API_TOKEN`, the API is not an authentication layer. Bind it
  to loopback or place it behind an authenticated, mutually trusted proxy.

## Compatibility

The endpoint paths and JSON field names are versioned by the `/api/v1` prefix.
Clients should ignore unknown response fields and should treat non-2xx status
codes as failed operations.
