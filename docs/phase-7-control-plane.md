# Phase 7: Operational Control Plane

## Objective

Expand the Go daemon from a start/stop wrapper into a versioned operational
control surface while preserving the single-model execution invariant.

## Delivered contract

1. `GET /api/v1/engine/status` exposes running state, active model path, and
   lifecycle counters.
2. `POST /api/v1/models` validates and registers readable non-empty model files;
   `GET /api/v1/models` returns deterministic metadata-only records.
3. Start requests acquire the Go execution lease before crossing the native
   boundary and cancel the lease if native startup fails.
4. Request cancellation is checked before and after native startup; canceled
   successful starts are stopped and rolled back.
5. Start and stop transitions are serialized in the control plane so a stop
   cannot race a still-pending native start.
6. JSON responses carry `X-Aabhas-API-Version: v1`, and method/error handling is
   JSON-shaped and bounded consistently.
7. Orchestrator tests cover registry validation and lifecycle lease accounting.
8. The CLI exposes the new engine status operation.

## Explicit non-goals

Authentication, authorization, metrics export, and multi-model scheduling are
later hardening phases. Phase 8 adds optional bearer-token authentication; the
API remains loopback-oriented by default when no token is configured.
