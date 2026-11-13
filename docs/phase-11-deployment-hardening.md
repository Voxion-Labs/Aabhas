# Phase 11: Deployment Hardening

Phase 11 makes container orchestration health-aware and safer by default.

## Health contract

The daemon exposes two unauthenticated, non-sensitive probe endpoints:

- `GET /healthz` confirms that the process is serving HTTP;
- `GET /readyz` confirms that the daemon is ready to receive control requests.

The endpoints return only a small status object and remain outside bearer-token
authentication so orchestrators can probe a deployment without receiving a
credential. Metrics and all control endpoints remain subject to the configured
API token.

## Container and Kubernetes behavior

The production image now includes the standard-library `aabhas-cli` health
command and declares a Docker `HEALTHCHECK`. Compose uses the same check and
passes `AABHAS_API_TOKEN` through environment substitution rather than baking a
token into the file.

The Kubernetes deployment uses HTTP startup, liveness, and readiness probes.
Its API-auth secret is required (`aabhas-api-auth`, key `token`); operators
should create that secret before applying the deployment. The workload remains
non-root, read-only-rootfs, capability-dropped, and service-account-token-free.
