# Kubernetes Deployment

Kubernetes manifests for deploying the IISA (Indexing Indexer Selection Algorithm) service.

## Files

- `deployment.yaml` - IISA deployment with health probes and resource limits
- `service.yaml` - Internal ClusterIP service on port 8080
- `networkpolicy.yaml` - Restricts ingress to dipper pods only
- `configmap-example.yaml` - Example ConfigMap for non-sensitive config

## Resource Requirements

IISA has two operational modes with different resource needs:

| Mode | CPU | Memory | Frequency |
|------|-----|--------|-----------|
| Inference | Low | ~1GB | Per-request |
| Training | High | 32GB+ | Daily |

The deployment is configured with limits to accommodate the daily training job:
- Memory limit: 40Gi
- CPU limit: 4 cores

## Startup Behavior

IISA loads data from BigQuery on startup, which can take several minutes. The probes are configured to account for this:

- Startup probe: 30s initial delay, up to 5 minutes total (30 retries x 10s)
- Readiness probe: 90s initial delay
- Liveness probe: 120s initial delay

## Prerequisites

Before deploying IISA:

1. Create GCP service account credentials secret:
   ```bash
   kubectl create secret generic iisa-gcp-credentials \
     --from-file=service-account.json=path/to/credentials.json
   ```

2. Create ConfigMap with GCP project info:
   ```bash
   kubectl create configmap iisa-config \
     --from-literal=gcp_project=your-project-id \
     --from-literal=gcp_location=US
   ```

3. Create secret for IPinfo API key:
   ```bash
   kubectl create secret generic iisa-secrets \
     --from-literal=ipinfo_api_key=your-api-key
   ```

## Network Policy

The NetworkPolicy restricts ingress traffic to IISA:
- Only pods with label `app: dipper` can access IISA
- All other ingress is denied

Ensure dipper is deployed with the label `app: dipper` for service discovery to work.

## Service Discovery

Dipper should configure IISA endpoint as:
```
http://iisa:8080
```

This uses Kubernetes DNS to resolve the service within the same namespace.
