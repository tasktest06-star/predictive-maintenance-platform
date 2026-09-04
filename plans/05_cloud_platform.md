# Product 5: Cloud Platform & API
> Multi-tenant backend, REST/gRPC API, integrations layer

---

## Product Overview

The cloud platform provides the multi-tenant backend that ties all products together: asset management, user authentication, alert management, third-party integrations (SAP, Maximo, Oracle), and the public API consumed by the dashboard and mobile apps. Designed for FedRAMP High authorization and ISO 27001 compliance.

---

## Functional Requirements

### FR-API-01: Asset Management
- CRUD operations for: Organizations, Sites, Areas, Assets, Sensor Installations
- Asset hierarchy: Org → Site → Area → Asset → Sensor
- Asset metadata: name, type, criticality (1–5), bearing geometry, nominal RPM, manufacturer, model, install date
- Bearing geometry database: 10,000+ bearing models (SKF, FAG, NSK, Timken); auto-import from SKF/Schaeffler APIs
- Asset tagging and custom attributes

### FR-API-02: User Management and RBAC
- Roles: Admin, Reliability Engineer, Technician, Requester, Viewer
- Role-based access: Technicians see only assigned work orders; Engineers see all alerts; Admins see everything
- SSO: SAML 2.0 and OAuth 2.0 / OIDC (Okta, Azure AD, Google Workspace)
- MFA required for Admin and Engineer roles
- Audit log: all API writes logged with user, timestamp, changed fields

### FR-API-03: Condition Monitoring API
- `GET /assets/{id}/health` — current health score, anomaly score, active alerts
- `GET /assets/{id}/metrics?from=&to=&resolution=` — time-series metric values (PromQL compatible)
- `GET /assets/{id}/spectrum?timestamp=` — FFT spectrum for given measurement
- `GET /assets/{id}/waveform?timestamp=` — raw time-domain waveform download URL
- `GET /alerts?site=&severity=&status=&from=` — alert list with filtering
- `POST /alerts/{id}/acknowledge` — acknowledge alert
- `POST /alerts/{id}/resolve` — mark resolved with resolution notes

### FR-API-04: Diagnostic Events API
- `GET /diagnostics/{asset_id}` — list of diagnosed fault events
- Each event: fault_class, confidence, severity_stage, rul_days, feature_attributions, work_instruction
- `POST /diagnostics/{event_id}/feedback` — technician confirms or disputes diagnosis (feedback for model retraining)

### FR-API-05: CMMS Work Orders API
- `POST /workorders` — create work order (manually or auto from diagnostic event)
- `GET /workorders/{id}` — retrieve with status, assignee, steps, parts, time log
- `PUT /workorders/{id}` — update status, add notes, log labor hours
- Auto-creation on P1/P2 alert with pre-populated instructions from LLM

### FR-API-06: Third-Party Integrations
- **SAP PM** (Plant Maintenance): bi-directional sync of functional locations, equipment records, notifications, work orders
- **IBM Maximo**: equipment, work order, failure history sync
- **Oracle Cloud/NetSuite**: asset registry sync
- **Webhook**: configurable outbound webhooks on alert/diagnostic events
- **Power BI**: published OData feed for custom reporting
- **Slack / Microsoft Teams**: alert notification bot

### FR-API-07: Multi-Tenancy
- Tenant data isolation: separate Kafka consumer groups, VictoriaMetrics tenants, PostgreSQL schemas
- Tenant-level resource quotas: max devices, API rate limits, storage retention
- White-label support: custom domain, logo, color scheme per tenant

### FR-API-08: Billing and Usage
- Usage metering: sensor count × active days + API call count
- Stripe billing integration
- Monthly usage report per tenant

---

## Non-Functional Requirements

### NFR-API-01: Performance
- API p99 response time: <200 ms for all read endpoints
- Time-series query (1-year range, hourly resolution): <1 second
- Waveform download (256 KB): <2 seconds

### NFR-API-02: Availability
- SLA: 99.9% API uptime
- Multi-AZ deployment
- Circuit breakers on all external integrations

### NFR-API-03: Security and Compliance
- FedRAMP High control baseline (for US government customers)
- ISO 27001 / SOC 2 Type II
- GDPR / CCPA data subject rights: export and deletion
- TLS 1.3 everywhere; HSTS; certificate pinning in mobile app
- WAF (Web Application Firewall) in front of API gateway
- Secrets management: HashiCorp Vault or AWS Secrets Manager

### NFR-API-04: API Design
- REST JSON API (OpenAPI 3.0 spec)
- gRPC for high-throughput internal services (ML engine, pipeline)
- API versioning: `/v1/`, `/v2/` with 2-version deprecation policy
- Rate limiting: 1,000 req/min per API key (configurable)

---

## Technical Architecture

```
                   API Gateway (Kong / AWS API GW)
                          ↓ Auth (JWT validation)
            ┌─────────────────────────────────┐
            │  Backend Services (Go / Python)  │
            │  asset-service                   │
            │  alert-service                   │
            │  diagnostic-service              │
            │  workorder-service               │
            │  integration-service             │
            │  user-service                    │
            └─────────────┬───────────────────┘
                          ↓ gRPC / SQL
          ┌───────────────────────────────────┐
          │  Data Stores                       │
          │  PostgreSQL (asset metadata)       │
          │  VictoriaMetrics (metrics/time-    │
          │    series via Flink pipeline)      │
          │  S3 (waveforms, reports)           │
          │  Redis (session cache, rate limit) │
          └───────────────────────────────────┘
```

### Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| API language | Go (primary services) | Low latency, small memory footprint, strong concurrency |
| ML service | Python (FastAPI) | PyTorch ecosystem; model serving |
| API Gateway | Kong (Apache 2.0) | Plugin ecosystem; rate limiting; auth; OpenTelemetry |
| Auth | Keycloak (Apache 2.0) | SSO, SAML, OIDC, MFA; self-hosted |
| Database | PostgreSQL | ACID; JSONB for metadata; full-text search |
| Cache | Redis | Session store; rate limit counters; real-time dashboard pub/sub |
| Container | Docker + Kubernetes | Standard; Helm charts for deployment |
| CI/CD | ArgoCD (Apache 2.0) | GitOps; Kubernetes-native CD |
| Secrets | HashiCorp Vault (MPL) | Secrets management; certificate authority |

---

## Implementation Phases

### Phase 0: Foundation (Months 1–2)
- [ ] PostgreSQL schema: organizations, sites, assets, sensors, users, roles
- [ ] Keycloak setup: RBAC roles, JWT tokens
- [ ] Kong API gateway with JWT validation plugin
- [ ] Asset CRUD API (Go + Chi router)
- [ ] OpenAPI 3.0 spec

### Phase 1: Core APIs (Months 3–6)
- [ ] Condition monitoring API: health, metrics, spectrum, waveform download
- [ ] Alert API: create, acknowledge, resolve, list with filters
- [ ] Work order API: create, update, list
- [ ] Webhook notification system
- [ ] Multi-tenant data isolation

### Phase 2: Integrations (Months 7–11)
- [ ] SAP PM connector (RFC/BAPI calls via SAP JCo)
- [ ] IBM Maximo REST connector
- [ ] Slack / Teams bot for alert notifications
- [ ] Power BI OData feed
- [ ] OAuth 2.0 / OIDC SSO (Okta, Azure AD)

### Phase 3: Compliance (Months 12–18)
- [ ] FedRAMP authorization documentation and control implementation
- [ ] SOC 2 Type II audit preparation
- [ ] GDPR data subject request workflows
- [ ] WAF deployment and DDoS protection
- [ ] Penetration test remediation

---

## Key GitHub Repos

| Repo | Use |
|---|---|
| [thingsboard/thingsboard](https://github.com/thingsboard/thingsboard) | Reference: device management patterns |
| [open-telemetry/opentelemetry-collector](https://github.com/open-telemetry/opentelemetry-collector) | API observability telemetry |
| [apache/airflow](https://github.com/apache/airflow) | Integration sync job scheduling |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | ML model serving API |

---

## Success Metrics

| Metric | Target |
|---|---|
| API p99 latency | <200 ms |
| API availability | 99.9% |
| Time-series query (1yr/1h) | <1 second |
| FedRAMP authorization | Achieved by Month 18 |
| SAP integration roundtrip | <5 seconds |
| New tenant onboarding | <15 minutes automated |
