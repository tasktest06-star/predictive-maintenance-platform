# Product 6: CMMS & Monitoring Dashboard
> Web application and mobile app for maintenance teams

---

## Product Overview

The user-facing application combining real-time condition monitoring visualization with computerized maintenance management (CMMS). Serves reliability engineers (who need spectrum analysis and fault diagnosis), maintenance technicians (who need work orders and guided repair), and plant managers (who need KPI dashboards and OEE tracking).

---

## Functional Requirements

### FR-DASH-01: Real-Time Spectrum Display
- Time-waveform display: X/Y/Z axes simultaneously, auto-scaling
- FFT spectrum display: log frequency scale, dB amplitude, up to 64 kHz
- Bearing defect frequency markers: BPFO, BPFI, BSF, FTF automatically calculated from bearing geometry + current RPM
- Running harmonics: 1×, 2×, 3× running speed overlaid on spectrum
- Waterfall / cascade plot: 3D time-frequency evolution view (2-week history)
- Octave band analysis (ISO 1/3 octave)
- Zoom, pan, cursor readout (frequency + amplitude at cursor)
- Comparison mode: overlay healthy vs current spectrum

### FR-DASH-02: Asset Health Dashboard
- Fleet overview: all assets on a site map / floor plan with color-coded health status (green/yellow/orange/red)
- Asset detail page: current health score, anomaly trend chart, active alerts, diagnostic events, sensor readings
- Health score history: 30/90/365-day trend chart
- Health KPI tiles: uptime%, mean time between alerts, mean time to repair
- Severity heatmap: assets × time grid showing fault stage history

### FR-DASH-03: Alert Management
- Alert feed: sorted by severity, filterable by site/area/asset/date/status
- Alert detail: timestamp, asset, fault class, confidence, severity stage, feature attribution chart (SHAP)
- Acknowledge / resolve workflow with required notes
- Alert assignment to technician
- Alert escalation rules configuration
- Notification preferences: email, SMS, push notification, Slack

### FR-DASH-04: Work Order Management
- Work order list: filterable by status (open/in-progress/complete), priority, assignee, due date
- Work order detail: asset, fault, LLM-generated instructions, steps checklist, parts list, time log, photos
- Create work order (manual or auto from alert)
- Technician mobile view: simplified, touch-friendly, offline-capable
- Labor and parts tracking
- Work order history per asset

### FR-DASH-05: Preventive Maintenance Scheduling
- Maintenance schedule calendar: by asset, by technician, by area
- Recurring maintenance tasks: interval (days/hours runtime/starts)
- PM due date calculation from asset runtime hours
- Schedule deviation tracking

### FR-DASH-06: Parts Inventory Management
- Spare parts list: part number, description, quantity on hand, reorder point, supplier
- Auto-suggest parts from work order (matched from bearing database)
- Low-stock alerts
- Purchase order creation

### FR-DASH-07: Reporting and KPIs
- OEE (Overall Equipment Effectiveness) continuous calculation and trend
- MTBF / MTTR per asset and asset class
- Maintenance cost per asset (labor + parts)
- Predictive savings: estimated cost avoidance from prevented failures
- Downloadable PDF maintenance reports
- Power BI connector (via API) for custom corporate reporting

### FR-DASH-08: Mobile App
- iOS and Android native apps
- Offline-capable: work orders, asset details, and last-known readings available offline
- QR code / NFC scan for asset identification in the field
- Photo capture and annotation for work order documentation
- BLE connectivity to sensor for local signal quality check during installation

---

## Non-Functional Requirements

### NFR-DASH-01: Performance
- Initial page load: <3 seconds on 10 Mbps connection
- Spectrum refresh: <500 ms for new measurement display
- Fleet overview (1,000 assets): <2 second load time
- Mobile app startup: <2 seconds

### NFR-DASH-02: Usability
- Accessibility: WCAG 2.1 AA
- Responsive design: works on desktop (1440px), tablet (768px), mobile (375px)
- Dark mode support (industrial control room environments)
- Localization: English, Spanish, Portuguese, German (initial)

### NFR-DASH-03: Offline Support
- Mobile: work orders and asset readings available for 24h without connectivity
- Service worker caching for web app (dashboard readable offline)

---

## Technical Architecture

```
Web App (React + TypeScript)
  → Apache ECharts (spectrum, trends, heatmaps)
  → React Query (API data fetching + cache)
  → Socket.IO (real-time alert feed + spectrum stream)
  → Tailwind CSS

Mobile App (React Native)
  → Shared business logic with web
  → Expo for native BLE and camera access
  → SQLite offline storage (WatermelonDB)

API Backend (Product 5)
  → REST JSON for CRUD operations
  → WebSocket / SSE for real-time updates

Data Visualization Stack:
  Apache ECharts — spectrum, waterfall, heatmap, trend lines
  D3.js — custom bearing frequency marker overlays
  Apache Superset — manager-facing KPI dashboards (optional)
```

### Key UI Components

**Spectrum Viewer** (most complex component):
- Canvas-based rendering (not SVG) for performance at 64K frequency bins
- Web Workers for FFT computation on imported raw waveform data
- Bearing frequency overlay computed client-side from geometry + RPM
- Keyboard shortcuts: +/- zoom, arrow keys pan, C for cursor mode
- Implemented with ECharts custom series + canvas layer

**Fleet Map:**
- SVG floor plan with asset pins (react-svg-pan-zoom)
- Color-coded by health score; pulsing animation for active critical alerts
- Cluster pins when zoomed out (Leaflet-style)

**Work Order Mobile View:**
- Bottom tab navigation (Work Orders / Assets / Notifications / Profile)
- Work order card with swipe-to-acknowledge gesture
- Camera integration for photo documentation
- Offline sync queue with conflict resolution

---

## Implementation Phases

### Phase 0: Design System (Month 1)
- [ ] Component library: color tokens, typography, spacing (Tailwind CSS)
- [ ] Dark mode theme for industrial environments
- [ ] Figma mockups for core screens: fleet overview, asset detail, spectrum viewer, work order

### Phase 1: Core Web App (Months 2–7)
- [ ] Asset hierarchy browser: Org → Site → Area → Asset
- [ ] Real-time spectrum display (ECharts + WebSocket streaming)
- [ ] Alert feed with acknowledge/resolve workflow
- [ ] Work order management (create, update, close)
- [ ] User management + RBAC

### Phase 2: Advanced Monitoring (Months 8–11)
- [ ] Waterfall / cascade plot
- [ ] Bearing frequency auto-overlay
- [ ] SHAP feature attribution chart on diagnostic events
- [ ] OEE and MTBF/MTTR dashboards
- [ ] Preventive maintenance scheduling calendar

### Phase 3: Mobile App (Months 10–14)
- [ ] React Native app: work orders + asset detail + alerts
- [ ] Offline sync (WatermelonDB)
- [ ] QR/NFC asset scan
- [ ] BLE sensor commissioning flow
- [ ] Photo documentation in work orders

### Phase 4: Advanced Features (Months 15–18)
- [ ] Fleet map with SVG floor plan import
- [ ] Power BI / Superset integration
- [ ] Localization: Spanish, Portuguese, German
- [ ] Accessibility audit (WCAG 2.1 AA)

---

## Key GitHub Repos

| Repo | Use |
|---|---|
| [apache/echarts](https://github.com/apache/echarts) | Spectrum display, waterfall, trend charts, heatmaps |
| [apache/superset](https://github.com/apache/superset) | Manager KPI dashboards (optional BI layer) |
| [thingsboard/thingsboard](https://github.com/thingsboard/thingsboard) | Alternative: IoT-native dashboard (faster to deploy) |

---

## Key Academic Papers

None directly applicable to UI/UX — reference TRACTIAN case study ROI figures (ICL 41% OEE, Whirlpool $1M+) to inform which KPIs to prioritize in dashboard design.

---

## Success Metrics

| Metric | Target |
|---|---|
| Page load time | <3 seconds |
| Spectrum refresh latency | <500 ms |
| Mobile offline capability | 24 hours |
| Work order completion rate | Track as product KPI |
| WCAG compliance | 2.1 AA |
| Technician NPS | >40 |
| Time to create work order | <2 minutes |
