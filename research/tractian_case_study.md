# TRACTIAN — Predictive Maintenance Case Study
> Compiled September 2026 from tractian.com

---

## Company Profile

| Field | Detail |
|---|---|
| Founded by | Igor Marinelli (CEO), Gabriel Lameirinhas, Leonardo Vieira |
| HQ | 756 W Peachtree St NW, Atlanta, GA 30308 |
| Series C (Dec 2024) | $120M led by Sapphire Ventures |
| Customers | 1,500+ U.S. and global manufacturers |
| Notable customers | Bosch, Hyundai, Embraer, Cargill, Unilever, Whirlpool, Danone, Pirelli |
| Recognition | Forbes AI 50, Inc. 5000 #269 |
| Certifications | FedRAMP High, ISO 27001, ISO 9001 |
| Partnerships | SAP Silver Partner |

---

## Product Portfolio

### 1. Smart Trac Ultra — Wireless Condition Monitoring Sensor

The flagship hardware SKU. An all-in-one wireless sensor combining vibration, temperature, ultrasound, and RPM in a single IP69K housing.

**Hardware Specifications**

| Spec | Value |
|---|---|
| Axes | Triaxial (X, Y, Z) |
| Frequency range | 0–64,000 Hz |
| Sampling rates | 500 / 1,000 / 2,000 / 4,000 / 8,000 / 16,000 / 32,000 Hz (configurable interval) |
| FFT resolution | 196,608 lines per axis |
| Max acceleration | 60g |
| Max velocity | 100 mm/s RMS |
| RPM range | 1–48,000 RPM (magnetometer-based tachometer) |
| Battery life | 3–5 years (default settings) |
| IP rating | IP69K |
| Operating temperature | -40°C to 120°C |
| Wireless protocol | 915 MHz ISM, IEEE 802.15.4g |
| Backup connectivity | 4G/LTE (direct cloud, no gateway required) |
| Range (obstructed) | 100 m |
| Range (line of sight) | 1,000 m |
| Encryption | AES-256 |
| Dimensions | 40 × 71 × 40 mm |
| Weight (sensor) | 180 g |
| Certifications | ATEX, IECEx, NFPA 70 CL1/CL2/CL3 (Class I Div 1), FCC |
| Signal validation | ISO 16063-21 |
| Bandwidth (validated) | 10 kHz Z-axis ±3 dB; transverse to 6 kHz |

**Patented Mechanical Design**

A flexible PCB signal path and compliant polymer battery mount decouple heavy internal components from the measurement accelerometer. The accelerometer sits directly at the mounting tip, eliminating battery resonance artifacts that corrupt the vibration spectrum. Validated against a traceable reference accelerometer per ISO 16063-21. No pronounced resonant peaks; ±3 dB flat across validated bandwidth.

### 2. AutoDiagnosis™ Platform — AI Condition Monitoring Software

The cloud-based monitoring platform powered by machine learning trained on 3.5 billion samples from hundreds of thousands of assets worldwide.

**AI Pipeline (5 stages):**
1. **Sensing** — Continuous capture: vibration, ultrasound, temperature, RPM
2. **Interpretation** — ML models compare incoming signals to 3.5B-sample baseline across asset classes
3. **Contextualization** — Weighted against asset history, operating conditions, production criticality
4. **Translation** — LLMs convert technical findings into plain-language work instructions
5. **Action** — Auto-generates CMMS work orders with step-by-step procedures and parts lists

**Fusion strategy:** Both early fusion (raw multi-modal data merged) and late fusion (independent per-modality networks with final decision combiner). Late fusion provides fault detection continuity when individual sensors degrade.

**Output per fault event:** fault type + severity stage + root cause + criticality rank + plain-language work instruction + auto-generated CMMS work order.

### 3. CMMS Platform — Computerized Maintenance Management

Integrated maintenance management system bundled with the sensor platform.

**Features:** Work order management, preventive maintenance scheduling, parts inventory management, KPI dashboards, OEE continuous tracking, reliability and root cause analysis, technician mobile app (offline-capable), asset location and utilization monitoring.

**Integrations:** SAP (Silver Partner), IBM Maximo, Oracle NetSuite, Oracle Cloud, UpKeep, Power BI Connector (Enterprise), SSO (Enterprise), Excel/Sheets import.

### 4. Electrical Monitoring

Non-invasive current and voltage monitoring using Hall-effect sensors or Rogowski coils. Detects motor electrical faults, power factor drift, current/voltage anomalies. Priced and sold separately.

---

## Sensing Modalities

| Modality | Transducer | Fault Classes |
|---|---|---|
| Vibration | MEMS or piezoelectric accelerometer (triaxial) | Unbalance, misalignment, bearing defects, gearbox faults, looseness |
| Acoustic/Ultrasonic | Microphone (audible to low-ultrasonic) | Early-stage friction, cavitation, bearing surface defects |
| Temperature | Contact thermal + optional IR | Overheating, cooling failure, energy inefficiency |
| RPM | Magnetometer tachometer | Speed anomalies, variable-speed analysis |
| Electrical | Hall-effect / Rogowski coil | Motor electrical faults, power factor, voltage/current anomalies |

**Note:** TRACTIAN's "ultrasound" refers to microphone-captured sound in the audible-to-lower-ultrasonic range. It is not true acoustic emission (AE) sensing at 100–400 kHz piezoelectric transducers.

---

## Fault Modes Detected

**Vibration domain:**
- Mechanical unbalance (1× running speed peak)
- Shaft misalignment (2× or 3× peaks)
- Bearing wear — Stage 2 specifically cited (defect frequencies + high-frequency noise floor rise)
- Gearbox faults (gear mesh frequency anomalies; requires ≥10 kHz sampling)
- Lubrication failure (predictive lubrication feature)
- Belt fraying / loose belt
- Fan blade warping

**Acoustic domain:**
- Early-stage friction and wear
- Cavitation (pumps)
- Bearing surface defects before vibration registration

**Thermal domain:**
- Overheating / cooling failure
- Energy inefficiency signatures

**Electrical domain:**
- Motor electrical faults
- Current/voltage anomalies
- Power factor drift

---

## Industries Served

Automotive & Parts, Fleet, Manufacturing (general), Oil & Gas, Chemical, Food & Beverage, Mills & Agriculture, Facilities/HVAC, Heavy Equipment, Mining & Metals, Government (FedRAMP High).

---

## Pricing

| Tier | Price | Min Users | Billing |
|---|---|---|---|
| Software Standard | $60/user/month | 5 | Annual |
| Software Enterprise | $100/user/month | 10 | Annual |
| Bundle (hardware + software) | Contact sales | — | — |

Hardware unit cost not published. ERP integration priced separately (IT implementation hours).

---

## Customer ROI

| Customer | Industry | Result |
|---|---|---|
| ICL | Chemical | 41% OEE increase; 400+ tons of production recovered |
| Whirlpool | Appliance manufacturing | $1M+ in savings |
| Ingredion | Food & Beverage | $1M+ saved at a single plant |
| Unilever | Food & Beverage | $700K+ in operational losses protected |
| Danone | Food & Beverage | Dairy production reliability improvement |
| Pirelli | Automotive | Comprehensive reliability program |
| Cadillac Casting | Metals | Reactive → preventive maintenance transition |

---

## Competitive Differentiators vs SKF / Emerson / Fluke / Bently Nevada

| Dimension | TRACTIAN Position |
|---|---|
| Sensor design | Patented mechanical hierarchy eliminates battery resonance artifacts |
| Modality | 4-in-1 (vibration + temp + ultrasound + RPM) vs separate instruments |
| Frequency range | 0–64 kHz vs typical 10–20 kHz for wireless IIoT sensors |
| IP rating | IP69K vs IP67 for most competitors |
| Hazardous area | ATEX + NFPA 70 CL1/CL2/CL3 Class I Div 1 |
| Connectivity | Sub-GHz 915 MHz (wall penetration) + 4G/LTE, no gateway required |
| AI output | Named fault + severity + work instruction, not just alert threshold |
| Platform bundling | Native CMMS + condition monitoring vs monitoring-only incumbent products |
| Training data | 3.5B samples across hundreds of thousands of assets |
| Entry price | $60/user/month SaaS vs enterprise contract pricing of incumbents |
| Deployment | Permanent glue/screw mount, no cabling, self-enrollment |

---

## Known Gaps and Competitive Openings

1. **No published ML accuracy metrics** — false-positive rate, detection lead time not independently validated.
2. **No magnetic mount** — permanent installation is a deployment friction point vs tool-free magnetic-mount competitors (Fluke 3563, SKF Axios).
3. **No true AE sensing** — acoustic emission at 100–400 kHz for very early crack/fatigue detection is absent.
4. **No thermal imaging (accessible page)** — TracVision IA product page 404; specs unknown.
5. **Hardware price opacity** — unit cost not published, complicates ROI calculation for prospects.
6. **Gateway architecture underdocumented** for the 915 MHz mesh mode.
7. **Detection lead time claim** ("weeks before failure") lacks third-party validation data.
