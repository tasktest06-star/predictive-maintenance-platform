# Input Prompt

> Original user prompt that initiated this research and planning effort

---

conduct a detailed case study of predictive maintenance startup called traction which uses sensor technology. website is get.traction.com create detailed requirements plan and implementation plan to develop a similar product. create different plans for different products. find github repos with apache 2 license that support the development and find papers that can be used for development of the tech. spawn multiple agents

---

## Clarifications

- The website `get.traction.com` does not exist (DNS NXDOMAIN).  
- User confirmed the intended company is **TRACTIAN** at `tractian.com`.
- Multiple implementation branches requested: create git branches with different technical approaches to the same problems, developed in parallel by separate agents.

---

## Research Artifacts Generated

| File | Description |
|---|---|
| `research/tractian_case_study.md` | Full TRACTIAN case study from tractian.com |
| `plans/00_overview.md` | Product portfolio overview and cross-cutting architecture |
| `plans/01_smart_sensor_hardware.md` | Wireless sensor hardware requirements + implementation plan |
| `plans/02_edge_firmware.md` | Embedded firmware requirements + implementation plan |
| `plans/03_iot_data_pipeline.md` | IoT data pipeline requirements + implementation plan |
| `plans/04_ml_ai_engine.md` | ML/AI diagnostic engine requirements + implementation plan |
| `plans/05_cloud_platform.md` | Cloud platform & API requirements + implementation plan |
| `plans/06_cmms_dashboard.md` | CMMS & monitoring dashboard requirements + implementation plan |
| `plans/07_electrical_monitoring.md` | Electrical monitoring product requirements + implementation plan |
| `resources/apache_repos.md` | 32 Apache 2.0 licensed GitHub repositories |
| `resources/academic_papers.md` | 32 academic papers with arXiv IDs |

## Implementation Branches

Each branch explores a different technical approach for a key architectural decision:

| Branch | Approach |
|---|---|
| `impl/ml-unsupervised-vae` | Unsupervised VAE-based anomaly detection (no labeled fault data needed) |
| `impl/ml-supervised-1dcnn` | Supervised 1D CNN fault classification (labeled training data) |
| `impl/ml-transformer-faultformer` | Transformer / masked pretraining approach (FaultFormer-style) |
| `impl/pipeline-kafka-flink` | Kafka + Apache Flink streaming pipeline |
| `impl/pipeline-thingsboard` | ThingsBoard IoT platform as unified pipeline + dashboard |
| `impl/sensor-mems-lowpower` | MEMS-only low-power sensor (STM32U5-based) |
| `impl/sensor-piezo-ae` | Piezoelectric + AE transducer high-frequency design (STM32H7-based) |
| `impl/dashboard-superset` | Apache Superset + VictoriaMetrics analytics dashboard |
