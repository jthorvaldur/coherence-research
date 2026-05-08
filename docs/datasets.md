# Public Datasets for Delta.72 Validation

> Track all candidate datasets, their size, domain, and status.

## Active / In Progress

| Dataset | Domain | Est. Size | Labeled Failures | Status |
|---------|--------|-----------|-----------------|--------|
| Energy Open Data (Phase 1) | Energy | 162 MB | No (coherence-derived) | **Done** (Exp 7) |

## Queued

| # | Dataset | Domain | Est. Size | Labeled Failures | Why it fits | Source |
|---|---------|--------|-----------|-----------------|-------------|--------|
| 1 | NASA C-MAPSS Turbofan | Aerospace | ~50 MB | Yes (RUL labels) | Direct time-to-failure, multi-sensor, exact lead-time comparison | [NASA Prognostics](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) |
| 2 | NAB (Numenta Anomaly Benchmark) | Mixed (AWS, traffic, tweets) | ~20 MB | Yes (labeled windows + scoring) | 58 time series, formal scoring framework, published baselines | [github.com/numenta/NAB](https://github.com/numenta/NAB) |
| 3 | Yahoo S5 Anomaly | Infrastructure | ~200 MB | Yes (labeled anomalies) | Server metrics, synthetic + real, standard benchmark | [Yahoo Webscope](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70) |
| 4 | SKAB (Skoltech Anomaly) | Industrial | ~10 MB | Yes (labeled faults) | Valve sensor data, manufacturing equipment | [github.com/waico/SKAB](https://github.com/waico/SKAB) |
| 5 | UCI Household Power | Energy | ~130 MB | No | 4 years, 1-minute resolution, 2M readings | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) |
| 6 | ERCOT Grid Frequency | Energy | ~500 MB+ | Yes (grid events) | Texas grid, 4-second resolution, directly relevant to energy work | [ERCOT.com](http://www.ercot.com/gridinfo/generation) |
| 7 | PhysioNet ECG (MIT-BIH) | Biomedical | ~100 MB | Yes (arrhythmia labels) | Heart rhythm, labeled cardiac events, biological stability | [physionet.org](https://physionet.org/content/mitdb/) |
| 8 | PhysioNet EEG (CHB-MIT) | Biomedical | ~23 GB | Yes (seizure labels) | Brain signals, pre-seizure detection, direct lead-time test | [physionet.org](https://physionet.org/content/chbmit/) |
| 9 | IRIS Seismic | Geophysics | Variable | Yes (earthquake catalogs) | Seismograph data, pre-event coherence degradation | [ds.iris.edu](https://ds.iris.edu/ds/nodes/dmc/) |
| 10 | Credit Card Fraud (Kaggle) | Financial | ~150 MB | Yes (fraud labels) | Transaction time series, anomaly detection benchmark | [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) |

## Future / Aspirational

| Dataset | Domain | Est. Size | Notes |
|---------|--------|-----------|-------|
| Financial tick data (various) | Finance | Multi-GB | Needs vendor access or crypto exchange APIs |
| Satellite telemetry (SMAP/MSL) | Aerospace | ~1 GB | NASA telemetry anomaly dataset |
| Water treatment (SWaT) | Infrastructure | ~1 GB | Cyber-physical system attack dataset |
| Air quality sensor networks | Environmental | ~500 MB | EPA + PurpleAir open data |
