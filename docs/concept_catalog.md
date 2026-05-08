# Incoming Data — Concept Catalog

> Generated: 2026-05-08
> Source: `data/incoming/` (15 files)

---

## Category 1: Delta.72 Coherence Framework (Core Theory)

**Source:** `delta72_experiments.md`

### Core Equation
```
Δ = (P · A · R) / (D + N)
```

| Symbol | Name | Meaning |
|--------|------|---------|
| P | Pattern Retention | Correlation with expected baseline |
| A | Phase Alignment | Temporal consistency / autocorrelation |
| R | Recovery | Ability to return to baseline after deviation |
| D | Drift | Mean deviation from baseline |
| N | Noise Amplification | Variance / instability |

### Extended Operators
| Operator | Name | Meaning |
|----------|------|---------|
| 𝓜 | Memory-of-Attractor | Does the system remember how to return to baseline |
| 𝓦 | Windowed Recovery | Does it recover within time bounds |

### Gated Detection Rule
```
ALERT if: Δ low AND 𝓜 low AND 𝓦 low
```

### 7-Experiment Suite (GPU-compatible)

| # | Experiment | Type | Status |
|---|-----------|------|--------|
| 1 | Coherence vs Noise Threshold | Synthetic | Not started |
| 2 | Recovery Dynamics After Shock | Synthetic | Not started |
| 3 | Hidden Drift Before Visible Failure | Synthetic | Not started |
| 4 | Shock Response vs Coherence | Synthetic | Not started |
| 5 | Cross-System Generalization | Synthetic | Not started |
| 6 | Monte Carlo Lead-Time Analysis | Synthetic | Not started |
| 7 | Real Data Validation (Energy Systems) | Real data | **Phases 1–9 complete** (see Category 2) |

---

## Category 2: Energy Open Data Validation (Applied Research)

**Source:** `energy_open_data_tests.zip` (192MB, 147 files)
**Manifest:** `docs/energy_open_data_manifest.md`

### Test Buildings
| Code | Building |
|------|----------|
| Hog_office_Betsy | Office building (Hog site) |
| Hog_office_Nia | Office building (Hog site) |
| Lamb_office_Vasiliki | Office building (Lamb site) |
| Rat_office_Avis | Office building (Rat site) |

### Phase Progression

| Phase | Name | What it covers | Key outputs |
|-------|------|---------------|-------------|
| 1 | Single-building coherence | Hour-of-week baseline, Δ scoring, episode detection | Per-building CSVs, coherence plots, validation report |
| 2 | Weather normalization | Temperature-adjusted baselines | Weather-normalized coherence, comparison plots |
| 3 | Baseline battle | Method comparison (multiple baselines) | Daily alert timelines, episode overlap matrices, synthesis |
| 4 | Operator coverage | 𝓜 and 𝓦 operator performance, lead time | Operator coverage tables, lead-time report |
| 5 | Value estimation | Dollar value of early detection | Low/mid/high scenario grids, portfolio scenarios |
| 6 | Pilot readiness | Partner-facing validation packet | Pilot checklist, success metrics, Waterloo follow-up email |
| 7 | Pilot simulation | Dashboard specs, SOW, building risk ranking | Pilot SOW, partner email, simulation report |
| 8 | Full-stack internal + partner-safe | Complete operator tables for all buildings | Internal bundle (21MB+), partner-safe bundle with reports |
| 9 | NDA materials | Executive summaries, demo script | NDA/non-NDA summaries, business partner demo script |

### Supplementary
- `Coherence_Engine_Pilot.pdf` — 9.4MB pilot document
- `open_data_findings_report_bundle/` — Final findings report (docx/md/pdf + summary CSV)
- Raw data: `electricity_phase1_sample.csv` (162MB), `metadata_phase1.csv`, `weather_phase2_lamb_hog_rat.csv`

---

## Category 3: Philosophical / Consciousness Framework

**Source:** `allison_photo_2026-05-08_1.jpg`, `allison_photo_2026-05-08_2.jpg`

### "12 Illusions / 12 Lenses" Framework

| # | Illusion (3D) | Lens / Reframe (5D) |
|---|--------------|---------------------|
| 1 | Love Must Be Earned | Love Is Inherent |
| 2 | Money Must Be Hard-Earned | Abundance Flows Through Me |
| 3 | The Divine Is Separate From The Body | The Divine Lives Through Me |
| 4 | Time Is Linear | Time Is An Illusion |
| 5 | Power Means Control | Power Is Presence |
| 6 | Success Must Be Proven | Success Is Alignment |
| 7 | Suffering Must Be Spiritual | Suffering Is A Teacher |
| 8 | One Has To Choose Between Purpose And Prosperity | Purpose Fuels Prosperity |
| 9 | Only Specialists Can Heal | Healing Lives Within All |
| 10 | Feminine Energy Must Be Passive | Feminine Energy Is Creative Power |
| 11 | Earth Is Not Sacred | Earth Is Sacred |
| 12 | One Must Wait | The Time Is Always Now |

### Growth Spiral / Lattice Numbers
- 12 people sent to disrupt → opening up to **72** → then **144** → reaching **144,000**
- Goal: "Sustain the lattice and shift humanity's lens from 3D to 5D"
- Timeline: Humanity awakens in 5D by 2029

### Connection to Delta.72
- The number **72** is embedded in the framework name "Delta.72"
- The Growth Spiral posits 72 as a critical mass threshold — a coherence tipping point
- This mirrors the coherence framework's threshold detection concept: a minimum group size at which coherent signal overpowers noise

---

## Category 4: Personal Background — Allison (Under The Oak Catering)

**Source:** `00010802`, `00010803`, `00010807`

- "Under The Oak Catering" — food/catering business, active at least 2015–2016
- Operated a substantial vegetable garden/farm (row crops, trellised plants)
- Heirloom/rainbow carrot harvest (May 2016) — suggests farm-to-table orientation
- Cape Cod-style house with property — context for the farming operation

---

## Category 5: Property Records — Johnston County, NC

**Source:** `00010808`, `00010814`, `00010815`, `00010816`, `00010817`

### Gotliffe Properties, LLC

| Property | Address | Market Value | Acreage | Sale Date | Sale Price |
|----------|---------|-------------|---------|-----------|------------|
| Parcel A | 1921 Polenta Rd, Clayton NC 27520 | $1,700,250 | 4.000 | 2024-02-26 | $545,000 |
| Parcel B | 2035 Indian Camp Rd, Clayton NC 27520 | $1,700,250 | 4.000 | 2024-02-26 | $545,000 |

- Book 06621, Page 0850
- Mailing: 1126 Ridge Dr, Clayton NC 27520-9185

### Blake and Megan Gotliffe Living Trust

| Property | Address | Market Value | Acreage | Sale Date | Sale Price |
|----------|---------|-------------|---------|-----------|------------|
| Trust parcel | 1126 Ridge Dr, Clayton NC 27520 | $329,430 | 1.050 | 2024-05-30 | $0 (transfer) |

- NCPin: 164900-91-4844, Mapsheet 1649
- Book 06677, Page 0508
- Trustee: Blake Gotliffe

---

## Category 6: Business Structure / Legal Services

**Source:** `00010828`, `00010829`

- **Upwork posting:** "Holdco + intercompany licensing structure project"
- **Scope:** Entity/IP audit, operating agreement drafting, intercompany license template
- **Applicant shown:** David L. — legal support professional (non-attorney), $35/hr, Las Vegas NV
  - Experience: Wyoming/Delaware formations, chain-of-title review, IP holding company frameworks
  - 4 milestones proposed, starting at $400
- Suggests active work on a holding company + IP licensing structure

---

## Concept Cross-References

| Concept | Appears In |
|---------|-----------|
| Coherence / Δ metric | Categories 1, 2, 3 |
| Threshold / tipping point | Categories 1, 3 |
| The number 72 | Categories 1, 3 |
| Recovery / resilience | Categories 1, 2 |
| Energy systems | Categories 1, 2 |
| Property / entity structure | Categories 5, 6 |
| Farm-to-table / catering | Category 4 |
