#!/usr/bin/env python3
"""Embed coherence-research concept catalog into fact_registry with correct named vectors."""

import sys
import uuid
from datetime import datetime

sys.path.insert(0, "/Users/jthor/GitHub/policy-orchestrator/scripts")

from lib.embedder import embed_text
from lib.qdrant_helpers import get_client
from qdrant_client.models import PointStruct

COLLECTION = "fact_registry"

FACTS = [
    # Category 1: Delta.72 Core Theory
    {
        "fact": "Delta.72 coherence equation: Δ = (P·A·R)/(D+N) where P=Pattern Retention (correlation with expected baseline), A=Phase Alignment (temporal consistency/autocorrelation), R=Recovery (ability to return to baseline after deviation), D=Drift (mean deviation from baseline), N=Noise Amplification (variance/instability).",
        "source_type": "other",
        "confidence": "documented",
        "domain": "technical",
        "source_ref": "coherence-research/data/incoming/delta72_experiments.md",
        "source_date": "2026-05-08",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Core equation of the Delta.72 coherence framework for instability detection",
    },
    {
        "fact": "Delta.72 extended operators: 𝓜 (Memory-of-Attractor) measures whether a system remembers how to return to baseline. 𝓦 (Windowed Recovery) measures whether it recovers within time bounds. Gated detection: ALERT if Δ low AND 𝓜 low AND 𝓦 low.",
        "source_type": "other",
        "confidence": "documented",
        "domain": "technical",
        "source_ref": "coherence-research/data/incoming/delta72_experiments.md",
        "source_date": "2026-05-08",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Extended operators and gated alert condition for coherence framework",
    },
    {
        "fact": "Delta.72 experiment suite: 7 GPU-parallelizable experiments. (1) Coherence vs Noise Threshold — synthetic, find threshold where coherence collapses. (2) Recovery Dynamics After Shock — synthetic, test recovery differences after perturbation. (3) Hidden Drift Before Visible Failure — synthetic, detect drift before breakdown. (4) Shock Response vs Coherence — synthetic, quantify deviation magnitude vs coherence level. (5) Cross-System Generalization — synthetic, test across sinusoidal/chaotic/piecewise/stochastic signals. (6) Monte Carlo Lead-Time Analysis — synthetic, N=100-1000 randomized simulations. (7) Real Data Validation on Energy Systems — multi-building electricity load with weather-adjusted baseline.",
        "source_type": "other",
        "confidence": "documented",
        "domain": "technical",
        "source_ref": "coherence-research/data/incoming/delta72_experiments.md",
        "source_date": "2026-05-08",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Full 7-experiment suite for GPU coherence validation. Experiments 1-6 synthetic, Experiment 7 real data.",
    },
    # Category 2: Energy Open Data Validation
    {
        "fact": "Energy Open Data Tests completed through 9 phases on 4 office buildings (Hog_office_Betsy, Hog_office_Nia, Lamb_office_Vasiliki, Rat_office_Avis). Phases: (1) single-building coherence, (2) weather normalization, (3) baseline battle/method comparison, (4) operator coverage and lead time, (5) value estimation with dollar figures, (6) pilot readiness packet, (7) pilot simulation with dashboard spec, (8) full-stack operator tables internal+partner-safe, (9) NDA business partner materials. 147 files total, 192MB.",
        "source_type": "other",
        "confidence": "documented",
        "domain": "technical",
        "source_ref": "coherence-research/data/incoming/energy_open_data_tests.zip",
        "source_date": "2026-05-08",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Complete energy validation dataset covering coherence framework application to real building energy data",
    },
    {
        "fact": "Energy open data raw datasets: electricity_phase1_sample.csv (162MB, hourly electricity load), metadata_phase1.csv (building metadata), weather_phase2_lamb_hog_rat.csv (weather data for normalization). Test buildings are in Hog, Lamb, and Rat sites. Coherence Engine Pilot PDF (9.4MB) included.",
        "source_type": "other",
        "confidence": "documented",
        "domain": "technical",
        "source_ref": "coherence-research/data/incoming/energy_open_data_tests.zip",
        "source_date": "2026-05-08",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Raw input data for coherence framework validation on energy systems",
    },
    # Category 3: Philosophical / Consciousness Framework
    {
        "fact": "Allison's '12 Illusions / 12 Lenses' framework maps 12 societal illusions to reframes: Love Must Be Earned→Love Is Inherent, Money Must Be Hard-Earned→Abundance Flows, Divine Is Separate→Divine Lives Through Me, Time Is Linear→Time Is An Illusion, Power Means Control→Power Is Presence, Success Must Be Proven→Success Is Alignment, Suffering Must Be Spiritual→Suffering Is A Teacher, Choose Purpose Or Prosperity→Purpose Fuels Prosperity, Only Specialists Heal→Healing Lives Within All, Feminine Must Be Passive→Feminine Is Creative Power, Earth Not Sacred→Earth Is Sacred, One Must Wait→The Time Is Always Now.",
        "source_type": "photograph",
        "confidence": "documented",
        "domain": "personal",
        "source_ref": "coherence-research/data/incoming/allison_photo_2026-05-08_2.jpg",
        "source_date": "2026-05-08",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Philosophical framework shared by Allison, visual infographic format",
    },
    {
        "fact": "Growth Spiral / Lattice concept: 12 people sent to disrupt, opening up to 72, then 144, then 144,000 — a critical mass sequence to 'sustain the lattice and shift humanity's lens from 3D to 5D'. The number 72 connects to the Delta.72 framework name. Timeline: humanity awakens in 5D by 2029.",
        "source_type": "photograph",
        "confidence": "asserted",
        "domain": "personal",
        "source_ref": "coherence-research/data/incoming/allison_photo_2026-05-08_1.jpg",
        "source_date": "2026-05-08",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "The number 72 as a coherence tipping point — bridges philosophical and technical frameworks",
    },
    # Category 4: Personal Background
    {
        "fact": "Allison operated 'Under The Oak Catering' with a substantial farm-to-table operation including vegetable gardens (row crops, trellised plants, heirloom/rainbow carrots). Active at least 2015-2016 based on Facebook posts.",
        "source_type": "photograph",
        "confidence": "documented",
        "domain": "personal",
        "source_ref": "coherence-research/data/incoming/00010802-PHOTO-2026-04-11-17-47-29.jpg",
        "source_date": "2015-11-03",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Background context — Allison's catering/farming history",
    },
    # Category 5: Property Records
    {
        "fact": "Gotliffe Properties LLC owns parcels at 1921 Polenta Rd and 2035 Indian Camp Rd, Clayton NC 27520. Market value $1,700,250, 4.0 acres each. Sold 2024-02-26 for $545,000. Book 06621 Page 0850. Mailing address 1126 Ridge Dr, Clayton NC 27520-9185.",
        "source_type": "public_record",
        "confidence": "verified",
        "domain": "property",
        "source_ref": "coherence-research/data/incoming/00010814-PHOTO-2026-04-11-19-28-57.jpg",
        "source_date": "2024-02-26",
        "claimed_by": "Johnston County NC MapClick",
        "repo": "coherence-research",
        "notes": "Johnston County public property records for Gotliffe Properties LLC",
    },
    {
        "fact": "Blake and Megan Gotliffe Living Trust owns 1126 Ridge Dr, Clayton NC 27520. Market value $329,430, 1.05 acres. Transfer date 2024-05-30, sales price $0 (trust transfer). Trustee: Blake Gotliffe. NCPin 164900-91-4844, Book 06677 Page 0508.",
        "source_type": "public_record",
        "confidence": "verified",
        "domain": "property",
        "source_ref": "coherence-research/data/incoming/00010816-PHOTO-2026-04-11-19-28-57.jpg",
        "source_date": "2024-05-30",
        "claimed_by": "Johnston County NC MapClick",
        "repo": "coherence-research",
        "notes": "Johnston County public property records for Gotliffe Living Trust",
    },
    # Category 6: Business Structure
    {
        "fact": "Upwork posting for 'Holdco + intercompany licensing structure project'. Scope: entity/IP audit, operating agreement drafting, intercompany license template. Applicant David L. (legal support professional, non-attorney), $35/hr, Las Vegas NV. Experience in Wyoming/Delaware formations, chain-of-title review, IP holding company frameworks. 4 milestones starting at $400.",
        "source_type": "photograph",
        "confidence": "documented",
        "domain": "legal",
        "source_ref": "coherence-research/data/incoming/00010828-PHOTO-2026-04-13-18-51-44.jpg",
        "source_date": "2026-04-13",
        "claimed_by": "Allison Hensgen",
        "repo": "coherence-research",
        "notes": "Active work on holding company + IP licensing structure",
    },
]


def main():
    client = get_client()

    for i, f in enumerate(FACTS):
        embed_parts = [f["fact"]]
        if f.get("notes"):
            embed_parts.append(f["notes"])
        if f.get("source_ref"):
            embed_parts.append(f"Source: {f['source_ref']}")
        embed_content = "\n".join(embed_parts)

        vector = embed_text(embed_content)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": vector},  # Named vector to match collection schema
            payload={
                "fact": f["fact"],
                "source_type": f["source_type"],
                "confidence": f["confidence"],
                "confidence_rank": {"verified": 0, "documented": 1, "asserted": 2, "disputed": 3, "inferred": 4, "unknown": 5}.get(f["confidence"], 5),
                "domain": f["domain"],
                "source_ref": f.get("source_ref", ""),
                "source_date": f.get("source_date", ""),
                "claimed_by": f.get("claimed_by", ""),
                "contradicts": "",
                "repo": f.get("repo", ""),
                "notes": f.get("notes", ""),
                "logged_at": datetime.now().isoformat(),
                "text": embed_content,
            },
        )

        client.upsert(collection_name=COLLECTION, points=[point])
        print(f"  [{i+1}/{len(FACTS)}] Embedded: {f['fact'][:80]}...")

    info = client.get_collection(COLLECTION)
    print(f"\nDone. fact_registry now has {info.points_count} total facts.")


if __name__ == "__main__":
    main()
