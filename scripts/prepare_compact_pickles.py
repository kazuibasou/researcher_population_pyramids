"""
Step 1 of the website data pipeline.

Reads OpenAlex author data from ./data/, assigns gender via nqg
(nomquamgender, Van Buskirk, Clauset & Larremore, ICWSM 2023), and emits
one compact gzipped pickle per country to ./compact_data/<CC>.pkl.gz.

Faithful to the paper (Nakajima & Mizuno 2026, PNAS Nexus 5(3), pgag059)
except the gender classifier is changed from Orbis-based ComplementNB to nqg.

The 58-country list is taken from the paper's selection (ComplementNB-based
`gender_analysis.country_selection`), reproduced here from the existing
pickles in ./data/ (a curated subset of the paper's working data).

Output schema (one record per author):
    {
        "gender": 0 | 1 | None,         # nqg + country-specific threshold
        "pub_dates": ["YYYY-MM-DD", ...],  # sorted ascending
        "domains": ["physical_sciences", ...]  # set of OpenAlex domains the
                                               # author has published in
                                               # (used for singleton filter
                                               # in compute_pyramids.py)
    }

Country file metadata:
    {
        "country": "JP",
        "nqg_threshold": 0.987,
        "n_authors": 12345,
        "n_female": 3400,
        "n_male": 7800,
        "n_unknown": 1145,
        "authors": [ ... ]
    }
"""

from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import sys
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration (faithful to paper)
# ---------------------------------------------------------------------------

DATA_DIR = "./data/"
OUT_DIR = "./compact_data/"

# ----- The 58-country list from the paper -----
#
# These are the exact 58 countries selected in Nakajima & Mizuno (2026),
# PNAS Nexus 5(3), pgag059. The selection criteria (from the paper's
# ComplementNB-based `gender_analysis.country_selection()`):
#   - Sufficient Orbis name samples (>= ORBIS_TEST_SIZE * MIN_RATIO_TRAIN_TO_TEST)
#     per gender (or Arab-minority pool eligibility)
#   - ComplementNB Orbis threshold >= 0.9
#   - ComplementNB WGND threshold >= 0.9
#   - Not in {CN, IN, BD, KR} (low AUC for ComplementNB)
#   - n_female >= 1000 and n_male >= 1000 (ComplementNB gender assignments)
#
# The list is hard-coded here so the pipeline only needs the per-country
# OpenAlex pickles + nqg threshold pickles + topic mapping (no ComplementNB
# intermediates). `select_paper_countries()` below reproduces the original
# logic for documentation/audit purposes but is not called at runtime.
SELECTED_COUNTRIES = [
    "AE", "AL", "AT", "AU", "BA", "BE", "BG", "BH", "BY", "CA",
    "CH", "CY", "CZ", "DE", "DK", "DZ", "EE", "EG", "ES", "FR",
    "GB", "GH", "IE", "IL", "IQ", "IR", "IS", "IT", "JM", "JO",
    "JP", "KE", "KW", "LB", "LK", "LT", "MA", "MD", "ME", "NG",
    "NL", "NO", "NZ", "OM", "PH", "PL", "PT", "RO", "RS", "RU",
    "SA", "SE", "SY", "TN", "TR", "US", "ZA", "ZW",
]
assert len(SELECTED_COUNTRIES) == 58, "Country list integrity"

# ----- Paper parameters (referenced by select_paper_countries; documentation only) -----
ARAB_LEAGUE = {
    "DZ", "BH", "KM", "DJ", "EG", "IQ", "JO", "KW", "LB", "LY", "MR",
    "MA", "OM", "PS", "QA", "SA", "SO", "SD", "SY", "TN", "AE", "YE",
}
ORBIS_TEST_SIZE = 450
MIN_RATIO_TRAIN_TO_TEST = 5
COUNTRIES_WITH_LOW_AUC = {"CN", "IN", "BD", "KR"}
INCLUDE_MIDDLE_NAME = False
INCLUDE_LAST_NAME = False
CLEANING_SPECIAL_CHARS = True
MIN_AUTHORS_PER_GENDER = 1000
PROB_THRESHOLD_MIN = 0.9


# ---------------------------------------------------------------------------
# Country selection (mirrors gender_analysis.country_selection)
# ---------------------------------------------------------------------------

def select_paper_countries():
    """Reproduce the paper's 58-country selection from existing pickles.

    NOT called at runtime — the result is hard-coded in SELECTED_COUNTRIES
    above. Kept here as documentation / audit trail for how those 58
    countries were chosen. Reading the inputs requires `data_sets.pkl`,
    `complementnb_prob_threshold_in_{orbis,wgnd}.pickle`, and the
    `author_{gender,sample_lst}_*.pkl` files from the paper's working
    directory — none of which need to ship with this pipeline.
    """

    print("[select] loading data_sets.pkl ...", flush=True)
    with open(DATA_DIR + "data_sets.pkl", "rb") as f:
        data_by_country = pickle.load(f)

    print("[select] loading ComplementNB thresholds ...", flush=True)
    with open(DATA_DIR + "complementnb_prob_threshold_in_orbis.pickle", "rb") as f:
        cnb_thresh_orbis = pickle.load(f)
    with open(DATA_DIR + "complementnb_prob_threshold_in_wgnd.pickle", "rb") as f:
        cnb_thresh_wgnd = pickle.load(f)

    suffix = (
        "_middle_name_" + str(INCLUDE_MIDDLE_NAME)
        + "_last_name_" + str(INCLUDE_LAST_NAME)
        + "_special_chars_" + str(CLEANING_SPECIAL_CHARS)
        + ".pkl"
    )
    print("[select] loading ComplementNB author_gender ...", flush=True)
    with open(DATA_DIR + "author_gender" + suffix, "rb") as f:
        cnb_author_gender = pickle.load(f)

    print("[select] loading author_sample_lst ...", flush=True)
    with open(DATA_DIR + "author_sample_lst" + suffix, "rb") as f:
        _, cnb_author_sample_lst = pickle.load(f)

    # --- Stage 1: filter by Orbis training-data size (incl. Arab minority pool)
    name_count = {
        c: (len(data_by_country[c][0]), len(data_by_country[c][1]))
        for c in data_by_country
    }
    all_countries = [
        c for (c, _) in sorted(name_count.items(), key=lambda x: x[1], reverse=True)
    ]
    candidates = []
    arab_minority = set()
    for country in all_countries:
        f_count, m_count = name_count[country]
        too_small_f = f_count < ORBIS_TEST_SIZE * MIN_RATIO_TRAIN_TO_TEST
        too_small_m = m_count < ORBIS_TEST_SIZE * MIN_RATIO_TRAIN_TO_TEST
        if too_small_f or too_small_m:
            if (
                country in ARAB_LEAGUE
                and f_count > ORBIS_TEST_SIZE
                and m_count > ORBIS_TEST_SIZE
            ):
                arab_minority.add(country)
            continue
        candidates.append(country)
    candidates += list(arab_minority)

    # --- Stage 2: threshold + low-AUC + author-count filter
    n_by_country = {}
    for country in candidates:
        n_f, n_m = 0, 0
        for a_id in cnb_author_sample_lst.get(country, []):
            g = cnb_author_gender.get(a_id, -1)
            if g == 0:
                n_f += 1
            elif g == 1:
                n_m += 1
        n_by_country[country] = (n_f, n_m)

    selected = []
    for country in candidates:
        p1 = cnb_thresh_orbis.get(country, 0.0)
        p2 = cnb_thresh_wgnd.get(country, 0.0)
        n_f, n_m = n_by_country.get(country, (0, 0))
        if (
            p1 < PROB_THRESHOLD_MIN
            or p2 < PROB_THRESHOLD_MIN
            or country in COUNTRIES_WITH_LOW_AUC
            or n_f < MIN_AUTHORS_PER_GENDER
            or n_m < MIN_AUTHORS_PER_GENDER
        ):
            continue
        selected.append(country)

    selected.sort()
    print(f"[select] selected {len(selected)} countries: {selected}", flush=True)
    return selected


# ---------------------------------------------------------------------------
# nqg threshold (paper's set_nqg_param: max of orbis & wgnd thresholds)
# ---------------------------------------------------------------------------

def load_nqg_thresholds(country_lst):
    """Load per-country nqg probability thresholds.

    Reads `data/nqg_thresholds.json`, which holds the paper's `set_nqg_param`
    result — i.e. max(orbis_threshold, wgnd_threshold) per country — already
    precomputed and shipped at
    https://github.com/kazuibasou/researcher_population_pyramids/blob/main/framework/data/cct_prob_threshold.json
    """
    thresholds_path = os.path.join(os.path.dirname(__file__), "nqg_thresholds.json")
    with open(thresholds_path) as f:
        thr = json.load(f)
    missing = [c for c in country_lst if c not in thr]
    if missing:
        print(f"[nqg] WARNING: missing nqg threshold for: {missing}", flush=True)
    return {c: float(thr[c]) for c in country_lst if c in thr}


# ---------------------------------------------------------------------------
# Field → OpenAlex domain mapping
# ---------------------------------------------------------------------------

def load_field_to_domain():
    with open(DATA_DIR + "openalex_topic_data.pickle", "rb") as f:
        topic_data = pickle.load(f)
    mapping = {}
    for t_id in topic_data:
        field = topic_data[t_id]["field"]["display_name"]
        domain = topic_data[t_id]["domain"]["display_name"]
        mapping[field] = domain
    return mapping


def normalize_domain(domain_str):
    """Map OpenAlex domain display name → snake_case key for JSON."""
    return domain_str.lower().replace(" ", "_").replace("&", "and")


# ---------------------------------------------------------------------------
# Per-country compact pickle generation
# ---------------------------------------------------------------------------

def process_country(country, nqg_model, threshold, field_to_domain, overwrite=False):
    cc_lower = country.lower()
    in_path = DATA_DIR + f"openalex_author_data_{cc_lower}.pickle"
    out_path = OUT_DIR + f"{country}.pkl.gz"

    if not overwrite and os.path.exists(out_path):
        print(f"[{country}] SKIP (output exists)", flush=True)
        return None

    if not os.path.exists(in_path):
        print(f"[{country}] SKIP (input pickle not found: {in_path})", flush=True)
        return None

    t0 = time.time()
    print(f"[{country}] loading {in_path}", flush=True)
    with open(in_path, "rb") as f:
        author_data = pickle.load(f)

    # --- Phase 1: filter to singleton-country authors and collect first names
    filtered = []  # (info, first_name)
    n_skip_country = 0
    n_skip_name = 0
    for a_id, info in author_data.items():
        country_set = set(info.get("country", []))
        if len(country_set) != 1 or country not in country_set:
            n_skip_country += 1
            continue

        name = str(info.get("name", "")).lower()
        if not name:
            n_skip_name += 1
            continue
        name_words = [w for w in name.split(" ") if w]
        if len(name_words) <= 1:
            n_skip_name += 1
            continue

        first_name = name_words[0]
        filtered.append((info, first_name))

    print(
        f"[{country}] kept {len(filtered)} authors "
        f"(skipped: {n_skip_country} non-singleton country, "
        f"{n_skip_name} invalid name)",
        flush=True,
    )

    if not filtered:
        print(f"[{country}] no authors after filtering — skipping", flush=True)
        return None

    # --- Phase 2: batch nqg prediction
    first_names = [fn for _, fn in filtered]
    print(f"[{country}] running nqg on {len(first_names)} names ...", flush=True)
    df = nqg_model.annotate(first_names, as_df=True)
    p_g_list = df["p(gf)"].tolist()

    # --- Phase 3: assemble compact records
    n_f = n_m = n_na = 0
    records = []
    for (info, _), p_g in zip(filtered, p_g_list):
        # gender
        gender = None
        if not (isinstance(p_g, float) and math.isnan(p_g)) and p_g is not None:
            p_f = float(p_g)
            p_m = 1.0 - p_f
            if p_f > p_m:
                g, prob = 0, p_f
            elif p_f < p_m:
                g, prob = 1, p_m
            else:
                g, prob = None, 0.0
            if g is not None and prob >= threshold:
                gender = g

        if gender == 0:
            n_f += 1
        elif gender == 1:
            n_m += 1
        else:
            n_na += 1

        # publication dates (sorted ascending, ISO strings sort chronologically)
        pub_dates = sorted(str(pub[1]) for pub in info.get("pubs", []))

        # domains the author has published in
        disciplines = set(info.get("discipline", []))
        domains = sorted({
            normalize_domain(field_to_domain[f])
            for f in disciplines
            if f in field_to_domain
        })

        records.append({
            "gender": gender,
            "pub_dates": pub_dates,
            "domains": domains,
        })

    payload = {
        "country": country,
        "nqg_threshold": threshold,
        "n_authors": len(records),
        "n_female": n_f,
        "n_male": n_m,
        "n_unknown": n_na,
        "authors": records,
    }

    print(f"[{country}] writing {out_path} ...", flush=True)
    with gzip.open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    dt = time.time() - t0
    print(
        f"[{country}] done in {dt:.1f}s — "
        f"F={n_f}, M={n_m}, N/A={n_na} — {size_mb:.1f} MB",
        flush=True,
    )

    return {
        "country": country,
        "threshold": threshold,
        "n_authors": len(records),
        "n_female": n_f,
        "n_male": n_m,
        "n_unknown": n_na,
        "size_mb": size_mb,
        "duration_sec": dt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse simple CLI: optional --overwrite, optional country subset
    overwrite = "--overwrite" in sys.argv
    explicit_countries = [
        a for a in sys.argv[1:] if a != "--overwrite" and not a.startswith("--")
    ]

    countries = list(SELECTED_COUNTRIES)
    if explicit_countries:
        wanted = set(explicit_countries)
        unknown = wanted - set(countries)
        if unknown:
            print(f"[main] WARNING: requested countries not in selection: {unknown}")
        countries = [c for c in countries if c in wanted]
        print(f"[main] processing subset: {countries}", flush=True)

    # Persist selected country list as the source of truth
    with open(OUT_DIR + "selected_countries.json", "w") as f:
        json.dump(countries, f, indent=2)

    nqg_thresholds = load_nqg_thresholds(countries)
    field_to_domain = load_field_to_domain()

    # Initialize nqg once (its model tables load lazily)
    print("[nqg] initializing nomquamgender model ...", flush=True)
    import nomquamgender as nqg
    nqg_model = nqg.NBGC()

    stats = []
    for country in countries:
        if country not in nqg_thresholds:
            print(f"[{country}] SKIP (no nqg threshold)", flush=True)
            continue
        result = process_country(
            country,
            nqg_model,
            nqg_thresholds[country],
            field_to_domain,
            overwrite=overwrite,
        )
        if result is not None:
            stats.append(result)

    # Summary
    if stats:
        total_authors = sum(s["n_authors"] for s in stats)
        total_f = sum(s["n_female"] for s in stats)
        total_m = sum(s["n_male"] for s in stats)
        total_na = sum(s["n_unknown"] for s in stats)
        total_size = sum(s["size_mb"] for s in stats)
        print(
            "\n=== Summary ===\n"
            f"Countries processed: {len(stats)} / {len(countries)}\n"
            f"Total authors: {total_authors:,}\n"
            f"  Female: {total_f:,} ({100*total_f/total_authors:.1f}%)\n"
            f"  Male:   {total_m:,} ({100*total_m/total_authors:.1f}%)\n"
            f"  N/A:    {total_na:,} ({100*total_na/total_authors:.1f}%)\n"
            f"Total compact pickle size: {total_size:.1f} MB\n",
            flush=True,
        )

    # Per-country stats file (for diagnostics + later use by aggregate.py)
    with open(OUT_DIR + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
