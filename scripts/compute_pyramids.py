"""
Step 2 of the website data pipeline.

Reads compact per-country pickles produced by prepare_compact_pickles.py,
computes researcher population pyramids (current 1970-2023 + Markov-chain
projection to 2050) plus aggregated timeline / rankings / scatter data, and
writes JSON files to ./web_data/ for the website frontend.

Faithful to the paper (Nakajima & Mizuno 2026, PNAS Nexus 5(3), pgag059):
- Per-(country, domain, gender) `pub_interval_threshold` from inter-pub
  survival probability with p < 0.02.
- Active researcher at year t: last pub on or after (end_of_year(t) - threshold).
- Cumulative productivity with reset when gap > threshold.
- Career length <= 40 years, total pubs <= 500.
- Markov-chain projection with newcomer counts for base years 2010 and 2023.

Designed for sequential execution on ABCI. No parallelism.

CLI:
    python scripts/compute_pyramids.py                  # all 58 countries
    python scripts/compute_pyramids.py US TN            # subset
    python scripts/compute_pyramids.py --skip-existing  # resume
"""

from __future__ import annotations

import bisect
import gzip
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import date

# ---------------------------------------------------------------------------
# Configuration (faithful to the paper)
# ---------------------------------------------------------------------------

COMPACT_DIR = "./compact_data/"
OUT_DIR = "./web_data/"

SURVIVAL_PROB_THRESHOLD = 0.02      # paper picks p = 0.02
CAREER_MAX_YEARS = 40               # paper: career length <= 40 years
PUB_THRESHOLD = 500                 # paper: total pubs <= 500
TARGET_YEAR_START = 1970
TARGET_YEAR_END = 2023  # OpenAlex 2024 snapshot is incomplete; paper uses 2023
PROJECTION_BASE_YEARS = (2023,)
PROJECTION_END_YEAR = 2050
PROJECTION_YEARS_TO_KEEP = tuple(range(2024, 2051))  # every year 2024-2050

DOMAINS = (
    "overall",
    "health_sciences",
    "life_sciences",
    "physical_sciences",
    "social_sciences",
)
NON_OVERALL_DOMAINS = DOMAINS[1:]

YEARS = tuple(range(TARGET_YEAR_START, TARGET_YEAR_END + 1))

# Pre-compute year boundary days (ordinal-based; cheap int math everywhere).
_EPOCH_ORD = date(1900, 1, 1).toordinal()
YEAR_END_DAYS = {
    y: date(y, 12, 31).toordinal() - _EPOCH_ORD
    for y in range(TARGET_YEAR_START - 1, PROJECTION_END_YEAR + 1)
}


def date_str_to_days(s: str) -> int:
    return date.fromisoformat(s).toordinal() - _EPOCH_ORD


# ---------------------------------------------------------------------------
# Region mapping for the 58 countries (UN M.49 rolled up to 6 regions).
# Used to enrich countries.json. Hard-coded to avoid extra data deps.
# ---------------------------------------------------------------------------

REGION = {
    # Africa
    "DZ": "Africa", "EG": "Africa", "GH": "Africa", "KE": "Africa",
    "MA": "Africa", "NG": "Africa", "TN": "Africa", "ZA": "Africa",
    "ZW": "Africa",
    # Asia
    "AE": "Asia", "BH": "Asia", "CY": "Asia", "IL": "Asia", "IQ": "Asia",
    "IR": "Asia", "JO": "Asia", "JP": "Asia", "KW": "Asia", "LB": "Asia",
    "LK": "Asia", "OM": "Asia", "PH": "Asia", "SA": "Asia", "SY": "Asia",
    "TR": "Asia",
    # Europe
    "AL": "Europe", "AT": "Europe", "BA": "Europe", "BE": "Europe",
    "BG": "Europe", "BY": "Europe", "CH": "Europe", "CZ": "Europe",
    "DE": "Europe", "DK": "Europe", "EE": "Europe", "ES": "Europe",
    "FR": "Europe", "GB": "Europe", "IE": "Europe", "IS": "Europe",
    "IT": "Europe", "LT": "Europe", "MD": "Europe", "ME": "Europe",
    "NL": "Europe", "NO": "Europe", "PL": "Europe", "PT": "Europe",
    "RO": "Europe", "RS": "Europe", "RU": "Europe", "SE": "Europe",
    # North America
    "CA": "North America", "US": "North America",
    # Latin America and the Caribbean
    "JM": "Latin America and the Caribbean",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania",
}

# Country display names (English, ISO 3166-1).
COUNTRY_NAME = {
    "AE": "United Arab Emirates", "AL": "Albania", "AT": "Austria",
    "AU": "Australia", "BA": "Bosnia and Herzegovina", "BE": "Belgium",
    "BG": "Bulgaria", "BH": "Bahrain", "BY": "Belarus", "CA": "Canada",
    "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "DZ": "Algeria", "EE": "Estonia",
    "EG": "Egypt", "ES": "Spain", "FR": "France",
    "GB": "United Kingdom", "GH": "Ghana", "IE": "Ireland",
    "IL": "Israel", "IQ": "Iraq", "IR": "Iran", "IS": "Iceland",
    "IT": "Italy", "JM": "Jamaica", "JO": "Jordan", "JP": "Japan",
    "KE": "Kenya", "KW": "Kuwait", "LB": "Lebanon", "LK": "Sri Lanka",
    "LT": "Lithuania", "MA": "Morocco", "MD": "Moldova",
    "ME": "Montenegro", "NG": "Nigeria", "NL": "Netherlands",
    "NO": "Norway", "NZ": "New Zealand", "OM": "Oman",
    "PH": "Philippines", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "RS": "Serbia", "RU": "Russia",
    "SA": "Saudi Arabia", "SE": "Sweden", "SY": "Syria",
    "TN": "Tunisia", "TR": "Turkey", "US": "United States",
    "ZA": "South Africa", "ZW": "Zimbabwe",
}


# ---------------------------------------------------------------------------
# Author preprocessing (one-time per country)
# ---------------------------------------------------------------------------

def preprocess_authors(raw_authors):
    """Materialize the fields we re-use across all year loops.

    Returns a list of dicts with:
        gender:     0 / 1   (None / -1 authors are dropped)
        pub_days:   sorted list[int] (days since 1900-01-01)
        first_day:  int
        domain:     str | None  (singleton domain key in DOMAINS, else None)
    """
    out = []
    for a in raw_authors:
        g = a.get("gender")
        if g not in (0, 1):
            continue
        dates = a.get("pub_dates") or []
        if not dates:
            continue
        days = [date_str_to_days(d) for d in dates]
        # days are already sorted in compact_data but guard against it.
        if any(days[i] > days[i + 1] for i in range(len(days) - 1)):
            days.sort()
        doms = a.get("domains") or []
        singleton_domain = doms[0] if len(doms) == 1 and doms[0] in NON_OVERALL_DOMAINS else None
        out.append({
            "gender": g,
            "pub_days": days,
            "first_day": days[0],
            "domain": singleton_domain,
        })
    return out


def author_in_domain(author, domain):
    if domain == "overall":
        return True
    return author["domain"] == domain


# ---------------------------------------------------------------------------
# Step A: per-(domain, gender) pub_interval_threshold from survival probability
# ---------------------------------------------------------------------------

def compute_thresholds(authors):
    """Return thresholds[domain][gender_key] = int days, or None.

    Implements `calc_survival_probability_for_publication_interval` +
    `calc_threshold_for_publication_interval(p = 0.02)` jointly: for each
    (domain, gender) bucket of inter-pub intervals, the threshold is the
    smallest y where P(interval >= y) < 0.02.
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for a in authors:
        days = a["pub_days"]
        if len(days) < 2:
            continue
        gkey = "female" if a["gender"] == 0 else "male"
        intvls = [days[i + 1] - days[i] for i in range(len(days) - 1)]
        buckets["overall"][gkey].extend(intvls)
        if a["domain"] is not None:
            buckets[a["domain"]][gkey].extend(intvls)

    result = {d: {"female": None, "male": None} for d in DOMAINS}
    for domain in DOMAINS:
        for gkey in ("female", "male"):
            intvls = buckets[domain].get(gkey, [])
            if not intvls:
                continue
            n = len(intvls)
            target = SURVIVAL_PROB_THRESHOLD * n
            sorted_intvls = sorted(intvls)
            # smallest y with (# intervals >= y) < target.
            # # >= y = n - bisect_left(sorted_intvls, y).
            # Binary search over y space using a derived monotonic property:
            # bisect_left is non-decreasing in y, so n - bisect_left is
            # non-increasing. We want smallest y where it < target.
            # Equivalent: smallest y such that bisect_left > n - target.
            # bisect_left(sorted_intvls, y) gives index of first element >= y.
            # We want the smallest y making (n - idx) < target,
            # i.e. idx > n - target. The minimum such idx is
            # ceil(n - target + epsilon) so y = sorted_intvls[idx] + 1
            # but the paper iterates over all y from 0 onwards. The result is
            # equivalent to: pick the smallest y > sorted_intvls[i*] where
            # i* is the smallest index with n - i* < target.
            i_star = -1
            for i in range(n):
                if n - i < target:
                    i_star = i
                    break
            if i_star == -1:
                # All survival probabilities >= target; pick max_interval + 1
                # as in the paper's loop falling off the end.
                threshold = sorted_intvls[-1] + 1
            else:
                # At this index the count is below target; the y boundary is
                # one larger than the (i_star - 1)-th value (the largest y
                # still satisfying P >= target).
                threshold = sorted_intvls[i_star - 1] + 1 if i_star > 0 else 0
            result[domain][gkey] = int(threshold)
    return result


# ---------------------------------------------------------------------------
# Step B: current pyramids + per-year newcomer counts (multi-year incremental)
# ---------------------------------------------------------------------------

CAREER_MAX_DAYS = CAREER_MAX_YEARS * 365


def compute_pyramids(authors, thresholds, years=YEARS):
    """Build per-(domain, year) pyramid histograms and newcomer counts.

    pyramids[domain][year][gkey][cum_prod] = int
    newcomers[domain][year][gkey] = int   (active at year, inactive at year-1)
    totals[domain][year][f"total_{gkey}"] = int
    """
    pyramids = {
        d: {y: {"female": defaultdict(int), "male": defaultdict(int)} for y in years}
        for d in DOMAINS
    }
    newcomers = {d: {y: {"female": 0, "male": 0} for y in years} for d in DOMAINS}

    sorted_years = sorted(years)
    year_prev_map = {y: y - 1 for y in sorted_years}

    for a in authors:
        gkey = "female" if a["gender"] == 0 else "male"
        pub_days = a["pub_days"]
        first_day = a["first_day"]
        if not pub_days:
            continue

        contrib_domains = ["overall"]
        if a["domain"] is not None:
            contrib_domains.append(a["domain"])

        for domain in contrib_domains:
            threshold_days = thresholds[domain][gkey]
            if threshold_days is None or threshold_days < 0:
                continue

            # Incremental scan: process all pubs once across the year loop.
            idx = 0
            valid_count = 0
            last_valid_day = None
            prev_active = False

            # Iterate over (warmup_year, sorted_years...): the warmup year
            # establishes prev_active for the first output year so that the
            # newcomer count at sorted_years[0] (e.g. 1970) is not inflated.
            warmup_year = sorted_years[0] - 1
            first_output_year = sorted_years[0]
            for year in [warmup_year] + list(sorted_years):
                end_day = YEAR_END_DAYS[year]
                deadline = end_day - threshold_days

                while idx < len(pub_days) and pub_days[idx] <= end_day:
                    p = pub_days[idx]
                    if last_valid_day is None or (p - last_valid_day) <= threshold_days:
                        valid_count += 1
                    else:
                        valid_count = 1  # reset
                    last_valid_day = p
                    idx += 1

                # Apply paper filters (i)-(iv)
                if idx == 0:
                    is_active = False
                elif end_day - first_day > CAREER_MAX_DAYS:
                    is_active = False  # career exceeds 40 yr
                elif idx > PUB_THRESHOLD:
                    is_active = False  # exceeds pub limit
                elif last_valid_day is None or last_valid_day < deadline:
                    is_active = False
                elif valid_count <= 0:
                    is_active = False
                else:
                    is_active = True

                if year >= first_output_year and is_active:
                    pyramids[domain][year][gkey][valid_count] += 1
                    if not prev_active:
                        newcomers[domain][year][gkey] += 1
                prev_active = is_active

    return pyramids, newcomers


def build_pyramid_payload(country, thresholds, pyramids, newcomers):
    payload = {
        "country": country,
        "thresholds": thresholds,
        "year_range": [TARGET_YEAR_START, TARGET_YEAR_END],
        "data": {},
    }
    for domain in DOMAINS:
        domain_data = {}
        for year in YEARS:
            f_hist = pyramids[domain][year]["female"]
            m_hist = pyramids[domain][year]["male"]
            domain_data[str(year)] = {
                # JSON object keys are strings; counts are ints.
                "female": {str(k): v for k, v in f_hist.items()},
                "male": {str(k): v for k, v in m_hist.items()},
                "total_female": sum(f_hist.values()),
                "total_male": sum(m_hist.values()),
                "newcomers_female": newcomers[domain][year]["female"],
                "newcomers_male": newcomers[domain][year]["male"],
            }
        payload["data"][domain] = domain_data
    return payload


# ---------------------------------------------------------------------------
# Step C: projection (Markov chain from base_year-1 / base_year transitions)
# ---------------------------------------------------------------------------

def compute_author_cum_prod_at_two_years(author, threshold_days, year_a, year_b):
    """Compute (cum_prod_a, cum_prod_b) for an author at year_a < year_b.

    cum_prod is None if the author is not active (per paper filters) at that
    year. Mirrors the relevant branches of `calc_productive_people_pyramid`.
    """
    pub_days = author["pub_days"]
    first_day = author["first_day"]
    if not pub_days:
        return (None, None)

    def state_at(year):
        end_day = YEAR_END_DAYS[year]
        deadline = end_day - threshold_days
        # Find pubs up to end_day.
        i_end = bisect.bisect_right(pub_days, end_day)
        if i_end == 0:
            return None
        if end_day - first_day > CAREER_MAX_DAYS:
            return None
        if i_end > PUB_THRESHOLD:
            return None
        last_pub = pub_days[i_end - 1]
        if last_pub < deadline:
            return None
        # Cumulative productivity with reset.
        valid = 1
        last_valid = pub_days[0]
        for p in pub_days[1:i_end]:
            if (p - last_valid) > threshold_days:
                valid = 1
            else:
                valid += 1
            last_valid = p
        if valid <= 0:
            return None
        return valid

    return (state_at(year_a), state_at(year_b))


def compute_projection_for_base(authors, thresholds, domain, base_year, target_year,
                              years_to_keep=None):
    """Markov-chain projection for one (domain, base_year) combination.

    Returns dict[year_str] -> {"female": {n: count}, "male": {n: count}, ...}.
    The Markov chain still iterates every year internally (to reach
    target_year correctly); only the years listed in years_to_keep are
    retained in the returned payload. If years_to_keep is None, all years
    in [base_year, target_year] are kept.
    """
    if years_to_keep is None:
        years_to_keep = set(range(base_year, target_year + 1))
    else:
        years_to_keep = set(years_to_keep)
    result = {y: {"female": {}, "male": {}, "total_female": 0.0, "total_male": 0.0}
              for y in sorted(years_to_keep)}

    for gkey, gender in (("female", 0), ("male", 1)):
        threshold_days = thresholds[domain][gkey]
        if threshold_days is None or threshold_days < 0:
            continue

        cp_prev = {}  # author_idx -> cum_prod at base_year - 1
        cp_curr = {}  # author_idx -> cum_prod at base_year

        for idx, a in enumerate(authors):
            if a["gender"] != gender:
                continue
            if not author_in_domain(a, domain):
                continue
            cp_a, cp_b = compute_author_cum_prod_at_two_years(
                a, threshold_days, base_year - 1, base_year
            )
            if cp_a is not None:
                cp_prev[idx] = cp_a
            if cp_b is not None:
                cp_curr[idx] = cp_b

        if not cp_curr:
            continue

        max_n = 0
        if cp_prev:
            max_n = max(max_n, max(cp_prev.values()))
        if cp_curr:
            max_n = max(max_n, max(cp_curr.values()))
        if max_n == 0:
            continue

        # Transition matrix prob[n1][n2]; n2 = 0 means inactive next year.
        prob = {n1: [0.0] * (max_n + 1) for n1 in range(1, max_n + 1)}
        for idx, n1 in cp_prev.items():
            n2 = cp_curr.get(idx, 0)
            prob[n1][n2] += 1
        for n1, row in prob.items():
            total = sum(row)
            if total > 0:
                prob[n1] = [v / total for v in row]

        # Newcomers at base_year: inactive at base_year - 1 but active now.
        newcomer = [0.0] * (max_n + 1)  # index 0 unused
        for idx, n2 in cp_curr.items():
            if idx not in cp_prev and 1 <= n2 <= max_n:
                newcomer[n2] += 1

        # Initial state = pyramid at base_year.
        state = [0.0] * (max_n + 1)
        for n in cp_curr.values():
            if 1 <= n <= max_n:
                state[n] += 1

        # Store base_year (if retained).
        if base_year in result:
            for n in range(1, max_n + 1):
                if state[n] > 0:
                    result[base_year][gkey][str(n)] = state[n]
            result[base_year][f"total_{gkey}"] = sum(state[1:])

        # Iterate t+1, t+2, ...
        for offset in range(1, target_year - base_year + 1):
            next_state = list(newcomer)  # copy
            for k in range(1, max_n + 1):
                if state[k] == 0:
                    continue
                row = prob[k]
                s_k = state[k]
                # n=0 is the "drop out" sink and isn't part of the next active
                # pool. We accumulate only into n in 1..max_n.
                for n in range(1, max_n + 1):
                    next_state[n] += s_k * row[n]
            state = next_state
            y = base_year + offset
            if y not in result:
                continue
            for n in range(1, max_n + 1):
                if state[n] > 0:
                    result[y][gkey][str(n)] = state[n]
            result[y][f"total_{gkey}"] = sum(state[1:])

    return result


def compute_projections(authors, thresholds):
    out = {d: {} for d in DOMAINS}
    for domain in DOMAINS:
        for base_year in PROJECTION_BASE_YEARS:
            out[domain][f"base_{base_year}"] = compute_projection_for_base(
                authors, thresholds, domain, base_year, PROJECTION_END_YEAR,
                years_to_keep=PROJECTION_YEARS_TO_KEEP,
            )
    return out


def build_projection_payload(country, thresholds, projections):
    return {
        "country": country,
        "thresholds": thresholds,
        "base_years": list(PROJECTION_BASE_YEARS),
        "target_year": PROJECTION_END_YEAR,
        "data": {
            domain: {
                base_key: {str(y): yd for y, yd in years_dict.items()}
                for base_key, years_dict in domain_dict.items()
            }
            for domain, domain_dict in projections.items()
        },
    }


# ---------------------------------------------------------------------------
# Step D: per-country timeline (gendered active counts by publication year)
# ---------------------------------------------------------------------------

def compute_timeline_from_pyramids(pyramid_payloads, projection_payloads):
    """Per-country, per-year active-author totals derived from the pyramids.

    For each country and year, returns [n_female, n_male] = total_female,
    total_male from the overall-domain pyramid. Historical years come from
    `pyramid_payloads` (1970-2023); projection years come from
    `projection_payloads` (2024-2050 under base_2023). Using the pyramid totals
    keeps the Trends, Pyramid, and Comparison pages on a single common
    definition of "active author".
    """
    out = {}
    for code, pyr in pyramid_payloads.items():
        series = {}
        for year_str, yd in pyr["data"]["overall"].items():
            series[year_str] = [int(round(yd["total_female"])),
                                int(round(yd["total_male"]))]
        fc = projection_payloads.get(code) if projection_payloads else None
        if fc is not None:
            fc_overall = fc["data"]["overall"].get("base_2023", {})
            for year_str, yd in fc_overall.items():
                series[year_str] = [int(round(yd["total_female"])),
                                    int(round(yd["total_male"]))]
        out[code] = series
    return out


# ---------------------------------------------------------------------------
# Aggregated outputs (rankings, scatter)
# ---------------------------------------------------------------------------

def _hist_mean(hist):
    """Mean cumulative productivity from histogram {n_str: count}."""
    total = 0
    weighted = 0.0
    for n_str, c in hist.items():
        n = int(n_str)
        total += c
        weighted += n * c
    return weighted / total if total > 0 else 0.0


def compute_rankings(pyramid_payloads, year=TARGET_YEAR_END):
    """Summary metrics per country at the given year (default 2023)."""
    countries = []
    for code, payload in pyramid_payloads.items():
        year_data = payload["data"]["overall"].get(str(year))
        if not year_data:
            continue
        f_hist = year_data["female"]
        m_hist = year_data["male"]
        n_f = year_data["total_female"]
        n_m = year_data["total_male"]
        total = n_f + n_m
        if total == 0:
            continue
        mean_f = _hist_mean(f_hist)
        mean_m = _hist_mean(m_hist)
        new_f = year_data["newcomers_female"]
        new_m = year_data["newcomers_male"]
        gap = (mean_f - mean_m) / mean_m if mean_m > 0 else 0.0
        countries.append({
            "code": code,
            "total_active": total,
            "n_female": n_f,
            "n_male": n_m,
            "female_ratio": n_f / total,
            "mean_cum_productivity": (mean_f * n_f + mean_m * n_m) / total,
            "mean_cum_productivity_female": mean_f,
            "mean_cum_productivity_male": mean_m,
            "inflow_ratio": (new_f + new_m) / total,
            "gender_gap": gap,
        })
    return {"year": year, "countries": countries}


def compute_scatter(pyramid_payloads, projection_payloads=None):
    """Year × country inflow_ratio + gender_gap series for the fig5-style scatter.

    Historical years (1970-2023) come from the empirical pyramids. If
    projection_payloads is provided, the series is extended with projected
    years (2024-2050) under the model's assumption that the base year's
    newcomer counts (2023) repeat each subsequent year.
    """
    series = {}
    for code, payload in pyramid_payloads.items():
        points = []

        # ---- Historical ----
        for year in YEARS:
            yd = payload["data"]["overall"].get(str(year))
            if not yd:
                continue
            n_f = yd["total_female"]
            n_m = yd["total_male"]
            total = n_f + n_m
            if total == 0:
                continue
            mean_f = _hist_mean(yd["female"])
            mean_m = _hist_mean(yd["male"])
            gap = (mean_f - mean_m) / mean_m if mean_m > 0 else 0.0
            inflow = (yd["newcomers_female"] + yd["newcomers_male"]) / total
            points.append({
                "year": year, "inflow_ratio": inflow,
                "gender_gap": gap, "projected": False,
            })

        # ---- Projection ----
        if projection_payloads is not None:
            fc = projection_payloads.get(code)
            base_yd = payload["data"]["overall"].get(str(TARGET_YEAR_END))
            if fc is not None and base_yd is not None:
                # The projection model holds newcomer counts constant at the
                # base year's empirical value.
                new_f_const = base_yd["newcomers_female"]
                new_m_const = base_yd["newcomers_male"]
                fc_overall = fc["data"]["overall"].get("base_2023", {})
                for year_str, yd in fc_overall.items():
                    year = int(year_str)
                    n_f = yd["total_female"]
                    n_m = yd["total_male"]
                    total = n_f + n_m
                    if total == 0:
                        continue
                    mean_f = _hist_mean(yd["female"])
                    mean_m = _hist_mean(yd["male"])
                    gap = (mean_f - mean_m) / mean_m if mean_m > 0 else 0.0
                    inflow = (new_f_const + new_m_const) / total
                    points.append({
                        "year": year, "inflow_ratio": inflow,
                        "gender_gap": gap, "projected": True,
                    })

        points.sort(key=lambda p: p["year"])
        series[code] = points

    all_years = sorted({p["year"] for cc in series for p in series[cc]})
    return {"years": all_years, "data": series}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_compact(country):
    with gzip.open(COMPACT_DIR + f"{country}.pkl.gz", "rb") as f:
        return pickle.load(f)


def build_countries_meta(country_codes):
    out = []
    for code in country_codes:
        out.append({
            "code": code,
            "name": COUNTRY_NAME.get(code, code),
            "region": REGION.get(code, "Unknown"),
        })
    # Group by region (alphabetical within), then ship as flat list ordered
    # accordingly so the frontend can build the dropdown directly.
    out.sort(key=lambda x: (x["region"], x["name"]))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR + "pyramids/", exist_ok=True)
    os.makedirs(OUT_DIR + "pyramids_projection/", exist_ok=True)

    with open(COMPACT_DIR + "selected_countries.json") as f:
        countries = json.load(f)

    explicit = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip_existing = "--skip-existing" in sys.argv
    if explicit:
        countries = [c for c in countries if c in explicit]

    pyramid_payloads = {}
    projection_payloads = {}

    for country in countries:
        out_pyr = OUT_DIR + f"pyramids/{country}.json"
        out_fcst = OUT_DIR + f"pyramids_projection/{country}.json"
        if skip_existing and os.path.exists(out_pyr) and os.path.exists(out_fcst):
            print(f"[{country}] SKIP (outputs exist)", flush=True)
            with open(out_pyr) as f:
                pyramid_payloads[country] = json.load(f)
            with open(out_fcst) as f:
                projection_payloads[country] = json.load(f)
            continue

        print(f"\n=== {country} ===", flush=True)
        t0 = time.time()

        compact = load_compact(country)
        authors = preprocess_authors(compact["authors"])
        n_auth = len(authors)
        print(f"  {n_auth:,} gendered authors", flush=True)

        # Step A: thresholds.
        t = time.time()
        thresholds = compute_thresholds(authors)
        print(f"  thresholds in {time.time() - t:.1f}s", flush=True)
        for d in DOMAINS:
            print(f"    {d:20s} F={thresholds[d]['female']} M={thresholds[d]['male']}", flush=True)

        # Step B: current pyramids + newcomers.
        t = time.time()
        pyramids, newcomers = compute_pyramids(authors, thresholds)
        print(f"  pyramids in {time.time() - t:.1f}s", flush=True)
        pyramid_payload = build_pyramid_payload(country, thresholds, pyramids, newcomers)
        with open(out_pyr, "w") as f:
            json.dump(pyramid_payload, f, separators=(",", ":"))
        size_kb = os.path.getsize(out_pyr) / 1024
        print(f"  wrote {out_pyr} ({size_kb:.0f} KB)", flush=True)
        pyramid_payloads[country] = pyramid_payload

        # Step C: projections.
        t = time.time()
        projections = compute_projections(authors, thresholds)
        print(f"  projections in {time.time() - t:.1f}s", flush=True)
        projection_payload = build_projection_payload(country, thresholds, projections)
        with open(out_fcst, "w") as f:
            json.dump(projection_payload, f, separators=(",", ":"))
        size_kb = os.path.getsize(out_fcst) / 1024
        print(f"  wrote {out_fcst} ({size_kb:.0f} KB)", flush=True)
        projection_payloads[country] = projection_payload

        print(f"  TOTAL {country}: {time.time() - t0:.1f}s", flush=True)

    # Step E: aggregated outputs.
    print("\n=== aggregating ===", flush=True)
    timelines = compute_timeline_from_pyramids(pyramid_payloads, projection_payloads)
    with open(OUT_DIR + "timeline.json", "w") as f:
        json.dump(timelines, f, separators=(",", ":"))

    rankings = compute_rankings(pyramid_payloads, year=TARGET_YEAR_END)
    with open(OUT_DIR + "rankings.json", "w") as f:
        json.dump(rankings, f, separators=(",", ":"))

    scatter = compute_scatter(pyramid_payloads, projection_payloads)
    with open(OUT_DIR + "scatter.json", "w") as f:
        json.dump(scatter, f, separators=(",", ":"))

    countries_meta = build_countries_meta(countries)
    with open(OUT_DIR + "countries.json", "w") as f:
        json.dump(countries_meta, f, ensure_ascii=False, indent=2)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
