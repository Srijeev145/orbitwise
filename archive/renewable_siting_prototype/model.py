"""
OrbitWise / Track A - from satellite pixels to an investment decision.

For every 0.1-degree cell we compute, for a 50 MW solar plant and a 50 MW wind farm:
  * exclusion screen (forest, water, built-up, steep terrain)
  * energy yield  (P50 and P90 - lenders finance on P90)
  * LCOE in Rs/kWh, NPV at the current auction tariff, project IRR, payback
  * a 0-100 site score and a plain-language "why" for the top sites

All assumptions live in ASSUMPTIONS so the team / judges can change them live.
"""
from math import gamma
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

ASSUMPTIONS = {
    "plant_mw": 50,
    "life_yr": 25,
    "wacc": 0.095,                 # real-ish Indian utility RE cost of capital
    "grid_capex_cr_per_km": 1.1,   # dedicated 66/110 kV evacuation line, Rs crore per km
    "solar": {
        "capex_cr_per_mw": 4.0,    # per MW(AC) incl. DC overloading, Rs crore (2025-26)
        "dc_ac": 1.35,             # MWp of modules per MW of inverter (Indian norm 1.3-1.4)
        "clipping": 0.02,
        "opex_lakh_per_mw_yr": 6.0,
        "pr": 0.80,                # performance ratio before temperature losses
        "tilt_gain": 1.08,         # fixed-tilt plane-of-array gain over GHI at ~15 N
        "temp_coeff": -0.0035,     # per deg C above 25 (cell ~ ambient + 20)
        "degradation": 0.005,
        "tariff": 2.60,            # Rs/kWh, recent SECI / state auctions
        "max_slope_deg": 8,
        "ha_per_mw": 1.8,
        "model_uncertainty": 0.04, # irradiance-to-energy model sigma
    },
    "wind": {
        "capex_cr_per_mw": 6.6,
        "opex_lakh_per_mw_yr": 9.0,
        "hub_m": 120,
        "turbine_mw": 3.3, "cut_in": 3.0, "rated_ws": 12.0, "cut_out": 25.0,
        "losses": 0.15,            # wake, availability, electrical
        "degradation": 0.002,
        "tariff": 3.20,
        "max_slope_deg": 20,
        "model_uncertainty": 0.10,
    },
    "land_lakh_per_acre": {"bare": 2.0, "scrub": 3.0, "crop": 9.0},
}
A = ASSUMPTIONS


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def annuity_factor_degr(r, n, d):
    """Present value of an energy stream that degrades by d per year."""
    t = np.arange(1, n + 1)
    return ((1 - d) ** (t - 1) / (1 + r) ** t).sum()


def irr(cashflows):
    lo, hi = -0.5, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        npv = sum(c / (1 + mid) ** i for i, c in enumerate(cashflows))
        lo, hi = (mid, hi) if npv > 0 else (lo, mid)
    return mid


# --------------------------------------------------------------------------- screens
def exclusion_reason(df, tech):
    s = A[tech]
    reasons = np.full(len(df), "", dtype=object)
    rules = [
        (df.lc_nodata + df.lc_water > 0.5, "sea / water body"),
        (df.lc_tree > 0.5, "forest cover (ecologically sensitive)"),
        (df.lc_built > 0.25, "urban area"),
        (df.slope_deg > s["max_slope_deg"], f"terrain too steep (> {s['max_slope_deg']} deg)"),
    ]
    for mask, why in rules:
        reasons = np.where((reasons == "") & mask.fillna(False).to_numpy(), why, reasons)
    return reasons


def developable_ha(df):
    """Land a developer could realistically lease: scrub, grass, bare + a share of farmland."""
    cell_ha = (0.1 * 110.6) * (0.1 * 111.3 * np.cos(np.radians(df.lat))) * 100
    usable = df.lc_bare + df.lc_shrub + df.lc_grass + 0.25 * df.lc_crop
    return cell_ha * usable


def land_cost_cr_per_mw(df, ha_per_mw):
    p = A["land_lakh_per_acre"]
    cheap = df.lc_bare + df.lc_shrub + df.lc_grass
    share_crop = (0.25 * df.lc_crop) / (cheap + 0.25 * df.lc_crop + 1e-9)
    lakh_per_acre = (1 - share_crop) * p["scrub"] + share_crop * p["crop"]
    return lakh_per_acre * ha_per_mw * 2.471 / 100  # lakh -> crore


# --------------------------------------------------------------------------- energy
def solar_yield(df):
    s = A["solar"]
    t_cell = df.t2m + 20
    pr = s["pr"] * (1 + s["temp_coeff"] * (t_cell - 25))
    kwh_per_kwp = df.ghi * 365 * s["tilt_gain"] * pr
    kwh_per_kwac = kwh_per_kwp * s["dc_ac"] * (1 - s["clipping"])
    sigma = np.sqrt(df.ghi_cv ** 2 + s["model_uncertainty"] ** 2)
    return kwh_per_kwp, kwh_per_kwac, kwh_per_kwac * (1 - 1.282 * sigma)


def wind_cf(df):
    w = A["wind"]
    alpha = np.clip(np.log(df.ws50 / df.ws10) / np.log(50 / 10), 0.10, 0.35)
    v_hub = df.ws50 * (w["hub_m"] / 50) ** alpha
    # Weibull k = 2 (Rayleigh); integrate a generic pitch-regulated power curve
    k = 2.0
    c = v_hub / gamma(1 + 1 / k)
    v = np.linspace(0, 30, 301)
    pc = np.clip((v ** 3 - w["cut_in"] ** 3) / (w["rated_ws"] ** 3 - w["cut_in"] ** 3), 0, 1)
    pc[(v < w["cut_in"]) | (v > w["cut_out"])] = 0
    cf = []
    for ci in c.to_numpy():
        pdf = (k / ci) * (v / ci) ** (k - 1) * np.exp(-(v / ci) ** k)
        cf.append(np.trapezoid(pc * pdf, v))
    cf = np.array(cf) * (1 - w["losses"])
    return v_hub, cf


# --------------------------------------------------------------------------- finance
def finance(capex_cr_mw, opex_lakh_mw, kwh_per_kw_p50, kwh_per_kw_p90, tariff, degr):
    r, n, mw = A["wacc"], A["life_yr"], A["plant_mw"]
    pv_energy = annuity_factor_degr(r, n, degr)        # per kWh of year-1 output
    capex = capex_cr_mw * 1e7                          # Rs per MW
    opex = opex_lakh_mw * 1e5
    e1 = kwh_per_kw_p50 * 1000                         # kWh per MW in year 1
    pv_opex = opex * ((1 - (1 + r) ** -n) / r)
    lcoe = (capex + pv_opex) / (e1 * pv_energy)
    npv_mw = tariff * e1 * pv_energy - capex - pv_opex
    npv_p90_mw = tariff * kwh_per_kw_p90 * 1000 * pv_energy - capex - pv_opex
    rev1 = tariff * e1 - opex
    out = pd.DataFrame({
        "lcoe": lcoe,
        "npv_cr": npv_mw * mw / 1e7,
        "npv_p90_cr": npv_p90_mw * mw / 1e7,
        "payback_yr": capex / np.maximum(rev1, 1),
        "capex_total_cr": capex_cr_mw * mw,
    })
    out["irr"] = [irr([-cx * 1e7] + [(tariff * e * 1000 * (1 - degr) ** t - opex) for t in range(n)])
                  for cx, e in zip(np.broadcast_to(capex_cr_mw, len(out)), kwh_per_kw_p50)]
    return out


def evaluate(df):
    df = df.copy()
    df["dev_ha"] = developable_ha(df)
    grid_cr_mw = A["grid_capex_cr_per_km"] * df.grid_km / A["plant_mw"]

    # ---- solar
    s = A["solar"]
    kwp, p50, p90 = solar_yield(df)
    capex = s["capex_cr_per_mw"] + grid_cr_mw + land_cost_cr_per_mw(df, s["ha_per_mw"]) \
        + 0.02 * s["capex_cr_per_mw"] * np.clip(df.slope_deg.fillna(0) - 3, 0, None)  # grading
    f = finance(capex.to_numpy(), s["opex_lakh_per_mw_yr"], p50.to_numpy(), p90.to_numpy(),
                s["tariff"], s["degradation"])
    df["solar_kwh_kwp"] = kwp
    df["solar_cuf"], df["solar_cuf_p90"] = p50 / 8760, p90 / 8760
    for c in f:
        df[f"solar_{c}"] = f[c].to_numpy()
    df["solar_excluded"] = exclusion_reason(df, "solar")
    df.loc[df.dev_ha < s["ha_per_mw"] * A["plant_mw"], "solar_excluded"] = \
        df.solar_excluded.where(df.solar_excluded != "", "not enough open land")

    # ---- wind
    w = A["wind"]
    v_hub, cf = wind_cf(df)
    e_kw = cf * 8760
    sig = w["model_uncertainty"]
    capex_w = w["capex_cr_per_mw"] + grid_cr_mw + 0.3 * land_cost_cr_per_mw(df, 0.5)
    f = finance(capex_w.to_numpy(), w["opex_lakh_per_mw_yr"], e_kw, e_kw * (1 - 1.282 * sig),
                w["tariff"], w["degradation"])
    df["wind_v_hub"], df["wind_cf"] = v_hub, cf
    for c in f:
        df[f"wind_{c}"] = f[c].to_numpy()
    df["wind_excluded"] = exclusion_reason(df, "wind")

    # ---- scores (0-100, percentile of P90 NPV among eligible cells)
    for tech in ("solar", "wind"):
        ok = df[f"{tech}_excluded"] == ""
        r = df.loc[ok, f"{tech}_npv_p90_cr"].rank(pct=True) * 100
        df[f"{tech}_score"] = 0.0
        df.loc[ok, f"{tech}_score"] = r.round(1)
    df["best_tech"] = np.where(df.solar_npv_p90_cr >= df.wind_npv_p90_cr, "solar", "wind")
    return df


def explain(row, tech, med):
    """Plain-language reasons a site ranks where it does (vs the regional median)."""
    bits = []
    if tech == "solar":
        d = (row.ghi / med.ghi - 1) * 100
        bits.append(f"sunlight {row.ghi:.2f} kWh/m2/day ({d:+.1f}% vs state median)")
        bits.append(f"specific yield {row.solar_kwh_kwp:.0f} kWh/kWp, AC CUF {row.solar_cuf * 100:.1f}% "
                    f"(P90 {row.solar_cuf_p90 * 100:.1f}%, from {int(med.years)} yrs of satellite data)")
    else:
        bits.append(f"hub-height wind {row.wind_v_hub:.1f} m/s, capacity factor {row.wind_cf * 100:.0f}%")
    bits.append(f"{row.grid_km:.0f} km to grid connection")
    if not np.isnan(row.slope_deg):
        bits.append(f"slope {row.slope_deg:.1f} deg")
    open_land = row.lc_bare + row.lc_shrub + row.lc_grass
    bits.append(f"{open_land * 100:.0f}% scrub/grass/bare land, {row.lc_crop * 100:.0f}% cropland")
    return "; ".join(bits)


if __name__ == "__main__":
    from features import build_table
    df = build_table()
    years = df.attrs.get("ghi_years", 20)
    res = evaluate(df)
    res.to_csv(DATA / "results.csv", index=False)
    med = res.median(numeric_only=True)
    med["years"] = years
    for tech in ("solar", "wind"):
        ok = res[res[f"{tech}_excluded"] == ""]
        print(f"\n=== {tech.upper()}: {len(ok)} eligible cells of {len(res)} ===")
        print(res[f"{tech}_excluded"].value_counts().to_string())
        top = ok.sort_values(f"{tech}_npv_p90_cr", ascending=False).head(8)
        cols = ["lat", "lon", f"{tech}_lcoe", f"{tech}_npv_p90_cr", f"{tech}_irr", f"{tech}_payback_yr"]
        print(top[cols].round(3).to_string(index=False))
        print("why #1:", explain(top.iloc[0], tech, med))
