Make the **download list** concrete and “trader‑useful,” so you can pull exactly what you need from ERCOT’s MIS by **EMIL ID**. I’ll group these into:

1. **Core price feeds** (minimum viable dataset for Δ = RT − DA).
2. **RTC+B co‑optimization & scarcity pack** (the AS‑slope/ORDC intelligence).
3. **Context drivers** (load, wind/solar, weather, outages).
4. **Handy references** (mappings, trials).

> **Tip on naming:** in ERCOT EMIL, **NP4 = Day‑Ahead**, **NP6 = Real‑Time**, and you’ll sometimes see **“-RTCMT”** items for **Market Trials** (same shape as go‑live postings). Where ERCOT has published separate Trials versions, I note them.

---

## 1) Core price feeds (foundation for Δ = RT − DA)

* **NP4‑190‑CD — DAM Settlement Point Prices (SPP)**: DA prices at hubs, load zones, resource nodes. This is the “DA leg” of Δ. ([ERCOT][1])
* **NP4‑183‑CD — DAM Hourly LMPs (by electrical bus)**: DA nodal LMPs at bus granularity; useful when you need bus→node mapping precision. ([ERCOT][2])
* **NP6‑788‑CD — LMPs by Resource Nodes, Load Zones and Trading Hubs (5‑min)**: RT LMPs every successful SCED run. This is your high‑frequency “RT leg.” (Trials version exists as **NP6‑788‑RTCMT**.) ([ERCOT][3])
* **NP6‑905‑CD — Settlement Point Prices at Resource Nodes, Hubs and Load Zones (15‑min)**: RT **settlement** prices (interval‑weighted from 5‑min LMPs). Use for PnL, not just signals. ([ERCOT][4])
* **NP6‑970‑CD — RTD Indicative LMPs (look‑ahead)**: RTD’s projected LMPs for the study horizon; great for anticipating near‑term RT and sizing virtuals. ([ERCOT][5])

---

## 2) RTC+B co‑optimization & scarcity pack (AS curves, MCPCs, adders)

**Demand & supply curves for Ancillary Services (what sets scarcity slope):**

* **NP4‑212‑CD — Ancillary Service Demand Curves (ASDC)**: the hourly demand curves DAM/SCED use; extract slope/elasticity features. ([ERCOT][6])
* **NP4‑19‑CD — Aggregated Ancillary Service Offer Curve**: the supply side; pairs with ASDC for a market‑clearing picture. ([ERCOT][7])

**Capacity prices (MCPC) & their real‑time analogs:**

* **NP4‑188‑CD — DAM Clearing Prices for Capacity (MCPC)** by AS type. (Trials version is posted as **NP4‑188‑RTCMT** prior to go‑live.) ([ERCOT][8])
* **NP6‑329‑CD — RTD Indicative Real‑Time MCPC** (by AS type; every RTD). Use for *expected* RT capacity price pressure. ([ERCOT][9])
* **NP6‑331‑CD — Real‑Time Clearing Prices for Capacity by 15‑Minute Settlement Interval**: settlement‑interval MCPCs. ([ERCOT][10])
* **NP6‑332‑CD — Real‑Time Clearing Prices for Capacity by **SCED** Interval (5‑min)**: the high‑frequency MCPCs that underlie the 15‑minute aggregation. ([ERCOT][11])

**ORDC & Reliability Deployment Price Adders (what lifts energy prices under scarcity):**

* **NP6‑323‑CD — Real‑Time ORDC & Reliability Deployment Price Adders and Reserves by SCED Interval**: 5‑min on‑line & off‑line adders **and** reserve quantities; your best “scarcity fingerprint.” (Trials also posted as NP6‑323‑RTCMT.) ([ERCOT][12])
* **NP6‑324‑CD — ORDC & Reliability Deployment Prices for 15‑minute Settlement Interval**: interval‑averaged adders for settlement alignment. ([ERCOT][13])

**System supply capability that constrains AS (vital post‑RTC+B):**

* **NP6‑328‑CD — Total Capability of Resources Available to Provide Ancillary Service**: includes ESR duration/SOC limits—this is where co‑optimization gets real. ([ERCOT][14])

**Decomposition & congestion context (to read LMP correctly):**

* **NP6‑322‑CD — SCED System Lambda** (RT marginal energy component). ([ERCOT][15])
* **NP4‑523‑CD — DAM System Lambda**. ([ERCOT][16])
* **NP6‑86‑CD — SCED Shadow Prices & Binding Transmission Constraints** (RT congestion drivers). ([ERCOT][17])
* **NP4‑191‑CD — DAM Shadow Prices** (DA congestion drivers). ([ERCOT][18])
* **NP6‑327‑CD — LMP By SOG Including Price Adders** (sanity‑check that adders are included where they should be). ([ERCOT][19])

---

## 3) Context drivers (what moves prices)

**Load, distribution, and weather:**

* **NP3‑561‑CD — Seven‑Day Load Forecast by Weather Zone** (hourly). Baseline demand signal. ([ERCOT][20])
* **NP4‑159‑CD — Load Forecast Distribution Factors (LDF)**: convert zonal load forecasts down to buses; essential for nodal views. ([ERCOT][21])
* **NP4‑722‑CD — Weather Assumptions** (hourly temperatures by zone used in ERCOT processes). ([ERCOT][22])

**Wind / Solar actuals & forecasts (system + regional):**

* **NP4‑732‑CD — Wind Power Production – Hourly Actual & Forecasted (System/Regional)**. ([ERCOT][23])
* **NP4‑742‑CD — Wind Power Production – Hourly by Region** and **NP4‑743‑CD — Wind 5‑Minute Actual by Region** (for ramps). ([lists.ercot.com][24])
* **NP4‑737‑CD — Solar Power Production – Hourly Actual & Forecasted (System‑wide)**. ([ERCOT][25])
* **NP4‑745‑CD — Solar Power Production – Hourly by Region**, and **NP4‑746‑CD — Solar 5‑Minute Actual by Region**. ([ERCOT][26])

**Outages & adequacy (capacity headroom matters for scarcity):**

* **NP3‑233‑CD — Hourly Resource Outage Capacity** (aggregated by load zone; next 168 hours). ([ERCOT][27])
* **NP3‑157‑CD — Consolidated Transmission Outage Report** (approved/accepted/proposed/rejected in one place). ([ERCOT][28])
* **NP1‑346‑ER — Unplanned Resource Outages Report** (snapshot of forced/maintenance outages). ([ERCOT][29])
* **NP3‑763‑CD — Short‑Term System Adequacy Report (STAR)**: system & regional capacity margins; high‑signal for scarcity set‑ups. ([ERCOT][30])

---

## 4) Handy references & trials (for cleaning and testing)

* **NP4‑160‑SG — Settlement Points List & Electrical Buses Mapping** (IDs, names, and mappings). ([ERCOT][31])
* **NP4‑192‑CD — DAM Total Energy Purchased** and **NP4‑193‑CD — DAM Total Energy Sold** (useful cross‑checks on DA totals). ([ERCOT][32])
* **Trials you can use right now** (when available on MIS): **NP4‑183‑RTCMT** (DAM Hourly LMPs) and **NP6‑788‑RTCMT** (RT LMPs by Node/LZ/Hub). ([ERCOT][33])

> **Note on Trials vs “‑CD”**: ERCOT sometimes posts Market Trials with the **same numeric ID** but labeled “Market Trials,” and sometimes with a **“-RTCMT”** suffix. You can see both patterns in the LMP feeds above. Use the Trials variants for **sim / model rehearsal**, then switch to the **‑CD** postings at go‑live. ([ERCOT][14])

---

### What this covers (and why it’s enough)

* **Δ = RT − DA** by node/hub: NP4‑190/NP4‑183 versus NP6‑905/NP6‑788/NP6‑970 give you DA, realized RT, and RT look‑ahead. ([ERCOT][1])
* **Scarcity & co‑optimization**: AS **demand** (NP4‑212), **offers** (NP4‑19), **capacity prices** (NP4‑188, NP6‑329/331/332), and **ORDC/adders** (NP6‑323/324) give you the **slope** and **price lift** drivers you asked to encode. ([ERCOT][6])
* **Drivers**: load, weather, renewables (NP3‑561, NP4‑722, NP4‑732/742/737/745/743/746), plus **outages** and **STAR** (NP3‑233, NP3‑157, NP1‑346, NP3‑763) let you model regimes/risk. ([ERCOT][20])

---

## Copy‑paste checklist (just the IDs & names)

```
NP4-190-CD  DAM Settlement Point Prices
NP4-183-CD  DAM Hourly LMPs (bus-level)
NP6-788-CD  RT LMPs by Resource Nodes, Load Zones, Hubs (5-min)
NP6-905-CD  RT Settlement Point Prices (15-min)
NP6-970-CD  RTD Indicative LMPs (look-ahead)

NP4-212-CD  Ancillary Service Demand Curves (ASDC)
NP4-19-CD   Aggregated Ancillary Service Offer Curve
NP4-188-CD  DAM Clearing Prices for Capacity (MCPC)
NP6-329-CD  RTD Indicative MCPC
NP6-331-CD  RT MCPC by 15-minute Settlement Interval
NP6-332-CD  RT MCPC by SCED Interval (5-min)
NP6-323-CD  RT ORDC & Reliability Deployment Adders + Reserves (SCED)
NP6-324-CD  RT ORDC & Reliability Deployment Prices (15-min)
NP6-328-CD  Total Capability to Provide AS (includes ESR SOC/duration caps)
NP6-322-CD  SCED System Lambda (RT energy component)
NP4-523-CD  DAM System Lambda
NP6-86-CD   SCED Shadow Prices & Binding Constraints
NP4-191-CD  DAM Shadow Prices

NP3-561-CD  Seven-Day Load Forecast by Weather Zone
NP4-159-CD  Load Forecast Distribution Factors (LDF)
NP4-722-CD  Weather Assumptions (hourly temps by zone)
NP4-732-CD  Wind – Hourly Actual & Forecast (system/regional)
NP4-742-CD  Wind – Hourly by Region
NP4-743-CD  Wind – 5-minute Actual by Region
NP4-737-CD  Solar – Hourly Actual & Forecast (system-wide)
NP4-745-CD  Solar – Hourly by Region
NP4-746-CD  Solar – 5-minute Actual by Region
NP3-233-CD  Hourly Resource Outage Capacity
NP3-157-CD  Consolidated Transmission Outage Report
NP1-346-ER  Unplanned Resource Outages Report
NP3-763-CD  Short-Term System Adequacy Report (STAR)
NP4-160-SG  Settlement Points List & Electrical Buses Mapping

# Trials you can use now (when posted):
NP4-183-RTCMT  DAM Hourly LMPs (Trials)
NP6-788-RTCMT  RT LMPs by Node/LZ/Hub (Trials)
```

---

### Two tiny operational hints (so you don’t trip at the starting line)

* **Time alignment:** DA is hourly; RT is 5‑min (SCED) and settles at 15‑min. When you compute Δ, align:
  DA **HE** → RT **15‑min** SPPs (NP6‑905), and keep a separate 5‑min Δ stream (NP6‑788) for training/high‑freq features. ([ERCOT][4])
* **Trials vs Production:** for simulator work, prefer **RTCMT** items where available (same schemas, safe to practice). At go‑live, flip to the **‑CD** product IDs. ([ERCOT][14])

If you want, I can sketch a **folder tree + file naming convention** next (so your manual downloads don’t turn into a shoebox of CSVs), and a **first pass feature map** showing exactly which columns from each EMIL feed plug into your **“scarcity slope → price → action”** pipeline.

[1]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD&utm_source=chatgpt.com "NP4-190, DAM Settlement Point Prices"
[2]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-183-CD&utm_source=chatgpt.com "DAM Hourly LMPs"
[3]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-788-CD&utm_source=chatgpt.com "LMPs by Resource Nodes, Load Zones and Trading Hubs"
[4]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-905-CD&utm_source=chatgpt.com "NP6-905, Settlement Point Prices at Resource Nodes ..."
[5]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-86-CD&utm_source=chatgpt.com "SCED Shadow Prices and Binding Transmission Constraints"
[6]: https://www.ercot.com/files/docs/2025/06/25/ERCOT-TWG-2025-4-24-RTC-B-CDR_PD-Slides-for-Market-Presentation.pptx?utm_source=chatgpt.com "PowerPoint Presentation"
[7]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-19-CD&utm_source=chatgpt.com "Aggregated Ancillary Service Offer Curve"
[8]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-188-CD&utm_source=chatgpt.com "DAM Clearing Prices for Capacity"
[9]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-329-CD&utm_source=chatgpt.com "Data Product Details"
[10]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-331-CD&utm_source=chatgpt.com "Real-Time Clearing Prices for Capacity by 15-Minute ..."
[11]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-332-CD&utm_source=chatgpt.com "Real-Time Clearing Prices for Capacity by SCED Interval"
[12]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-323-CD&utm_source=chatgpt.com "Data Product Details"
[13]: https://www.ercot.com/files/docs/2025/04/07/RTCB_Market_Trials_Handbook_3_OpenLoop_RTC_SCED.docx?utm_source=chatgpt.com "Handbook #3"
[14]: https://www.ercot.com/mktinfo/rtm?utm_source=chatgpt.com "Real-Time Market"
[15]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-322-CD&utm_source=chatgpt.com "SCED System Lambda"
[16]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-523-CD&utm_source=chatgpt.com "DAM System Lambda"
[17]: https://www.ercot.com/files/docs/2025/04/21/6_RTCBTF_Reports_CDR_PD_04182025.pptx?utm_source=chatgpt.com "6. RTCBTF Reports CDR PD 04182025"
[18]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-191-CD&utm_source=chatgpt.com "DAM Shadow Prices"
[19]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-327-CD&utm_source=chatgpt.com "LMP By SOG Including Price Adders"
[20]: https://www.ercot.com/mp/data-products/data-product-details?id=np4-212-cd&utm_source=chatgpt.com "DAM and SCED Ancillary Service Demand Curves"
[21]: https://www.ercot.com/mktrules/issues/reports/nprr?utm_source=chatgpt.com "Nodal Protocol Revision Requests (NPRRs)"
[22]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-722-CD&utm_source=chatgpt.com "Weather Assumptions"
[23]: https://www.ercot.com/gridmktinfo/dashboards/combinedwindandsolar?utm_source=chatgpt.com "Combined Wind and Solar"
[24]: https://lists.ercot.com/cgi-bin/wa?A3=1704&B=--_000_3E87B359A497CF43BB3CB18506AECF8C80E8AED6CPW0089ercotcom_&E=quoted-printable&L=NOTICE_RELEASE_WHOLESALE&P=44892&T=text%2Fplain%3B+charset%3Dus-ascii&header=1&utm_source=chatgpt.com "LISTSERV - NOTICE_RELEASE_WHOLESALE Archives - lists.ercot ..."
[25]: https://www.ercot.com/mp/data-products/data-product-details?id=np4-737-cd&utm_source=chatgpt.com "Solar Power Production - Hourly Averaged Actual and ..."
[26]: https://www.ercot.com/services/comm/mkt_notices/M-B050422-03?utm_source=chatgpt.com "M-B050422-03 Update - Implementation of NPRR935 and ..."
[27]: https://www.ercot.com/mp/data-products/data-product-details?id=NP3-233-CD&utm_source=chatgpt.com "Hourly Resource Outage Capacity"
[28]: https://www.ercot.com/mp/data-products/data-product-details?id=NP3-157-CD&utm_source=chatgpt.com "Data Product Details"
[29]: https://www.ercot.com/mp/data-products/data-product-details?id=NP1-346-ER&utm_source=chatgpt.com "Unplanned Resource Outages Report"
[30]: https://www.ercot.com/mp/data-products/data-product-details?id=NP3-763-CD&utm_source=chatgpt.com "Short-Term System Adequacy Report"
[31]: https://www.ercot.com/files/docs/2010/10/05/05._all_issues___100110_final.xls?utm_source=chatgpt.com "Summary"
[32]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-192-CD&utm_source=chatgpt.com "DAM Total Energy Purchased"
[33]: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-183-RTCMT&utm_source=chatgpt.com "RTC Market Trials DAM Hourly LMPs"
