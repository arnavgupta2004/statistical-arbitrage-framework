> **Technical record.** This is the stage-by-stage engineering and research log that the final report
> ([README.md](../README.md)) summarises: every design decision, validation, mutation-testing result and correction, in
> the order it happened. Section numbers (`§9.4`, ...) refer to *this* file and are cited in the report as
> "TR §9.4". Statements written as "this commit" in the status table refer to the commit of that stage.
> The status table below is complete: all ten stages are done.

# Statistical Arbitrage Research Framework

A research framework for one question: **do apparent statistical-arbitrage relationships in equities
survive the methodological problems that routinely create false discoveries?** — parameter selection,
multiple testing, survivorship bias, look-ahead leakage, transaction costs, structural breaks and
out-of-sample evaluation.

This is a research project, not a trading system. Every strategy is a *hypothesis*; the deliverable is
the evidence for or against it, including the negative results. A high in-sample Sharpe ratio is not
the goal and is not reported as a finding.

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | Data pipeline, corporate actions, validation, point-in-time universe, leakage tooling | **done** |
| 2 | Statistical toolkit: ADF, Engle–Granger, Johansen, OU, half-life | **done** |
| 3 | Pair discovery on training data only | **done** |
| 4 | Pair strategy: hedge ratios, spread, z-score, entry/exit/stop | **done** |
| 5 | Walk-forward backtest, train/validation/holdout | **done** |
| 6 | Transaction costs: spread, commissions, impact | **done** |
| 7 | PCA statistical arbitrage | **done** |
| 8 | Multiple-testing correction, deflated Sharpe, reality check | **done** |
| 9 | Robustness: sensitivity, regimes, structural breaks, capacity | **done** |
| 10 | Final research report | **done** (README.md, docs/research_log.md, docs/figures/) |

Sections below marked *(Stage N)* are the specification for work that has not been done. They contain
no results, and none should be inferred from them.

## Repository layout

The specification sketches top-level packages (`data/`, `selection/`, …). Here they live under one
importable package, `statarb/`, so that a directory of stored *data* never shadows a Python package
named `data`; the sub-package names are unchanged.

```
configs/               YAML configuration and the ticker-alias table
statarb/
  config.py            validated (pydantic, extra=forbid) configuration
  data/
    sources/           yahoo, csv (manual Stooq-style files), synthetic (offline vendor for tests)
    download/          incremental refresh with restatement detection
    cleaning/          calendar, sanitising, audits, corporate-action arithmetic, alignment
    universe/          point-in-time membership, S&P 500 reconstruction, survivorship diagnostics
    storage/           Parquet files + DuckDB catalog
    leakage.py         reusable look-ahead detectors
    pipeline.py        the facade research code reads through
  selection/           adf, cointegration (Engle-Granger), johansen, hurst (exploratory only),
                       correlation, clustering, screening (pair discovery + empirical null)
  models/              ou, rolling, hedge_ratio (static/expanding/rolling), kalman, pca
  signals/             zscore (closed-form, time-varying hedge), pairs (entry/exit/stop state machine),
                       residuals (rolling PCA residual scores, position state machine)
  portfolio/           construction (leg weights, volatility targeting), neutral (sizing, dollar/factor-neutral projection)
  backtest/            pair_pnl (single-pair accounting), walkforward (folds, engine), metrics,
                       costs (spread, commission, impact, borrow; Corwin-Schultz), pca_walkforward (PCA book engine, placebo),
                       capacity (analytic capacity from cost components)
  research/            registry (append-only trial log)
  statistics/          bootstrap (stationary block bootstrap), sharpe (PSR, deflated Sharpe, minimum backtest
                       length), multiple_testing (Reality Check, SPA, Romano-Wolf, BH/BY, CSCV-PBO),
                       breaks (sup-F mean break, subspace overlap), regimes (causal regimes, regime Sharpe test)
  data/holdout.py      sealed-holdout guard and ledger
tests/                 pytest + hypothesis
experiments/           runnable experiments and their JSON results (registry: experiments/registry.jsonl)
```

Later stages extend `portfolio/`, `backtest/`, `statistics/` and add `research/` inside `statarb/`.

## 1. Data

### 1.1 Sources and their limits

| Source | Role | Notes |
|---|---|---|
| Yahoo Finance via `yfinance` | primary, daily OHLCV + dividends + splits | unofficial scraper of a public endpoint; personal/non-commercial research use; the local store must not be redistributed |
| Wikipedia (CC BY-SA) | S&P 500 constituents and change log | volunteer-maintained; see §2 for its weaknesses |
| Manually downloaded CSVs (Stooq format) | optional independent cross-check | Stooq's bulk endpoint sits behind an interactive browser check, so it is **not** scripted; files downloaded by hand can be dropped into a directory and compared with `compare_sources` |
| Synthetic vendor | tests and demos | deterministic latent world with known splits/dividends and a controllable "vendor date" |

### 1.2 What the vendor delivers, and why that matters

Yahoo's `Close`/`Open`/`High`/`Low` are **split-adjusted as of the download date**, `Volume` is scaled
the opposite way, and `Adj Close` additionally folds in every dividend *paid after each bar*. Two
consequences drive the design:

1. **Vendor history is restated retroactively.** A split after the last download rescales every earlier
   price. Appending new rows to old rows would splice two bases together and create a fake ~50 % "crash"
   at the join (there is a test that demonstrates it, `test_naive_append_would_have_spliced_two_bases`).
   Every incremental refresh therefore re-fetches a 10-day overlap window and compares it with the
   store; any close that moved by more than 1e-4 (relative) triggers a full re-download of that ticker.
   A full re-download is also forced every 90 days to catch restatements the overlap window cannot see.
2. **`Adj Close` leaks the future.** It is rescaled by dividends that had not yet been paid at the time
   of each earlier bar. It is never stored. The store keeps split-adjusted OHLCV plus the raw actions
   table, and dividend adjustment is computed here (§1.4). The vendor `Adj Close` is used only to
   *cross-check* our adjustment on each fresh download (`ADJ_CLOSE_MISMATCH`).

Only **finalised** sessions are stored: a bar downloaded while the market is open, or within two hours
of the close, is provisional and would be revised later.

### 1.3 Storage

One Parquet file per ticker and table under `var/store/` (git-ignored: it is regenerable and
vendor-licensed) plus a DuckDB catalog holding a per-ticker manifest and an append-only fetch log.
`store.sql("select … from prices")` queries all files at once. Writes are atomic
(temp file + `os.replace`); a failure for one ticker is logged and never aborts a run or touches
other data.

### 1.4 Corporate-action handling

With `C_t` the vendor's split-adjusted close, `D_t` the cash dividend on the ex-date and `k_s` a
split ratio (new/old shares):

* as-traded price: `raw_t = C_t · Π(k_s : s > t)`, exposed as `DataPipeline.raw_close`;
* total return, dividend reinvested at the ex-date close: `r_t = (C_t + D_t)/C_{t-1} − 1`;
* dividend-adjusted price `I_t = C_t · G_t`, `G_t = Π(1 + D_u/C_u : u ≤ t)`, so `I_t/I_{t-1} = 1 + r_t`
  exactly, and `I` back-adjusted from an anchor date uses no dividend after the anchor;
* a dividend whose ex-date bar is missing is credited to the next stored bar rather than lost.

### 1.5 Point-in-time reads

`DataPipeline.panel(tickers, start, end, as_of=None)` returns what could have been observed on
`as_of` (default: `end`). It removes every bar, dividend and split dated after `as_of` **and**
re-expresses the remaining prices on the split basis a vendor would have used on that day
(`F_T = Π(k_s : s > T)`). Returns are invariant to this rescaling; price *levels* are not, so any
level-based statistic (a spread `Y − βX`) computed on today's basis silently depends on splits that
had not happened yet.

The central test: a store downloaded at the end of a synthetic world, viewed as-of an earlier date,
must be **bit-identical** to a store that was actually downloaded on that earlier date — checked
around a split, with dividends, a late listing and a delisting
(`test_panel_as_of_equals_the_store_a_vendor_would_have_had_on_that_day`).

Missing data policy: **no imputation.** A missing bar is NaN and `observed` is False; a return exists
only between two consecutive observed sessions; prices are never forward-filled. Panels keep a stable
column set (a not-yet-listed name is an all-NaN column, listed in `panel.empty`).

### 1.6 Validation

Structural repairs at ingest are limited to changes that cannot alter economics (sorting, de-duplicating
keeping the last row, dropping rows with no valid close) and each is reported. Everything judgemental is
**flagged, not fixed**:

| Code | Detects |
|---|---|
| `MISSING_SESSIONS` (warn; error if a run exceeds 5) | gaps against the exchange calendar (`exchange_calendars`, incl. special closures) |
| `CALENDAR_MISMATCH`, `FUTURE_DATES` | rows on non-sessions; rows after the last finalised session |
| `STALE_PRICE` | ≥ 5 identical closes in a row |
| `EXTREME_RETURN` | \|log return\| > 0.5 not explained by a recorded split |
| `SPLIT_UNADJUSTED` | a recorded split whose ratio is still visible as a price jump |
| `POSSIBLE_UNRECORDED_SPLIT` | a jump matching a common split ratio with no split recorded |
| `OHLC_INCONSISTENT`, `ZERO_VOLUME` | bad bars; untradeable days |
| `ADJ_CLOSE_MISMATCH` | our dividend adjustment vs the vendor's |
| `VENDOR_RESTATEMENT` | a full re-download that differs from the stored history |

### 1.7 Look-ahead controls

`statarb/data/leakage.py` provides two detectors that any function of history can be put through:

* **truncation invariance** — `f(data[:t])[:t] == f(data)[:t]` for every cut `t`;
* **future-perturbation invariance** — overwriting everything after `t` with noise must not change
  `f(...)[:t]`.

Both are validated against seven deliberately leaky functions (a one-step lead, a centred window,
full-sample z-scoring, normalising by the last row, …) which they must catch, and eight causal ones
(rolling/expanding/EWM statistics, lags, drawdown) which they must pass. Passing shows no leak was
found at the tested cuts; it is not a proof. The detectors are reused by every signal (Stage 4+) and
the backtester (Stage 5).

### 1.8 What the audit found on the real store

Yahoo, 2005-01-03 → 2026-09-18, **645 tickers, 3,049,623 daily rows** (fingerprint `07d3aa14cdacd9c9`).
Nothing below is auto-repaired: findings are recorded (`validation_issues.csv`, `ingest_issues.csv`)
and later stages must consume them.

| Finding | Tickers | Obs. | Reading |
|---|---|---|---|
| `STALE_PRICE` (warn) | 16 | 11,376 | dominated by a few tickers whose early "history" is an illiquid stub — CPWR 2,872 rows, SW 2,459, EP 2,113, FERG 1,856, AMCR 1,331 (see symbol reuse, §2.3) |
| `EXTREME_RETURN` (warn) | 59 | 262 | for 18 of the 59 the first flagged move falls in Sep–Nov 2008 or Mar 2020: genuine crashes, flagged not removed |
| `POSSIBLE_UNRECORDED_SPLIT` (warn) | 67 | 222 | a heuristic: a −50 % day looks like a 2:1 split. **Not validated** as data errors and not used to exclude anything |
| `LARGE_DIVIDEND` (warn) | 30 | 40 | "dividends" > 10 % of the prior close: spin-offs, special dividends, cash mergers |
| `MISSING_SESSIONS` | 1 error, 2 warn | 25 + 2 | LEG: 25 of 30 sessions absent 2026-07-20 → 08-21 (5 rows in total); CPWR and FISV one session each |
| `OHLC_INCONSISTENT` (warn) | 2 | 2 | HUBB and UA, 2021-05-05 |
| `ZERO_VOLUME` (info) | 45 | 12,274 | mostly the same stub histories |
| `ADJ_CLOSE_MISMATCH` (ingest, warn) | 29 | 34 | our total return vs Yahoo's `Adj Close` — see below |
| `INVALID_CLOSE` (ingest, warn) | 1 | 1 | LEG, 2026-08-10, row dropped |

**Non-ordinary distributions are a real hazard for spread and return series.** Two examples from the
store (close-to-close move / dividend as share of prior close / our total return / Yahoo's `Adj Close`
return): KDP 2018-07-10 (special dividend) −82.1 % / 83.9 % / **+1.8 %** / +11.5 %; DHR 2016-07-05
(spin-off) +3.6 % / 35.7 % / **+39.4 %** / +61.2 %. For KDP our figure is the economically sensible
buy-and-hold return and Yahoo's differs because its adjustment formula diverges at extreme yields. For
DHR the close series already looks adjusted *and* a 35.7 % "dividend" is booked on top, so both
total-return figures are implausible: an apparent vendor double-count. Ordinary large moves (OXY
2020-03-09, −53 % with a 2.9 % yield) differ from Yahoo by only ~1.5 pp for the same second-order
reason. The pipeline therefore reports `LARGE_DIVIDEND`; pair screening (Stage 3) will mask windows
around such events rather than trust either vendor's treatment.

**Reproducibility of the download itself.** Two complete downloads of all 645 tickers about ten minutes
apart produced identical content hashes for **645/645** tickers and the same dataset fingerprint, with
zero vendor restatements. That says the pipeline is deterministic for a fixed vendor state; it says
nothing about how Yahoo will look next month.

## 2. Universe construction and survivorship bias

`universe.mode` selects one of:

* `static` — an explicit ticker list. Carries a caveat: a list chosen with hindsight is biased by an
  unknown amount.
* `membership_file` — a user-supplied `ticker,start,end[,sector]` CSV (e.g. licensed point-in-time data).
* `sp500_wikipedia` — membership **reconstructed** by walking Wikipedia's effective-date change log
  backwards from the current constituents. Memberships are half-open intervals `[start, end)` using
  *effective* dates, so announced-but-not-yet-effective changes are never used.

### 2.1 Reliability of the reconstruction

S&P 500 snapshot 2026-09-20: 503 current constituents, 407 change-log rows, **858 tickers that were ever
members** (846 since 2005), 6 aliases applied, **24 reconstruction anomalies** (log entries inconsistent
with the state they should apply to — renames or gaps; reported, not repaired).

Two checks decide from when the membership can be trusted, and they disagree:

* **Size check.** The rebuilt index has 502–512 members at every yearly sample from 2006 to 2026
  (within 2.4 % of 500). Taken alone this would say "reliable from 2006".
* **Log-density check.** The size check cannot see a swap missing from the log entirely: an omitted
  swap drops one name and adds one, so the size is unchanged while the membership is wrong. Recorded
  changes per year are 11 (2007), 8 (2008), 13 (2009), 11 (2010) against a recent median of 21.5
  (2016–2025); a year below 0.6 × that median is sparse. The log is dense only **from 2011**.

`reliable_from = 2011-01-01` (the later of the two). **No result in this project will use membership
before 2011-01-01.** Even after that date, incompleteness is not zero, and where the log is incomplete
the reconstruction is itself survivor-biased: each omitted swap replaces a former member by a current
one, so the "still members today" figures below are, if anything, too *high*.

### 2.2 Survivorship, quantified

| Date | Index size (rebuilt) | Still members today | Naive-universe overlap | Members with verified prices | Price coverage |
|---|---|---|---|---|---|
| 2011-01-01 | 506 | 297 | 58.7% | 341 | 67.4% |
| 2013-01-01 | 504 | 315 | 62.5% | 366 | 72.6% |
| 2015-01-01 | 503 | 328 | 65.2% | 382 | 75.9% |
| 2017-01-01 | 507 | 360 | 71.0% | 416 | 82.1% |
| 2019-01-01 | 506 | 388 | 76.7% | 440 | 87.0% |
| 2021-01-01 | 506 | 416 | 82.2% | 460 | 90.9% |
| 2023-01-01 | 504 | 442 | 87.7% | 480 | 95.2% |
| 2025-01-01 | 504 | 473 | 93.8% | 491 | 97.4% |
| 2026-09-18 | 503 | 503 | 100.0% | 502 | 99.8% |

* **Naive-universe overlap** is the share of each date's true members still in the index today — what a
  "use today's constituents" backtest would retain. On 2011-01-01 it drops ~41 % of the true universe.
* **Price coverage** is the share of each date's members for which we hold verified prices. Free data
  has *no rows at all* for 244 membership intervals (by inspection of the ticker list, overwhelmingly
  names that were delisted or acquired; not verified name by name), so on 2011-01-01
  one third of the true universe is missing, falling to 3 % by 2025. Averaged over 2011–2026, coverage
  is 85.5 % (minimum 67.4 %).
* Over 2011–2026 the store contains no series ending before 2026-08-04: names that delisted are simply
  absent, not truncated. The names we lose are the ones that left the index by acquisition, failure or
  demotion — precisely where a spread relationship breaks. The expected direction of the bias is
  therefore **optimistic** for any strategy formed among surviving names; its magnitude is unobservable
  with free data. **The resulting backtests are not free of survivorship bias and must never be
  described as such.**

### 2.3 Symbol reuse: prices that belong to a different company

Vendors key history by the *current* holder of a ticker symbol. CPWR (Compuware, an S&P 500 member until
2014), EP (El Paso) and SII (Smith International) now return unrelated illiquid instruments whose stale,
mostly zero-volume history sits under a former index member's name. `universe/identity.py` tests each
membership interval: over the bars inside the interval, at least 95 % must have positive volume. Of
868 intervals: **620 ok, 244 `no_prices`, 4 `suspect`** (CPWR, EP, SII, and SW). SW is a reconstruction
error rather than a price error: a current member with no "added" record, so the reconstruction treats
it as a member since before 2005 while its old history is an unrelated stub. Suspect and no-price
intervals are excluded from the coverage figures above and from `members_mask(verified=True)`. The test
is a heuristic with false negatives (a reused symbol that happens to trade actively passes); it bounds
the contamination, it does not remove it.


## 3. Reproducibility

* `python -m statarb.data snapshot` writes `var/store/reports/dataset_manifest.json`: a fingerprint of
  the stored **values** (per-ticker content hashes, not file bytes), the full config, universe
  provenance and caveats, library versions and the git commit.
* Vendor data cannot be frozen: re-downloading later may differ (restatements, back-filled dividends).
  Reproducibility therefore means *archive the store and quote its fingerprint*, not *re-run the
  download*. The fingerprint is deterministic for a fixed store (tested).
* Every stochastic component takes an explicit seed (later stages record it in the experiment registry).

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m statarb.data universe --fetch        # snapshot + build the universe
.venv/bin/python -m statarb.data refresh                 # incremental; safe to re-run
.venv/bin/python -m statarb.data audit                   # validation report
.venv/bin/python -m statarb.data survivorship            # coverage / naive-overlap / size check
.venv/bin/python -m statarb.data snapshot                # fingerprint + provenance
.venv/bin/python -m pytest
```

## 4. Statistical toolkit (Stage 2)

`statarb/selection/{adf,cointegration,johansen,hurst}.py`, `statarb/models/{ou,rolling}.py`. Everything is
implemented from first principles and validated against a reference implementation or an analytically
known case; the MacKinnon and Johansen critical-value tables are taken from statsmodels **as data** and
are never re-derived. Each test function documents its null hypothesis, assumptions and interpretation in
its module docstring, and every call is *one hypothesis test* to be counted when selection is corrected
(Stage 8; `engle_granger_both` reports `n_tests = 2`).

### 4.1 What was validated, against what

| Component | Reference | Agreement |
|---|---|---|
| ADF (`n`, `c`, `ct`, `ctt` × AIC/BIC/fixed lags) | `statsmodels.adfuller` | statistic ≤ 1e-9, lag and nobs identical |
| Engle–Granger, 1 and 2 regressors | `statsmodels.coint` | statistic ≤ 1e-9, p-value and critical values identical |
| Johansen, `det_order` ∈ {−1, 0, 1}, `k_ar_diff` ∈ {1, 2, 3}, ranks 0–2 | `statsmodels.coint_johansen` | eigenvalues, trace/max-eig statistics, critical values, eigenvectors ≤ 1e-8 |
| Johansen rank decisions | `statsmodels` `select_coint_rank` | identical |
| Johansen, every `k_ar_diff` incl. 0 | independent brute-force reduced-rank regression | ≤ 1e-7 (see the `k = 0` note in the module) |
| OU / half-life | exact recovery on a noiseless path; Monte-Carlo consistency; AR(1) coverage | analytic |
| Rolling statistics | brute-force windows; the Stage-1 look-ahead detectors | exact / pass |
| Hurst | fractional Brownian motion with known H | ±0.06 |

Beyond agreement, the tests check *statistical behaviour*: size and power by Monte Carlo (ADF: 5 % nominal,
600 draws; Engle–Granger 500 draws; Johansen 400 draws), rank recovery on simulated VECMs, and failure
modes (a level shift read as a unit root). **301 tests in total.** A mutation sweep of the new code injected
20 deliberate bugs (wrong degrees of freedom, `y_t` instead of `y_{t-1}`, plain instead of cointegration
critical values, a centred rolling window, …); 19 are caught. It exposed one test gap (a zero-variance
z-score) which is now covered; the one survivor is an equivalent mutant (exact information-criterion
ties do not occur with continuous data).

### 4.2 Findings that change how the tools may be used

1. **Engle–Granger needs its own critical values.** On 400 pairs of independent random walks (T = 250)
   the ordinary ADF table applied to the residual rejects **55.2 %** of the time; the cointegration
   tables reject **5.5 %**. OLS chooses β to make the residual look stationary, so the naive test is the
   spurious-regression trap. (`test_ordinary_adf_critical_values_on_the_residual_over_reject_badly`)
2. **OLS half-lives are biased fast.** With a true half-life of 23 days and n = 100, the median OLS
   κ is **0.068 (a ~10-day half-life, 2.3× too fast)**; Kendall's correction gives 0.041. Half-life
   intervals are asymmetric and can be unbounded (a `b` interval reaching 1). Spread selection on a
   short-sample half-life therefore favours spuriously fast reversion.
3. **`det_order` is an assumption about drift.** `0` (constant) uses critical values that assume the
   data drift. On driftless rank-2 data it recovers the right rank 69 % of the time against 95 % for
   `det_order=-1`. Log prices drift; a demeaned spread does not. The rank must be reported with its
   `det_order`.
4. **Johansen's rank recovery is imperfect even when the model is right:** on simulated VECMs (T = 600)
   the true rank is recovered 96–99 % (rank 0), 97–98 % (rank 1) and 89 % (rank 2) of the time, and
   the test over-rejects a true rank 0 at 8.1 % (nominal 5 %; three series, T = 250, `k_ar_diff = 0`,
   1000 draws).
5. statsmodels' `coint_johansen` pairs `Δy_t` with `y_t` (not `y_{t-1}`) when `k_ar_diff = 0`. This
   implementation follows Johansen's VAR(1) form and differs from it there (and only there); neither
   was shown to have the better size.

### 4.3 Real-data calibration of the Engle–Granger p-value

`experiments/stage2_null_size.py` (seed 20260920, data fingerprint `07d3aa14cdacd9c9`, results in
`experiments/results/stage2_null_size.json`). Null with **real marginals**: each stock's log-price path is
paired with a *different* stock's returns, circularly shifted by a random offset and re-accumulated. Fat tails
and volatility clustering are preserved; shared shocks are destroyed; the pair is independent by
construction, so P(p < 0.05) should be 5 %. Windows use only tickers with full data (survivors), which is
appropriate for a size check but says nothing about the index as it was. This is a size diagnostic: no pair
is ranked or reported.

| Window | Days | Pairs | Null @5 % | Null @1 % | Null @5 %, `ct` tables | Best-of-2 dirs @5 % | Ctrl: both Gaussian @5 % | Ctrl: real y, Gaussian x @5 % | Unshifted real pairs @5 % |
|---|---|---|---|---|---|---|---|---|---|
| 2011–2015 | 252 | 768 | 9.1% | 1.6% | 8.6% | 14.2% | 5.5% | 9.4% | 9.9% |
| 2011–2015 | 1250 | 768 | 7.3% | 2.0% | 11.7% | 10.5% | 4.8% | 6.5% | 9.0% |
| 2016–2020 | 252 | 843 | 2.1% | 0.4% | 1.9% | 5.1% | 5.0% | 4.9% | 4.6% |
| 2016–2020 | 1250 | 843 | 14.5% | 4.3% | 13.9% | 15.8% | 4.2% | 10.0% | 9.0% |
| 2021–2025 | 252 | 897 | 7.9% | 1.9% | 7.8% | 12.3% | 5.8% | 8.2% | 8.1% |
| 2021–2025 | 1250 | 897 | 10.6% | 3.0% | 8.9% | 13.9% | 4.0% | 10.8% | 10.1% |

* **The nominal 5 % is not the false-positive rate on real prices.** For five-year windows the null
  rate is 7.3–14.5 % (all six cells above 5 %; at a nominal 1 % it is 1.7–4.3 %, i.e. 2–4× nominal);
  for one-year windows it ranges from 1.9 % to 9.1 %. That spread across windows is itself the message:
  the miscalibration is unstable.
* **Testing both regression directions and keeping the better one raises it further** (3.8–21.5 %), which
  is why `engle_granger_both` counts two tests.
* **The Gaussian control sits at 4.0–5.8 %**, so the harness and the implementation are sound; the
  distortion comes from the data. With only *one* real leg (real y, Gaussian x) the five-year rate is
  already 6.5–10.8 %, so the real-return dynamics of a single stock are enough to distort the test.
* **A hypothesis I held was refuted.** I expected drift in log prices to be the cause (a drifting
  regressor detrends the residual, so the constant-only tables over-reject) and the `ct` tables to fix
  it. They do not (5-year: 11.7 %, 13.9 %, 8.9 %). The cause is therefore unexplained here; the leading
  candidates are volatility clustering, fat tails and return autocorrelation, none isolated yet.
* Unshifted real pairs reject at 4.6–10.1 % (`c` tables): the same order as the *null*, so an "apparently
  significant" rate of about 5–10 % among random stocks is, to a first approximation, what miscalibration
  alone produces; it is not evidence of hundreds of cointegrated pairs.
* The Wilson intervals stored in the JSON assume independent pairs. They are not (every y carries the
  market factor; stocks recur across pairings), so they are too narrow; treat the between-window range as
  the uncertainty. Reported rates also move by 1–2 pp with the random pairing (they did when the controls
  shared the random stream); the final run keeps the controls on a separate stream.

**Consequence for later stages.** Nominal p-values are a ranking, not error rates. Stage 3 will calibrate
the screen against an empirical null built with this same circular-shift construction, and Stage 8's
multiple-testing corrections (which assume valid p-values) will be applied to calibrated ones.

## 5. Pair discovery (Stage 3)

`statarb/selection/{correlation,clustering,screening}.py`, experiments `stage3_pair_discovery.py` and
`stage3_null_diagnostics.py`. Every step of a screen sees the training window only.

### 5.1 The funnel

1. **Universe**: point-in-time S&P 500 members on `train_end`, identity-verified (§2.3).
2. **Eligibility**, each exclusion recorded with its reason: price history present and verified; full
   coverage of the window (no imputation); median dollar volume ≥ $5 M; < 1 % zero-volume days; no
   non-ordinary distribution (> 10 % of price, §1.8) *inside the window*; a sector label.
3. **Candidates**: each stock's `k = 5` best-correlated peers inside its sector (or inside a
   hierarchical cluster). *Rank-based, never a correlation threshold*: the null below could never meet
   a threshold, so the calibration would be empty.
4. **Test**: Engle–Granger in both directions on dividend-adjusted log prices; statistic
   `T = min(stat_YX, stat_XY)`, which makes the direction choice part of the statistic instead of a
   hidden second test.
5. **Diagnostics** for every candidate: bias-corrected OU half-life and interval, spread volatility,
   Hurst (exploratory only), hedge ratio in each half of the window, ADF p-value in each half.
6. **Empirical null.** Stage 2 showed nominal p-values are not error rates on real prices, so the whole
   pipeline (correlation neighbours → both-direction test) is re-run on 10 panels in which every
   stock's returns are circularly shifted by its own random offset: fat tails, volatility clustering and
   autocorrelation are kept, all cross-stock dependence is destroyed, and the same *selection* is
   applied. A candidate's calibrated p-value is the share of null candidates at least as extreme.
   `estimated_fdr(α) = α · n_candidates / n_discoveries(α)` is the permutation estimate of the false-
   discovery proportion.
7. **Selection rule, fixed a priori and not tuned**: calibrated p ≤ 0.05, bias-corrected half-life
   between 5 and 60 trading days (faster is untradeable, slower ties up capital), hedge ratio > 0, at
   most 20 pairs, ranked by an economic proxy `κ · σ_stationary` (bps of log price per day). The counts
   Stage 8's corrections need (`n_family` = all same-sector pairs, `n_candidates`, `n_tests`) are
   recorded.

### 5.2 Validation

Tests (331 in total). The shifted panels preserve each stock's return distribution exactly and destroy
cross-sectional correlation (mean |ρ| 0.22 → below 0.06, asserted). On a **factor-null panel** (a market
factor plus idiosyncratic random walks: no cointegration exists, yet candidates are chosen for high
correlation and the better direction is kept) the measured rejection at a nominal 5 % is **5.2 %** with
the calibrated p-value against **10.0 %** with the naive best-of-two MacKinnon p-value (6 panels, 658
candidates). Planted cointegrated pairs are recovered (the tests assert a hedge ratio within 5 % and a
half-life within 20 % of the truth, averaged over seeds), and a planted pair that reverts in under a day
is significant but correctly rejected by the half-life rule. **Three look-ahead tests** run the screen on
a store where every price after `train_end` is scrambled, on a store that ends at `train_end`, and on a
store with a later stock split: the results are identical (the split case to 1e-9). A mutation sweep of
the new code injected 21 bugs (an unshifted null, a missing `+1` in the empirical p-value, panels read
past `train_end`, membership taken at the wrong date, …) and **all 21 are caught**; three initial
survivors led to stronger tests, and the tests found a real pandas-3 bug (a read-only array in the
clustering code).

### 5.3 Result: one pre-specified window, 2011-01-03 → 2015-12-31

Written down before running: the window, the default configuration, the seed (`20260920`) and three
hypotheses; nothing was changed afterwards, and no second window was run. The data fingerprint is
`07d3aa14cdacd9c9`.

**Funnel.** 506 index members on 2015-12-31 → 190 excluded
(37.5%) → **316 eligible**. Exclusions: 86 members
have no Yahoo price history, 57 have no GICS sector (they are not in today's constituent list), 20 have
no bars in the window, 20 fail coverage, 6 have a non-ordinary distribution and 1 fails the identity
check. **At least 143 of 506 true members (28 %) are lost to survivorship**
(no history or no sector: by inspection overwhelmingly companies that later left the index, not
verified name by name) before any statistical step.
Of 5,405 same-sector pairs, 1,259 were tested (2,518 regressions).

**H1 — does the screen find more than the null expects? No.**

| Level α | Naive (best-of-two MacKinnon p < α) | Calibrated discoveries | Expected under the null (α · 1,259) | Estimated FDR |
|---|---|---|---|---|
| 0.05 | 144 (11.4%) | 43 (3.4%) | 63.0 | 1.00 |
| 0.01 | – | 5 | 12.6 | 1.00 |
| 0.005 | – | 2 | 6.3 | 1.00 |
| 0.001 | – | 1 | 1.3 | 1.00 |

At every level the number of calibrated discoveries is *below* what pure chance produces under the same
selection: the estimated false-discovery proportion is capped at 1, i.e. **the screen cannot distinguish
its discoveries from noise**. The naive best-of-two p-value would have reported 144
"cointegrated pairs". This is the multiple-testing and calibration problem, seen directly.

*Post-hoc check (added after seeing this, explanatory only).* Real candidates are far more correlated than
null ones (median ρ 0.60 vs 0.04), so I
checked whether the null is simply the wrong benchmark. It is not: the two distributions of `T` agree
through the bulk and differ only in a slightly *thinner* lower tail for real pairs, and the calibrated rate
shows no trend with correlation (terciles 3.3% / 2.6% / 4.3%).

| Quantile of `T` | Real candidates | Shifted-null candidates |
|---|---|---|
| 1% | -4.23 | -4.45 |
| 5% | -3.65 | -3.82 |
| 25% | -2.97 | -2.99 |
| 50% | -2.49 | -2.47 |
| 75% | -1.99 | -1.98 |
| 95% | -1.31 | -1.35 |

**H2 and H3 — do the selected pairs stay mean-reverting out of sample?** The next 252 sessions (2016) are
scored with the *training* hedge ratio and mean (no re-fitting; this is a descriptive look, not a
backtest; Stage 5's walk-forward supersedes it):

| Group | n | OOS ADF rejects at 5 % | median OOS sd / training sd | share with ratio < 1.5 | median OOS half-life (days) |
|---|---|---|---|---|---|
| **Selected** | 20 | **0%** (95 % CI 0–16%) | 1.28 | 80% | 283 |
| Significant, not selected | 23 | 13% | 1.05 | 78% | 78 |
| Baseline: non-significant candidates | 1,216 | 10.5% | 0.71 | 96% | 46 |

* **H2 is refuted.** None of the 20 selected spreads rejects a unit root out of sample, against 10.5 % of
  the baseline. With n = 20 the interval (0–16 %) cannot exclude the baseline rate, so this is *no evidence
  of persistence*, not proof of its absence. The baseline's own 10.5 % (at a nominal 5 %) is Stage 2's
  miscalibration again.
* Their in-sample half-lives (median 29 days, by construction 5–60) do not persist: the median out-of-
  sample half-life is 283 days over the 14 pairs that revert at all, and **6 of 20 show no mean reversion
  whatsoever** (bias-corrected AR(1) coefficient ≥ 1).
* **H3 holds only loosely**: median out-of-sample spread volatility is 1.28× the training figure (80 %
  below 1.5×), but the median mean-shift is 1.35 training standard deviations, 10 of 20 hedge ratios
  moved by > 20 % between the two halves of the *training* window, and 4 spreads widened by more than 1.5×.

Representative failures (the full list is `experiments/results/stage3_selected_pairs_2011_2015.csv`;
they are shown because they are the typical outcome, not the exception):

| Rank | Pair (Y/X) | β | Calibrated p | Train half-life | OOS half-life | OOS sd ratio | OOS mean shift (sd) |
|---|---|---|---|---|---|---|---|
| 1 | CRM/ADBE | 0.73 | 0.015 | 26 | 95 | 0.90 | 0.54 |
| 4 | AMAT/MU | 0.45 | 0.015 | 30 | 1,239 | 2.03 | 7.59 |
| 10 | MCHP/TEL | 0.59 | 0.005 | 23 | none | 2.50 | 4.02 |
| 11 | DHR/HSIC | 0.80 | 0.001 | 14 | none | 7.45 | 5.38 |
| 14 | EL/KMB | 0.77 | 0.020 | 32 | 31 | 0.55 | 0.01 |
| 15 | PH/PCAR | 0.87 | 0.002 | 24 | 212 | 0.96 | 1.21 |

AMAT/MU's spread is twice as wide with a 7.6σ mean shift; DHR/HSIC's is 7.4× wider, which coincides with
DHR's July 2016 spin-off, the very `LARGE_DIVIDEND` event of §1.8. It sat *after* the training window, so
no training-time filter could have removed it: a live strategy needs stop-losses and event masks (Stage 4).

### 5.4 What this does and does not show

* One window, one configuration, 20 selected pairs: low power. It is **no evidence for** persistent
  cointegration in this sample; it is not proof that no cointegrated pair exists.
* The universe is survivor-tilted (28 % of true members lost up front); the direction of the bias for
  cointegration is not obvious, but it is not zero.
* Sectors are today's GICS labels (look-ahead in classification).
* "Estimated FDR = 1" means the discoveries cannot be told apart from the null, not that all are false.
  Stage 8 applies the formal corrections to these same counts.
* Half-life bounds, `k`, `α` and the 20-pair cap are researcher degrees of freedom; they were fixed before
  the run and any change now counts as a new trial (Stage 9's sensitivity analysis is registered: §11).

## 6. Pair strategy and hedge ratios (Stage 4)

`statarb/models/{hedge_ratio,kalman}.py`, `statarb/signals/{zscore,pairs}.py`,
`statarb/portfolio/construction.py`, `statarb/backtest/pair_pnl.py`, `statarb/statistics/bootstrap.py`;
experiments `stage4_hedge_ratio_synthetic.py` and `stage4_real_pairs.py`.

### 6.1 Design

* **Hedge ratio** (`ly = α + β lx`, dividend-adjusted log prices), four causal estimators: `static` (OLS
  on the training window only, then frozen), `expanding`, `rolling` (trailing window) and a random-walk
  `Kalman` filter (a stretch goal, compared rather than assumed better). At index `t` each uses data
  `≤ t` only.
* **Z-score in closed form.** With hedge ratio `b` the windowed spread has mean `m_y − b m_x` and
  variance `v_y − 2b c_xy + b² v_x`, so `z_t` follows from trailing moments and the intercept cancels.
  The *current* `b` is applied to the whole window: a spread stitched from each day's own `β_s` jumps by
  `Δβ · lx` (with `lx ≈ 4`) whenever `β` updates; a test asserts that a 0.05 update moves the stitched
  spread's z-score by more than 4 units and this construction's by less than 1.5. Every hedge method shares this
  one construction, so comparisons isolate the hedge ratio.
* **State machine.** Enter when `entry < |z| < stop` (never beyond the stop: that is a break, not a
  bargain); exit on reversion (`u ≤ exit`), stop-loss, `max_hold` bars, or missing data; after a stop or
  time exit the same direction is blocked until `|z| ≤ exit` again, so a stretched spread cannot churn;
  no same-bar reversal; nothing before `train_end`. Defaults 2.0 / 0.5 / 4.0 / 60, fixed a priori.
* **Sizing and P&L.** Legs at constant dollar weights `(+N, −βN)`; optional volatility targeting frozen
  at entry. A signal at the close of `t` earns the return of bar `t + 1` (one-bar lag). **Turnover is
  drift-aware**: constant dollar weights must be traded back to target as prices move, so `trade_t`
  counts entries, exits, hedge changes *and* that rebalancing; Stage 6 will charge costs on it.
  Everything here is **gross and frictionless**.

### 6.2 Validation

Tests (392 in total). The Kalman filter matches statsmodels' state-space filter to 1e-12 (states,
innovations, innovation variances) and, as the state noise → 0, reproduces expanding OLS (an analytic
check); its standardised innovations are calibrated (mean ≈ 0, sd ≈ 1) when the model is right. The
z-score equals a from-scratch window computation for static and time-varying `β`. The state machine is
checked on hand-worked sequences and by a property test of its invariants (positions in {−1, 0, 1},
entries only inside `(entry, stop)`, no trade longer than `max_hold`, no re-entry after a stop/time exit
before the spread neutralises). P&L, the one-bar lag and turnover are checked by hand. **Look-ahead:** the
whole chain (prices → hedge ratio → z-score → positions → P&L) is run through the Stage-1 detectors for
five hedge methods (scramble or drop everything after `t`; nothing earlier moves). A mutation sweep
injected 25 bugs (a same-bar return, a centred window, a hedge ratio or Kalman prior fitted on all data,
a stop-loss that never fires, wrong leg sign, a volatility estimate one bar ahead, …) and **all 25 are
caught**.

### 6.3 Which hedge-ratio estimator? A simulator with a known answer

On real prices the true hedge ratio is unobserved, so `experiments/stage4_hedge_ratio_synthetic.py` uses
one where it is known (spread OU with half-life 8.7 days; static fitted on 1,250 training days, 500 test
days; 200 simulations per regime; fixed before running). **R1** constant β = 0.8; **R2** β drifting as a
random walk (step sd 0.002); **R3** β jumping 0.8 → 1.05 mid-test. Metrics are gross and unit-sized;
Sharpe entries show the *paired* difference from static (± its standard error over the shared
simulations).

RMSE of the estimated β over the test period:

| Method | R1 constant | R2 drifting | R3 break | R3b mild break (post-hoc) |
|---|---|---|---|---|
| static | 0.017 | 0.265 | 0.177 | 0.040 |
| expanding | 0.015 | 0.273 | 0.271 | 0.053 |
| rolling 60 | 0.267 | 0.457 | 2.116 | 0.489 |
| rolling 120 | 0.173 | 0.408 | 2.089 | 0.451 |
| rolling 250 | 0.093 | 0.351 | 1.997 | 0.413 |
| Kalman δ=1e-6 | 0.035 | 0.087 | 0.318 | 0.075 |
| Kalman δ=1e-5 | 0.047 | 0.079 | 0.286 | 0.075 |
| Kalman δ=1e-4 | 0.050 | 0.072 | 0.238 | 0.068 |

Gross annualised Sharpe (paired difference from static ± s.e.):

| Method | R1 constant | R2 drifting | R3 break | R3b mild break (post-hoc) |
|---|---|---|---|---|
| static | 1.92 | 0.76 | 1.15 | 1.31 |
| expanding | 1.90 (-0.01 ± 0.01) | 0.76 (+0.00 ± 0.02) | 0.99 (-0.16 ± 0.03) | 1.30 (-0.01 ± 0.02) |
| rolling 60 | 1.64 (-0.28 ± 0.03) | 0.80 (+0.05 ± 0.05) | 0.57 (-0.58 ± 0.07) | 1.10 (-0.21 ± 0.05) |
| rolling 120 | 1.74 (-0.18 ± 0.03) | 0.73 (-0.02 ± 0.05) | 0.48 (-0.68 ± 0.06) | 1.10 (-0.21 ± 0.04) |
| rolling 250 | 1.85 (-0.06 ± 0.03) | 0.76 (+0.00 ± 0.04) | 0.31 (-0.84 ± 0.06) | 0.94 (-0.37 ± 0.04) |
| Kalman δ=1e-6 | 1.88 (-0.03 ± 0.02) | 0.87 (+0.12 ± 0.04) | 1.02 (-0.13 ± 0.04) | 1.29 (-0.02 ± 0.03) |
| Kalman δ=1e-5 | 1.89 (-0.03 ± 0.02) | 0.88 (+0.12 ± 0.04) | 1.06 (-0.09 ± 0.04) | 1.34 (+0.03 ± 0.03) |
| Kalman δ=1e-4 | 1.91 (-0.01 ± 0.02) | 0.89 (+0.13 ± 0.04) | 1.12 (-0.03 ± 0.04) | 1.32 (+0.01 ± 0.03) |

Turnover per year (units of pair capital):

| Method | R1 | R2 | R3 | R3b |
|---|---|---|---|---|
| static | 21.8 | 20.4 | 22.2 | 21.3 |
| expanding | 21.8 | 20.5 | 23.0 | 21.4 |
| rolling 60 | 27.1 | 28.0 | 37.1 | 27.6 |
| rolling 120 | 23.7 | 23.7 | 35.0 | 23.9 |
| rolling 250 | 22.2 | 21.5 | 33.4 | 22.2 |
| Kalman δ=1e-6 | 21.8 | 20.8 | 23.5 | 21.7 |
| Kalman δ=1e-5 | 21.8 | 20.7 | 23.5 | 21.9 |
| Kalman δ=1e-4 | 21.5 | 20.7 | 23.2 | 21.4 |

Against the hypotheses written down beforehand:

* **HS1 (constant β: static/expanding best): supported.** They estimate β best (0.017 / 0.015) and short
  rolling windows lose 0.28 Sharpe (rolling 60) to estimation noise and 24 % more turnover.
* **HS2 (drift: rolling and Kalman follow it): only the Kalman filter does.** It cuts the β error by
  about 3.5× (0.07–0.09 vs 0.27) and gains **+0.12 to +0.13 Sharpe** (≈ 3 s.e.). Rolling OLS is *worse than
  static* at every window (0.35–0.46 vs 0.27): 60–250 days of highly autocorrelated spread residuals are
  too little to pin β down, so the window's noise exceeds the drift it tries to follow.
* **HS3 (after a break rolling/Kalman recover, static does not): refuted as stated.** The pre-specified
  break moves the spread by ≈ 1.0, about 33 stationary standard deviations, which is far larger than a
  realistic break and wrecks any OLS window that contains it (β error ≈ 2, Sharpe −0.6 to −0.8 vs static).
  A **post-hoc** milder break (R3b, ≈ 7 sd, labelled as added after seeing R3) gives the sober version:
  static and Kalman are statistically indistinguishable (paired Δ Sharpe −0.02 to +0.03 ± 0.03), and every
  rolling window is significantly worse (−0.21 to −0.37 ± 0.04).
* **HS4 (no method wins everywhere): partly refuted.** Kalman with δ = 1e-4 ranks 2nd, 1st, 2nd; static
  ranks 1st, 7th, 1st. Kalman δ = 1e-4 is the most robust in this world, but the best δ sits at the edge
  of the pre-specified grid, and the simulator is *exactly the state-space model the filter assumes*: a
  favourable test, not evidence about real prices.
* **HS5 (β accuracy does not imply Sharpe): supported.** In R2 rolling 60 has a worse β error than static
  (0.457 vs 0.265) yet a slightly higher Sharpe (+0.05 ± 0.05); accuracy and profit decouple, and
  turnover is what separates the methods (Kalman ≈ static; rolling 60 +24 % in R1, +67 % in R3).

### 6.4 The frozen Stage 3 pairs in 2016 (a descriptive look, gross)

`experiments/stage4_real_pairs.py`: the 19 selected pairs (DHR/HSIC dropped, see below) and, as a
baseline, the 1,192 non-significant candidates of the same screen, traded over the 252 sessions
of 2016 with the training hedge ratio, default rules, no costs. Variants were fixed in advance and all
are reported. Legs with a non-ordinary distribution in the window (DHR, GEN, JCI, VMRK; §1.8) are
dropped: vendor total returns around a spin-off are not trustworthy.

| Variant | Selected (19 pairs) Sharpe [95 % CI] | Baseline (1,192 pairs) Sharpe [95 % CI] | Random 19-pair subsets of the baseline: median (5–95 %) | Selected's percentile | Turnover / pair | Baseline median β drift |
|---|---|---|---|---|---|---|
| static, z 30 | +1.16 [-0.73, +3.02] | +1.67 [+0.26, +3.37] | +0.81 (-0.50, +2.10) | 69% | 30.6 | 0.0% |
| static, z 60 | +1.19 [-0.66, +2.94] | +1.67 [+0.20, +3.34] | +0.83 (-0.50, +2.06) | 69% | 19.2 | 0.0% |
| static, z 120 | +0.51 [-1.65, +2.54] | +1.25 [-0.29, +2.76] | +0.61 (-0.74, +1.98) | 46% | 11.3 | 0.0% |
| expanding | +0.96 [-0.89, +2.68] | +1.77 [+0.27, +3.40] | +0.87 (-0.41, +2.13) | 54% | 18.6 | 4.6% |
| rolling 60 | +0.47 [-1.49, +2.25] | +0.50 [-1.23, +2.47] | +0.24 (-1.21, +1.72) | 60% | 24.6 | 66.3% |
| rolling 120 | +0.46 [-1.15, +2.04] | +0.51 [-1.05, +2.40] | +0.30 (-1.06, +1.66) | 58% | 20.6 | 54.5% |
| rolling 250 | -0.71 [-2.87, +1.17] | +0.51 [-1.02, +2.31] | +0.34 (-0.86, +1.55) | 7% | 17.1 | 46.5% |
| Kalman δ=1e-5 | -0.34 [-2.10, +1.28] | +1.64 [-0.01, +3.56] | +0.85 (-0.40, +2.10) | 6% | 16.2 | 7.2% |

* **Selection adds nothing detectable.** The naive comparison (selected 1.19 vs baseline 1.67, static) is
  unfair: the baseline holds 1,192 pairs and diversification alone raises its Sharpe. The size-matched
  comparison (added *after* seeing the first run, labelled post-hoc) puts the selected group at the
  46th–69th percentile of random 19-pair baseline subsets for the static and expanding variants (nowhere near
  the 95th), and at the 6th–7th percentile for Kalman and rolling 250.
* **The strategy earned a gross return with no selection at all.** Trading z-score reversion on
  sector-correlated pairs, chosen without any cointegration evidence, gives a gross Sharpe of
  +1.67 [+0.20, +3.34] over 1,192 pairs (mean 1.22 bp/day per unit
  notional, 3.1 % over the year, 18 units of turnover per pair). That is
  not evidence of alpha: it is one year, gross of costs, on a survivor-tilted universe with today's sector
  labels, and it is what Stage 6's costs and Stage 5's walk-forward exist to test. It does say the
  cointegration filter is not what drives a positive gross number.
* **Hedge-ratio behaviour matches the simulator.** The static and expanding hedges barely move (median
  drift 0 % / 4.6 %); rolling windows wander (47–66 % of β), lose Sharpe against static and add turnover.
  The Kalman filter, best in the simulator, is no better than static on the baseline (1.64 vs 1.67) and
  worse on the selected pairs (−0.34 vs +1.19): the simulator is not the world.
* **Uncertainty is large.** The selected group's intervals span roughly −0.7 to +2.9 (static): with 19
  pairs and one year nothing in this table separates from zero, from each other, or from random subsets.

### 6.5 Limits of Stage 4

* Frictionless and gross: turnover of ≈ 18 units per pair-year makes Stage 6's costs decisive.
* One year, one strategy configuration, one δ (1e-5) on real data; the z-window sensitivity is reported
  (static: 30 → 1.16, 60 → 1.19, 120 → 0.51) but not used to choose.
* The 2016 window was already used in Stage 3's out-of-sample look; no parameter has been tuned on it, but
  it is no longer fresh, and Stage 5's train/validation/holdout split must treat it as research data.
* The simulator favours the Kalman filter by construction; real hedge ratios are not random walks.
* Stops fired on 5–10 % of trades in the 60- and 120-day-window variants (0.7 % with a 30-day window);
  the DHR/HSIC spin-off blow-up shows the stop and an event mask matter.
* **Correction (found in Stage 7).** The state machine kept the re-entry block for one direction only, so a forced
  (stop / time) exit in the *opposite* direction overwrote an earlier block and let the first direction re-enter before
  `|z| ≤ exit` had been seen, contradicting the rule above. A property test found it on a five-bar example
  (`z = [3, 1, −3, −1, 3]`, `max_hold = 1`); blocks are now kept per direction, with a regression test that fails on
  the old code. **It changed no reported result**: the old and corrected engines were run side by side on every pair
  and configuration of Stages 4–6 (43,983 runs for the nine Stage 5 configurations over all selected and placebo-pool
  pairs of the four folds, plus 9,688 for Stage 4's eight 2016 variants) with **0 differing positions or exit reasons**,
  and Stage 4's regenerated results file is identical to the committed one. Reaching the bug needs the z-score to jump
  past both entry thresholds without a single bar inside the exit band.

## 7. Walk-forward backtesting and the sealed holdout (Stage 5)

`statarb/backtest/{walkforward,metrics}.py`, `statarb/data/holdout.py`, `statarb/research/registry.py`,
`configs/data_default.yaml` (the `split:` block); experiment `stage5_walkforward.py`.

### 7.1 The chronological split, and how it is enforced

Fixed **before any result after 2018 was seen**:

| Phase | Dates | Role |
|---|---|---|
| Research / training | 2011-01-03 → 2018-12-31 | development, configuration search (all trials here count) |
| Validation | 2019-01-01 → 2021-12-31 | pre-selected finalists are scored once; not used to tune |
| **Final holdout** | 2022-01-01 → end of data | sealed until the final evaluation; one look |

2011-01-03 is the first date with a reliable point-in-time universe (§2.1). The blocks are calendar years,
so a walk-forward test block never straddles two phases (the engine refuses a split that would).

**Enforced in code, not by intention.** With a split configured, every research-facing read
(`DataPipeline.panel`, `raw_close`, `identity_report`) refuses a request that reaches the holdout, whether
through its `end` *or* through `as_of`, which could otherwise smuggle later splits into a price basis
(`HoldoutViolation`). Unlocking is one call, **allowed once**, and is written to an append-only ledger
(`experiments/holdout_ledger.jsonl`, committed: it is empty as of this commit, i.e. the holdout has never
been opened). A second unlock raises unless it is explicitly recorded as a contaminating `reopen`. Data
*operations* (refresh, audit) read the whole store because data integrity is not a modelling choice, and
code that bypasses the pipeline and reads Parquet files directly is outside the guard: it prevents
accidents, not sabotage.

**Disclosures.**
* Stage 2's null-size diagnostic used a 2021–2025 window, which now overlaps the holdout. It paired
  *independent* shifted stocks to test a p-value's calibration; no strategy, parameter, feature or universe
  was selected from it, and its conclusion (five-year null rejection of 7.3 % and 14.5 %) is visible in the
  2011–2015 and 2016–2020 windows alone. It predates the guard and is logged as such in the registry.
* The 2016 look in Stages 3–4 is research-phase data.
* **A look-ahead defect was found and fixed while building this stage.** The identity check
  (§2.3) judged each membership interval on bars up to *today*, not up to the fold's `train_end`: later bars
  could launder early zero-volume rows. It is now point-in-time (`identity_report(as_of, since)`), with a
  regression test that the earlier tests could not have caught (they scrambled prices, not volumes). I
  re-ran Stage 3 with the fix: the funnel, the 43 discoveries, the 20 selected pairs and every
  out-of-sample number are **identical**; four tickers that would have passed only because of later bars are
  excluded earlier by other filters. The defect was in the code, not in a reported result.

### 7.2 The engine

A **fold** screens pairs on a training window and trades the following block::

    train [train_start, train_end] -> point-in-time universe, eligibility, pair screen, hedge ratios (frozen)
    test  [test_start,  test_end]  -> trade bar by bar; every position is forced flat at test_end

* **Blocks are independent**: a position opened in a block is closed at its end (the exit trade is
  counted), so there is nothing to purge or embargo: no label is built from future returns, and a fold's
  selection uses data at or before its `train_end`.
* **Portfolio**: fixed-slot equal weight. Each of 20 slots gets 1/20 of capital and a pair's unit-notional
  P&L is scaled by that, so the return is on total capital and does not depend on how many pairs passed the
  screen (unused slots earn nothing). In the run below the mean gross exposure is 0.81× capital (max 1.75×)
  with 8.5 pairs open on average.
* **Events**: an ex-date with a distribution above 10 % of price (a spin-off; §1.8) distorts total returns
  and the trailing z-window. Ex-dates are public in advance, so the spread is blacked out from one day before
  to `z_window` days after. This is the only place the engine reads a calendar fact one day ahead of a
  decision.
* **Costs**: a `cost_fn(frame)` hook charges the drift-aware turnover of Stage 4; the default is none, so
  every number in this stage is **gross**.
* **Registry**: `experiments/registry.jsonl` is an append-only log of every trial (date, stage, universe, phases
  used, windows, parameters, costs, results, seed, git commit, data fingerprint). Stages 2–4 were back-filled.
  **18 real-data strategy trials are registered so far** (Stage 3 screen: 1, Stage 4 variants: 8, Stage 5
  grid: 9); simulations and diagnostics are logged but not counted. Stage 8's corrections use this count.

### 7.3 Validation

Tests (431, all green). The look-ahead tests exercise the whole engine: scrambling every price after
`test_end`, scrambling every price from the first test day (the screen is unchanged; trading is not),
scrambling prices after a date *inside* the block (P&L up to that date is identical), and a store that
simply ends at `test_end` (identical) all leave the results unchanged. Other tests cover fold boundaries and
phase labels, the holdout guard on every read path, one-time unlocking, slot arithmetic, forced liquidation
with its exit turnover, blocks being independent, event blackouts, the cost hook, and the metrics against
hand-worked values (drawdowns, Sortino, profit factor). A mutation sweep injected 26 bugs (a screen trained
through the test block, a hedge ratio fitted through it, no liquidation, a guard that ignores `as_of` or is
off by one, a ledger that permits a second unlock, an identity check reading to today, …); **all 26 are
caught**. Three initial survivors exposed real test gaps (a placebo that could include significant pairs; a
drawdown that ignored the starting equity), now covered.

### 7.4 Result: the research phase, walk-forward, gross (2015–2018)

`experiments/stage5_walkforward.py`, pre-specified: four test years (2015–2018), each screened on the four
years before it; the 9-configuration grid (hedge ∈ {static, expanding, Kalman} × entry ∈ {1.5, 2.0, 2.5});
a **placebo** that trades random pairs (same count as the screen selected, drawn from that fold's
*non-significant* candidates with β > 0, 300 draws shared across configurations). Validation and the holdout
are not touched. The grid was pruned to these hedge methods using Stage 4, and all 9 are registered as trials.

The screen in each fold (a fresh window each time; the calibrated p-values use 10 shifted-null panels):

| Test year | Training window | Eligible | Candidates | Calibrated discoveries @5 % | Expected under the null | Estimated FDR |
|---|---|---|---|---|---|---|
| 2015 | 2011-01-03 → 2014-12-31 | 308 | 1,222 | 68 (5.6%) | 61 | 0.90 |
| 2016 | 2012-01-04 → 2015-12-31 | 324 | 1,275 | 47 (3.7%) | 64 | 1.00 |
| 2017 | 2013-01-03 → 2016-12-30 | 340 | 1,345 | 52 (3.9%) | 67 | 1.00 |
| 2018 | 2014-01-02 → 2017-12-29 | 361 | 1,401 | 54 (3.9%) | 70 | 1.00 |

Stage 3's pattern **replicates in four new windows**: the calibrated discoveries are about what the null
produces (3.7–5.6 % of candidates against 5 % by construction), and the estimated false-discovery
proportion is 0.90–1.00.

Gross performance of the stitched 2015–2018 out-of-sample series (1,006 days; Sharpe with a stationary
block-bootstrap 95 % interval; placebo = 300 random-pair portfolios):

| Configuration | Screened Sharpe [95 % CI] | Placebo Sharpe: median (5–95 %) | Screened's percentile in placebo | Ann. vol | Max drawdown | Turnover / yr | Trades |
|---|---|---|---|---|---|---|---|
| static, entry 1.5 | +0.23 [-0.67, +1.24] | +0.37 (-0.33, +1.20) | 37% | 4.7 % | -6.4 % | 27.7 | 520 |
| static, entry 2.0 | -0.16 [-1.10, +0.80] | +0.33 (-0.34, +1.12) | 11% | 4.2 % | -8.2 % | 19.3 | 359 |
| static, entry 2.5 | -0.32 [-1.29, +0.69] | +0.21 (-0.46, +0.99) | 11% | 3.6 % | -9.0 % | 12.7 | 233 |
| expanding, entry 1.5 | +0.30 [-0.59, +1.26] | +0.33 (-0.37, +1.12) | 49% | 4.6 % | -5.8 % | 27.9 | 527 |
| expanding, entry 2.0 | +0.01 [-0.93, +0.98] | +0.29 (-0.38, +1.05) | 23% | 4.1 % | -7.0 % | 20.0 | 374 |
| expanding, entry 2.5 | -0.30 [-1.29, +0.76] | +0.21 (-0.48, +0.99) | 12% | 3.6 % | -9.5 % | 12.8 | 236 |
| Kalman δ=1e-5, entry 1.5 | -0.25 [-1.02, +0.56] | +0.30 (-0.38, +1.06) | 10% | 4.4 % | -7.5 % | 23.6 | 514 |
| Kalman δ=1e-5, entry 2.0 | -0.11 [-0.97, +0.74] | +0.27 (-0.40, +1.04) | 17% | 4.0 % | -7.2 % | 17.2 | 367 |
| Kalman δ=1e-5, entry 2.5 | -0.18 [-1.09, +0.69] | +0.30 (-0.45, +1.09) | 13% | 3.4 % | -6.8 % | 11.4 | 240 |

* **H5a (the screened portfolio beats random pairs) is refuted.** The best configuration (expanding,
  entry 1.5, +0.30) sits at the 49th percentile of the placebo, i.e. at its median; in **all nine**
  configurations the screened portfolio is at or below the placebo median (10th–49th percentile), and none
  is anywhere near the 95th. Its mean daily return is below the placebo's by 0.2–0.9 bp in eight of nine.
* **H5b (the strategy itself is positive gross): weakly.** The placebo median is +0.21 to +0.37 and 69–80 %
  of the random portfolios have a positive Sharpe, but every placebo 5–95 % band spans zero. This is far
  below Stage 4's +1.67 for a 1,192-pair 2016 portfolio: that figure was inflated by diversification and one
  favourable year.
* **H5c (hedge ordering echoes the simulator): partly.** Static and expanding are close (e.g. entry 2.0:
  −0.16 vs +0.01); the Kalman filter is no better. Differences are far inside the intervals.
* **Uncertainty dominates.** Every screened Sharpe interval is roughly ±1, and the by-year Sharpe of one
  configuration swings from -1.51 to +0.68 (static, entry 2.0: 0: -0.50, 1: +0.68, 2: -1.51, 3: +0.54), so a single block says little.
* **Entry threshold and cost exposure.** Lower entry thresholds have the higher gross Sharpe *and* the higher
  turnover (27.7 vs 12.7 units/yr, entry 1.5 vs 2.5), so the ranking is exactly what costs will reshuffle;
  no configuration is chosen here (Stage 6 re-scores them with costs, and the finalists go to validation
  once).

**What this establishes.** On four fresh walk-forward blocks, pairs chosen by the calibrated cointegration
screen are not distinguishable from randomly chosen correlated same-sector pairs, and both earn close to
nothing gross. That is a negative result for Experiment A on the research phase; it is *not* evidence that
no exploitable relationship exists, and it says nothing yet about validation or the holdout.

### 7.5 Limits of Stage 5

* Four test blocks and 20-pair portfolios: low power (intervals of ±1 Sharpe); the placebo controls the
  selection effect but not the sampling noise.
* Gross of costs and financing; one screening configuration; the survivor-tilted universe and today's sector
  labels apply unchanged.
* Positions are closed at each block boundary, which a live system would not do; it removes cross-block
  leakage at the price of some realism (an extra exit and entry per surviving trade at each year end).
* Missing returns (delistings) are treated as zero and counted; none occurred in these blocks.
* The event blackout uses a one-day-ahead calendar fact; the results with it disabled were not run.

## 8. Transaction costs (Stage 6)

`statarb/backtest/costs.py`; experiment `stage6_costs.py`. Every number in this section is **net of costs
unless it says gross**, on the same trades as Stage 5.

### 8.1 The cost model, and its assumptions

For each leg, on the traded dollar notional `Q` of a bar (`Q` = traded amount × `capital / slots`):

| Component | Charge | Central assumption |
|---|---|---|
| Spread | `Q · half-spread` | tiered by trailing dollar ADV: 1.0 bp above $500 M, 1.5 bp $100–500 M, 2.5 bp $25–100 M, else 5 bp |
| Commission and fees | `Q · (commission + regulatory)` | 0.5 bp + 0.25 bp on all traded notional |
| Market impact | `Q · Y · σ · √(Q / ADV)` | `Y` = 0.5; ADV a trailing 20-day median, σ a trailing 20-day volatility |
| Short borrow | short notional held × rate / 252 per day | 50 bps a year |

* **Capital enters through impact.** Impact grows like `√Q` per dollar, so it is the one component that grows
  faster than trade size; the central case is $100 M over 20 slots ($5 M per pair). Capital is an explicit,
  swept assumption, not a hidden constant.
* **Spread.** Free data has no historical quotes. The central model is the documented tier above (a
  conservative-to-realistic reading of S&P 500 quoted half-spreads). A **Corwin–Schultz (2012)** estimate from
  daily high/low is implemented and reported as a sensitivity: on simulated bars it recovers a 100 bps spread
  (mean 101.5 bps) but is **biased upward and very noisy for tiny spreads** (a true 0 gives ≈ 5 bps on average
  and is negative on a large share of days), which is why it is not the central case.
* **Execution** is at the decision close (market-on-close), with no partial fills or rejections. Inputs are
  **causal**: ADV, σ and spread use data through `t − 1`; a trade decided at the close of `t` cannot know
  `t`'s own volume. Participation above 10 % of ADV is flagged, not cured.
* Not modelled: hard-to-borrow names, financing of the long leg, taxes, intraday timing, queue position.

### 8.2 Validation

Tests (453 in total): cost components and their sum against hand-worked values (impact = `Y σ √(Q/ADV)` for
each leg with its own ADV, σ and spread; borrow only on the *held* short leg, which is the y-leg for a short
spread), scaling laws (impact ×2 when capital ×4, ×3 when σ ×3, ×1/2 when ADV ×4; spread and commission
linear), monotonicity in every parameter, the participation flag, a loud error if a trade has no ADV, causal
inputs (the leakage detectors on the market data, and a planted enormous volume *today* that must not move
today's ADV), Corwin–Schultz against the formula by hand, the overnight adjustment (this test caught a sign
error in my first version), and recovery on simulated bars. A mutation sweep injected 20 bugs (impact linear
instead of square-root, ignoring σ, a same-day ADV, borrow on the wrong leg, capital not divided by slots, a
missing regulatory fee, …); **all 20 are caught** after two tests were strengthened (a median that swapping
one value on the same side cannot move; hand tests that all used σ = 0.02).

### 8.3 Result: the research phase, net of costs (2015–2018)

The fold screens were re-computed from the same seed and the gross series **reproduces Stage 5 to 2e-16**,
so gross and net are on identical trades. Central cost case, with the Stage 5 placebo re-scored net (same
300 draws):

| Configuration | Gross Sharpe | **Net Sharpe** [95 % CI] | Gross return / yr | Net return / yr | Cost / yr | Turnover / yr | Breakeven cost multiple | Net placebo median | Screened's percentile |
|---|---|---|---|---|---|---|---|---|---|
| static, entry 1.5 | +0.23 | **-0.64** [-1.51, +0.33] | +1.09 % | -2.99 % | 4.08 % | 27.7 | 0.27 | -0.55 | 41% |
| static, entry 2.0 | -0.16 | **-0.84** [-1.80, +0.11] | -0.66 % | -3.53 % | 2.87 % | 19.3 | 0 | -0.41 | 13% |
| static, entry 2.5 | -0.32 | **-0.85** [-1.79, +0.16] | -1.16 % | -3.05 % | 1.89 % | 12.7 | 0 | -0.37 | 13% |
| expanding, entry 1.5 | +0.30 | **-0.58** [-1.46, +0.35] | +1.41 % | -2.66 % | 4.07 % | 27.9 | 0.35 | -0.61 | 52% |
| expanding, entry 2.0 | +0.01 | **-0.71** [-1.65, +0.24] | +0.05 % | -2.92 % | 2.97 % | 20.0 | 0.02 | -0.45 | 26% |
| expanding, entry 2.5 | -0.30 | **-0.83** [-1.81, +0.20] | -1.06 % | -2.96 % | 1.89 % | 12.8 | 0 | -0.38 | 12% |
| Kalman δ=1e-5, entry 1.5 | -0.25 | **-1.01** [-1.79, -0.23] | -1.12 % | -4.43 % | 3.31 % | 23.6 | 0 | -0.58 | 13% |
| Kalman δ=1e-5, entry 2.0 | -0.11 | **-0.71** [-1.56, +0.11] | -0.43 % | -2.82 % | 2.39 % | 17.2 | 0 | -0.44 | 25% |
| Kalman δ=1e-5, entry 2.5 | -0.18 | **-0.65** [-1.55, +0.19] | -0.59 % | -2.18 % | 1.59 % | 11.4 | 0 | -0.27 | 16% |

* **H6a (costs remove the gross edge): confirmed.** Net Sharpe is below gross for all nine and negative for all
  nine (−0.58 to −1.01). The best gross configurations give up about 4 points a year: static, entry 1.5 earns
  1.09 % gross and pays 4.08 % in costs. The **breakeven cost multiple** (the factor on *all* costs at which
  net return reaches zero) is 0.27 and 0.35 for the two configurations with a clearly positive gross return
  and ~0 for the rest: costs would have to fall to about a third of the central case.
* **H6b (turnover ranks the drag): confirmed.** Cost per year is 4.07–4.08 % at entry 1.5, about 2.9 % at 2.0
  and 1.9 % at 2.5. An average traded unit costs 14–15 bps all-in.
* **Where the cost comes from.** 77% of the cost is market impact, 11% spread,
  5% commission and fees, 7% borrow (expanding, entry 1.5; the shares are within
  three points across the grid). Participation is mostly small (median about 0, 90th percentile 0.2–0.5 % of ADV;
  2.4% of trade-days above 5 %, 0.6% above 10 %), but the square root is concave, so small
  participations still cost several bps each.
* **Against the placebo, net.** Costs hit random pairs about as hard (net placebo median −0.27 to −0.61); the
  screened portfolio's percentile is 12–52 %. The one configuration above the placebo median (expanding, entry
  1.5, 52 %) has a net Sharpe of −0.58.

Sharpe as each cost component is added (central case):

| Configuration | Gross | + spread | + commission | + borrow | + impact |
|---|---|---|---|---|---|
| static, entry 1.5 | +0.23 | +0.13 | +0.09 | +0.03 | -0.64 |
| static, entry 2.0 | -0.16 | -0.23 | -0.27 | -0.32 | -0.84 |
| static, entry 2.5 | -0.32 | -0.38 | -0.41 | -0.45 | -0.85 |
| expanding, entry 1.5 | +0.30 | +0.21 | +0.16 | +0.10 | -0.58 |
| expanding, entry 2.0 | +0.01 | -0.07 | -0.10 | -0.16 | -0.71 |
| expanding, entry 2.5 | -0.30 | -0.35 | -0.38 | -0.42 | -0.83 |
| Kalman δ=1e-5, entry 1.5 | -0.25 | -0.34 | -0.38 | -0.43 | -1.01 |
| Kalman δ=1e-5, entry 2.0 | -0.11 | -0.18 | -0.21 | -0.26 | -0.71 |
| Kalman δ=1e-5, entry 2.5 | -0.18 | -0.23 | -0.26 | -0.30 | -0.65 |

Spread, commission and borrow together remove about 0.2 Sharpe; impact takes the two best configurations from
≈ 0 to about −0.6.

### 8.4 Sensitivity: what would have to be true

Net Sharpe of the two best-gross configurations across impact coefficient and capital (the other cost
components at their central values):

*static, entry 1.5*

| Capital | Y = 0 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| $10 M | +0.03 | -0.07 | -0.18 | -0.39 | -0.82 |
| $100 M | +0.03 | -0.30 | -0.64 | -1.31 | -2.66 |
| $1 B | +0.03 | -1.03 | -2.10 | -4.17 | -7.47 |

*expanding, entry 1.5*

| Capital | Y = 0 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| $10 M | +0.10 | -0.00 | -0.11 | -0.33 | -0.76 |
| $100 M | +0.10 | -0.24 | -0.58 | -1.27 | -2.64 |
| $1 B | +0.10 | -0.98 | -2.07 | -4.16 | -7.50 |

* **Impact is the swing factor, and it scales with size.** At Y = 0 the two configurations net +0.03 and +0.10
  (indistinguishable from zero: the intervals are about ±1). Impact costs ≈ 0.2 Sharpe at $10 M, ≈ 0.7 at
  $100 M and ≈ 2.1 at $1 B (Y = 0.5): at $1 B it swamps everything else (H6c: confirmed for $1 B; at $10 M
  impact is *comparable* to the other three components together, not minor as I had guessed).
* **Even a free spread does not rescue it.** With a zero half-spread (impact still at Y = 0.5) the best
  configuration nets −0.48. Across the whole impact × capital grid the highest net Sharpe anywhere is +0.10
  (expanding, entry 1.5, Y = 0); across the spread grid −0.48. Borrow of 0 / 50 / 150 bps moves the same
  configuration only from −0.52 to −0.70.
* No cost assumption in the swept range makes any configuration's net Sharpe distinguishable from zero on the
  research folds, and the reasonable ones make it clearly negative.

### 8.5 The pre-specified finalist rule, and its outcome

Written before any net result: a configuration advances to validation only if its research-phase net Sharpe
(central costs) is positive **and** at or above the median of its own net placebo; the finalist is the best
qualifier; if none qualifies, nothing advances and validation is not run.

**No configuration qualifies. No finalist advances; the validation phase (2019–2021) and the holdout remain
untouched.** That is the honest outcome of this stage, not a failure to be tuned away.

### 8.6 Limits of Stage 6

* The cost model is a model. Spreads are a documented tier (no quotes), impact uses a coefficient of order 0.5,
  and market-on-close execution is idealised; real costs could be lower (better execution, internal crossing)
  or higher (crowding, adverse selection, a wider spread when the signal fires).
* Costs were evaluated on the four research folds only; the conclusion is about *this* strategy family on
  *these* blocks, with intervals of roughly ±1 Sharpe: it does not prove that no profitable configuration exists.
* The cost hook was changed to `cost_fn(frame, y, x)` and the P&L frame now carries per-leg turnover; Stage 5's
  gross results are unchanged (checked to 2e-16).
* Costs re-score the nine registered Stage 5 configurations (logged as diagnostics, not new strategy trials, so
  the registry's count of real-data strategy trials stays at 18); the sensitivity grid is explanatory and no
  configuration was chosen from it.

## 9. PCA statistical arbitrage (Stage 7)

`statarb/models/pca.py`, `statarb/signals/residuals.py`, `statarb/portfolio/neutral.py`,
`statarb/backtest/pca_walkforward.py`; experiments `stage7_pca.py` and `stage7_placebo_checks.py`. This is a
second, independent strategy family — trade the *residual* of each stock against a statistical factor model
(Avellaneda & Lee, 2010) instead of a cointegrated partner — held to the **same standard as the pairs**: the same
point-in-time universe and eligibility, the same four walk-forward research blocks (test years 2015–2018, rolling
four-year training window), the same Stage 6 cost model at the same $100 M, a placebo, and a pre-specified
finalist rule. Validation (2019–2021) and the holdout (2022+) are untouched. Every number below is generated from
`experiments/results/stage7_pca.json` and `stage7_placebo_checks.json`.

### 9.1 The model

* **Factors.** Daily total returns of the fold's eligible names (308–361 across the four folds) are
  standardised and eigen-decomposed (`numpy.linalg.eigh` on the correlation matrix; checked against
  scikit-learn). The PCA is **refitted every 21 sessions on the trailing 504 sessions, using only rows before the
  first day it scores**; nothing is fitted on the block it trades. Each fold's `k` first components are the
  factors; the first 8–11 eigenvalues exceed the Marchenko–Pastur noise edge, and `k = 5` /
  `k = 15` explain 41–53 % / 52–61 % of variance.
  The grid brackets the MP count from both sides.
* **Residuals.** Each day, every stock's returns over the last 60 days are regressed (no intercept) on the factor
  returns; the residual of the *scored day* is out of sample (betas fitted through the day before). Factor returns
  are recomputed with the current fit's weights, so betas never straddle a rotated factor definition.
* **Two signals**, both "positive = rich": `sscore` — Avellaneda–Lee: the cumulative residual is an OU process,
  `s = (X_t − m)/σ_eq`, names with mean-reversion slower than ~30 days (`κ ≤ 8.4`) get no score
  (24–29 % of stock-days); `reversal` — the standardised sum of the last five
  residuals (short-horizon liquidity reversal). Both use fixed literature-style constants; nothing was tuned.
* **Positions.** Enter at `|score| > 1.25`, exit inside `0.5`, stop at 4, 60-day time stop (stops and time stops block
  re-entry until the score returns inside the band; NaN flattens). Each name is 2 % of capital at median residual
  volatility, scaled by inverse residual volatility (capped at 2.5×). The book is then projected to be **exactly
  dollar- and factor-neutral** (the smallest change in weights with `Σw = 0` and `B'w = 0`; the hedge is spread over the
  whole tradable universe and re-traded daily, and is charged). Gross floats with the number of open names (≈ 2.1–2.5× capital
  on average, about 85–94 signal names), and every position is flat at each block's end.
* **Timing and events** are as in the pair engine: a position set at the close of `d` earns bar `d + 1`; a name is flat
  from the day before a non-ordinary distribution's ex-date, and its event-bar return is missing.
* **Grid (6 registered trials):** signal ∈ {s-score, reversal} × `k` ∈ {5, 10, 15}. **Control:** `reversal` with
  `k = 0` (raw return, dollar-neutral only) — a diagnostic that cannot be selected.

### 9.2 Validation

Built and tested before the grid was run (the last item was found afterwards):

* PCA: eigenvalues and eigenvectors against a correlation-matrix `eigh` and scikit-learn; the decomposition
  identities (`Z = FV' + ε`, `F'ε = 0`, `var(F) = λ`); recovery of a planted three-factor structure
  (Marchenko–Pastur counts exactly three; the fitted space contains the planted loadings); standardisation with the
  *fitted* mean and scale, never the new data's.
* **Causality**, with the leakage detectors of §1.7 applied to the scores, the residual volatilities, the betas and
  the out-of-sample residuals (truncation and future-perturbation), including a negative control; betas and
  volatilities do not move when the scored day's return is perturbed (also on the first day of a refit segment, where
  a fit that peeked at that day would move them); an end-to-end test rebuilds the store with everything after a date
  replaced by noise and finds P&L, costs and turnover before it unchanged.
* The OU score against `statsmodels` OLS and its closed forms; the reversal score against a direct computation; the
  residual-volatility estimator is unbiased (a `1/(w − k)` divisor, checked against the known idiosyncratic σ); the state
  machine (thresholds, stops, time stops, NaN, prefix property, column independence).
* Neutrality to 1e-12 for the fitted factors and dollars (also for `k = 0`); the batch cost model **reproduces the pair
  cost model** component by component on a two-name book; drift-aware turnover and the one-bar delay by hand.
* **Mutation sweep:** 45 single-bug mutants, 43 of them applicable to the final code (two mutated clauses were deleted as
  redundant); 42 are killed and one survives as an *equivalent* mutant (projecting an all-zero book gives zero either
  way). The first sweep left seven survivors: four were real test gaps (a fresh entry beyond the stop; the event mask; the
  direction of the placebo relabelling, which my test had re-implemented instead of calling; NaN handling with no
  exit band), two were redundant clauses (now deleted), one equivalent.
* **A bug found by reading the results, not by a test.** In the first run the no-factor control had a net exposure of
  6.5× capital: the projection returned the weights unchanged when there were no factors, contradicting the
  specification, and a test asserted the wrong behaviour. Fixed (the control is now dollar-neutral), the test replaced,
  a regression mutant added, and the experiment re-run in full. The grid (`k ≥ 5`) was unaffected and its results are
  identical; the registry keeps both control runs and the first is void.
* **A second, older bug surfaced by the full test run** (a property test on the Stage 4 pair state machine failed
  on a rare example): see the correction in §6.5. It changed no Stage 3–6 result.

### 9.3 The placebo, and what it can and cannot say

The null for a cross-sectional signal is a **label permutation**: each score path is kept exactly (its persistence,
its NaN pattern, its turnover) but attached to another stock — the same permutation of the names for every day of
the block — and sizing, neutralisation and costs then run on the stock that actually holds the position. 200 draws per
fold, shared across configurations. It severs the link between a signal and *that stock's* returns and nothing else.

* **Calibration under a true null** (`stage7_placebo_checks.py`, synthetic returns with a random-walk residual, 40
  seeds × 60 draws): mean rank of the real book 0.56; ranks below the 5 % / above the 95 % point in
  5 % / 10 % of seeds; rank histogram over
  quintiles [6, 8, 7, 6, 13]; the placebo's dispersion is 0.92 of the real
  book's. Acceptable, with a mild excess in the upper tail (slightly anti-conservative). *The pass thresholds in that
  script were written after I had seen a first run of the same calibration, so they are generous by construction and
  are a consistency check, not an independent test.*
* **Real-data comparability** (`pca_reversal_k5`, research folds): the placebo book turns over 0.91× as much as the
  real book, holds 0.93× the gross exposure and pays 0.91× the impact
  (each cost component within 0.91–0.93×):
  selecting on a standardised score favours names whose volatility estimate is low, and sizing then loads up on
  them, which a relabelled book cannot copy. The real book's **daily P&L volatility is 1.7× the
  placebo's** (positions sit in names that have just moved, and volatility clusters).
* **Consequence.** The *gross* percentile is the meaningful test of a signal. The *net-Sharpe* percentile is **not**
  interpretable here: when net returns are dominated by costs, Sharpe is scaled by volatility and the higher-volatility
  real book looks less negative than the placebo (the JSON's net percentiles of 94–100 % would be read as skill by
  mistake). The finalist rule therefore rests on `net Sharpe > 0`, which decides it on its own.

### 9.4 Result: the research phase, net of costs (2015–2018)

| Configuration | Gross Sharpe (95 % CI) | Gross Sharpe percentile in placebo | Net Sharpe | Turnover / yr | Signal names held | Gross P&L, bps/yr | Cost, bps/yr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s-score, k = 5 | +0.89 [-0.05, +1.83] | 96.5 | -3.18 | 195× | 85 | 380 | 1741 |
| s-score, k = 10 | +0.53 [-0.41, +1.39] | 84.0 | -4.82 | 235× | 86 | 204 | 2056 |
| s-score, k = 15 | +0.26 [-0.67, +1.13] | 71.0 | -6.18 | 280× | 87 | 96 | 2427 |
| reversal, k = 5 | +1.13 [+0.07, +2.17] | 99.5 | -3.54 | 307× | 93 | 684 | 2800 |
| reversal, k = 10 | +0.75 [-0.28, +1.78] | 93.5 | -5.09 | 333× | 93 | 386 | 2987 |
| reversal, k = 15 | +0.76 [-0.30, +1.79] | 94.0 | -5.82 | 361× | 94 | 369 | 3189 |
| reversal, k = 0 (control) | -0.22 [-1.10, +0.68] | 46.5 | -2.69 | 312× | 94 | -228 | 2593 |

*Gross* and *net* Sharpe, annualised on 252 days over the four stitched blocks (n = 1006 days); intervals from the stationary
block bootstrap; "percentile" is the share of 200 placebo books with a lower gross Sharpe. The control is dollar-neutral
only; the six configurations are exactly neutral to their `k` factors (largest |net exposure| ≈ 3e-12).

Per-block gross Sharpe (one row per configuration; each block is one calendar year):

| Configuration | 2015 | 2016 | 2017 | 2018 |
| --- | ---: | ---: | ---: | ---: |
| s-score, k = 5 | +2.65 | +0.45 | +0.01 | +0.76 |
| s-score, k = 10 | +1.33 | +0.95 | +0.21 | -0.12 |
| s-score, k = 15 | +0.84 | +1.02 | -0.33 | -0.43 |
| reversal, k = 5 | +1.28 | +0.06 | +1.44 | +1.80 |
| reversal, k = 10 | +1.13 | +0.31 | +0.19 | +1.38 |
| reversal, k = 15 | +0.38 | +0.56 | +0.32 | +1.54 |
| reversal, k = 0 (control) | -0.52 | -0.06 | -0.46 | -0.03 |

Where the money goes (bps of capital per year, central cost model at $100 M):

| Configuration | Gross | Spread | Commission | Impact | Borrow | Total cost | Net if impact = 0 | Break-even cost multiple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s-score, k = 5 | 380 | 330 | 147 | 1211 | 53 | 1741 | -150 | 0.22 |
| s-score, k = 10 | 204 | 395 | 176 | 1431 | 54 | 2056 | -421 | 0.10 |
| s-score, k = 15 | 96 | 471 | 210 | 1691 | 55 | 2427 | -640 | 0.04 |
| reversal, k = 5 | 684 | 516 | 230 | 1992 | 62 | 2800 | -124 | 0.24 |
| reversal, k = 10 | 386 | 559 | 249 | 2116 | 62 | 2987 | -484 | 0.13 |
| reversal, k = 15 | 369 | 606 | 270 | 2250 | 63 | 3189 | -569 | 0.12 |
| reversal, k = 0 (control) | -228 | 525 | 234 | 1777 | 57 | 2593 | -1044 | 0.00 |

Sensitivity of the **net** Sharpe (descriptive; nothing was chosen from it):

| Configuration | $10 M | $100 M (central) | $1 B |
| --- | ---: | ---: | ---: |
| s-score, k = 5 | -1.25 | -3.18 | -9.01 |
| s-score, k = 10 | -2.28 | -4.82 | -12.29 |
| s-score, k = 15 | -3.13 | -6.18 | -14.71 |
| reversal, k = 5 | -1.26 | -3.54 | -10.64 |
| reversal, k = 10 | -2.26 | -5.09 | -13.54 |
| reversal, k = 15 | -2.65 | -5.82 | -15.01 |
| reversal, k = 0 (control) | -1.54 | -2.69 | -6.21 |

| Configuration | costs × 0 (= gross) | × 0.25 | × 0.5 | × 1 (central) | × 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| s-score, k = 5 | +0.89 | -0.13 | -1.15 | -3.18 | -7.15 |
| s-score, k = 10 | +0.53 | -0.81 | -2.15 | -4.82 | -9.97 |
| s-score, k = 15 | +0.26 | -1.36 | -2.98 | -6.18 | -12.18 |
| reversal, k = 5 | +1.13 | -0.03 | -1.19 | -3.54 | -8.22 |
| reversal, k = 10 | +0.75 | -0.71 | -2.17 | -5.09 | -10.78 |
| reversal, k = 15 | +0.76 | -0.88 | -2.53 | -5.82 | -12.12 |
| reversal, k = 0 (control) | -0.22 | -0.84 | -1.46 | -2.69 | -5.10 |

Exposures and overlap with the pair strategy (descriptive; gross P&L):

| Configuration | Market beta (vs equal-weight universe) | Correlation with equal-weight universe | Correlation with the mean Stage 5 pair P&L (gross) |
| --- | ---: | ---: | ---: |
| s-score, k = 5 | +0.027 | +0.08 | +0.15 |
| s-score, k = 10 | +0.026 | +0.09 | +0.09 |
| s-score, k = 15 | +0.027 | +0.09 | +0.07 |
| reversal, k = 5 | +0.054 | +0.12 | +0.08 |
| reversal, k = 10 | +0.041 | +0.11 | +0.08 |
| reversal, k = 15 | +0.046 | +0.13 | +0.08 |

For comparison, the pair strategy on the same blocks (Stages 5–6, nine configurations): gross Sharpe -0.32 to +0.30,
net -1.01 to -0.58, at a turnover of 11–28× capital a year. The PCA books have more gross signal and
roughly 7–32× the turnover.

### 9.5 Hypotheses and the finalist rule

* **H7a — residual reversal is real gross (reversal configs above the 95th placebo percentile): partly.** `k = 5`
  is at the 99.5th percentile (Sharpe +1.13, the only interval that excludes zero,
  [+0.07, +2.17]); `k = 10` and `k = 15` are at the 93.5th and
  94.0th, just short of the pre-specified line. The s-score with `k = 5` reaches the 96.5th; with
  `k = 10, 15` it does not. Gross Sharpe falls as factors are added (s-score +0.89 → +0.53 → +0.26;
  reversal +1.13 → +0.75 → +0.76). The best of six correlated configurations looks better than a typical one by construction, and its Sharpe varies by block (+1.28, +0.06, +1.44, +1.80 for
  reversal `k = 5`): this is **not** corrected for having looked at six here; §10 does that.
* **H7b — every configuration is net negative at $100 M: confirmed**, net Sharpe -6.18 to -3.18.
  Turnover is 195–361× capital a year (≈ 85–94 signal names at a time)
  against gross of 96–684 bps a year; costs are 1741–3189 bps, about
  70 % of it market impact. On 19–32 % of days some trade exceeds 10 % of the name's ADV (flagged, not cured).
  Costs would have to fall to 0.04–0.24 of the central model for break-even.
* **H7c — the factor model adds something: confirmed, decisively.** The dollar-neutral raw-return control has gross Sharpe
  -0.22 (46th percentile): nothing. The signal lives in the *residual after
  removing the common factors*, not in short-term reversal of raw returns.
* **H7d — capacity: not the constraint.** Even at $10 M the best net Sharpe is -1.25; and with market impact set to
  zero every configuration still loses (net -640 to -124 bps a year): spread, commission and borrow
  alone exceed the gross. The strategy is not too big; it trades too much for what each trade earns. (At a quarter of all costs the best net Sharpe is
  -0.03.)

**Finalist rule** (identical to Stage 6: net Sharpe > 0 *and* ≥ the median of its own net placebo; the best qualifier
advances): **no configuration qualifies. No finalist advances; validation and the holdout remain untouched.**

### 9.6 What this does and does not show

* A statistical factor model's residuals **do** carry a short-horizon reversal that the raw returns do not, on these
  blocks, before costs — a Sharpe of about 1 at best, with an interval of about ±1. It is
  the strongest gross result in the project so far, and costs are 4–25 times as large as it.
* It does **not** show that no profitable implementation exists: a lower-turnover variant, better execution (passive
  fills earning the spread rather than paying it), or a different number of factors could differ.
  None of those was tried, on purpose: they would be new trials selected after seeing this result.
* Diversification: the gross P&L is nearly uncorrelated with the pair strategy (+0.07 to +0.15) and has
  |market beta| ≤ 0.05. Neither matters while the net is negative.

### 9.7 Limits of Stage 7

* **Survivorship probably works in the strategy's favour here.** The universe is names with data; reversal buys recent
  losers, and a loser that was later delisted is exactly a name this data does not contain. The gross result is likely biased
  *upward*, by an amount that cannot be measured with this data.
* **Execution is at the same close that generates the signal.** Short-horizon reversal partly reflects the bid–ask
  bounce; the model charges a half-spread, but a signal computed from the closing print and filled at it is optimistic.
* **Leverage is free of financing** on the long side (gross ≈ 2.1–2.5× capital; only borrow on shorts is charged); no
  hard-to-borrow names; no taxes.
* **The refit schedule differs from the pairs'** (monthly PCA refit on the trailing data, against a screen frozen for a
  year); both use only data before the decision, but the families are not perfectly like for like.
* Four blocks; the intervals are about ±1 Sharpe. The six configurations are all registered trials (the registry's count of
  real-data strategy trials is now 24 distinct, in 30 runs — six are the re-run after the control fix); the control is a diagnostic.
* The placebo preserves score paths, not the link between a score and the stock's own volatility estimate (§9.3); the
  s-score does not correct the OU coefficient for small-sample bias, and cross-sectional demeaning of its mean is not used.
* The hedge trades the whole tradable universe daily; a real implementation would trade a few eigenportfolio proxies
  and would not reproduce this turnover exactly.

## 10. Multiple-testing correction (Stage 8)

`statarb/statistics/sharpe.py`, `statarb/statistics/multiple_testing.py` (and a vectorised stationary bootstrap in
`bootstrap.py`); experiment `stage8_multiple_testing.py`. The question: **of the gross performance found so far, how
much is what a search over many strategies would produce from luck alone?** Everything is computed from series and
screens that already exist for the research phase (2015–2018); **this stage adds no trial**, and validation (2019–2021)
and the holdout (2022+) are untouched. Every number is generated from `experiments/results/stage8_multiple_testing.json`.

### 10.1 What is counted

The registry holds **24 distinct real-data strategy trials** (30 runs, six of them the Stage 7 re-run): Stage 3: 1, Stage 4: 8, Stage 5: 9, Stage 7: 6
(the Stage 3 screen counts once although it tests ~1,300 hypotheses inside). Stage 6's re-scoring, the Stage 7 no-factor control, the
placebos, the simulations and this stage are diagnostics and are not counted. Only **15** of the 24 have daily return series on the
same 2015–2018 sample (1006 days: the nine Stage 5 pair configurations and the six Stage 7 PCA configurations); the joint tests use
those, and the deflated Sharpe uses N = 24. Two things make 24 a **lower bound**: design choices made while building (the z-window, the
`k` grid, which hedge methods to carry into Stage 5, the choice to try a PCA family after the pairs failed) were never registered; and
the trials are correlated, which pulls the *effective* number down. The 15 gross series have a mean pairwise correlation of
0.41; the effective number of independent trials is **9.3** by the average-correlation formula
(Bailey & López de Prado) and **3.0** by the participation ratio of the eigenvalues. The deflation is reported at all three
(plus N = 15).

### 10.2 The tools, and how they were validated

| Tool | Answers |
|---|---|
| Probabilistic Sharpe ratio (PSR) | is the Sharpe positive, given its sampling error *including skew and kurtosis*? |
| Expected maximum of N luck-only Sharpes; Deflated Sharpe ratio (DSR) | is it positive *after* the best of N was picked? (`DSR = PSR` against the luck benchmark `SR0`) |
| Minimum track record / backtest length | how much data would make a given Sharpe believable, for a given N? |
| White's Reality Check, Hansen's SPA | can the **best of the family** be told from luck (stationary bootstrap of the dates, same dates for every strategy)? |
| Romano–Wolf step-down; BH / BY | which **individual** strategies (or pairs) survive family-wise / false-discovery control? |
| CSCV probability of backtest overfitting (PBO) | does the in-sample winner stay a winner out of sample? |

* **Closed forms by hand and by simulation:** the PSR's non-normal standard error is checked against the simulated sampling
  error of a skewed, fat-tailed series (within 6 %; the naive formula is off by more than three times as much); the expected
  maximum against simulated maxima; the DSR against a null world (20 zero-skill strategies: the naive PSR of the best rejects in
  more than 40 % of runs, the DSR in under 9 %) and a power case.
* **BH and BY equal `statsmodels`** to 1e-12 on random p-values with ties; BH holds the FDR in simulation; the vectorised
  bootstrap has the stationary bootstrap's law; the **size and power** of the Reality Check, SPA and Romano–Wolf are checked by Monte
  Carlo (≈ 5 % rejection under the null; > 90 % power for an effect of about 4.8 standard errors among ten); SPA is shown to be **scale-invariant**
  and the Reality Check **not**, and SPA to keep its power when poor strategies are added where the Reality Check loses it; the CSCV
  agrees with a brute-force enumeration and gives ≈ 0.5 on noise, ≈ 0 for a persistent winner.
* **Mutation sweep:** 38 single-bug mutants of the two modules and the bootstrap, 37 applicable to the final code; the first sweep left
  12 survivors, of which eight were genuine test gaps (a benchmark of zero hid a sign error in the minimum track record; unit-variance
  test data hid the missing studentisation; a threshold, a floor, a `+1` in the bootstrap p-value and Romano–Wolf's step-down
  monotonicity were untested); all are now killed, one redundant line was deleted, and two equivalent mutants remain.
* Three of my own first power thresholds were mis-sized by arithmetic (e.g. a true Sharpe of 2 with 20 trials and two years of data is
  detected ~25 % of the time, not 60 %), which is why those tests now use effects that are detected ~99 % of the time.
* **A property of CSCV worth knowing:** the regression slope of out-of-sample on in-sample Sharpe is *negative* even for pure noise
  (−0.35 to −0.86 in simulation; the two halves are complementary). It is stored but carries no information here.

### 10.3 The selection-luck benchmark

With 4.0 years of daily data, the best of N strategies with **no** skill shows an annualised Sharpe of:

| Luck-only strategies tried (N) | Expected best annual Sharpe at 4.0 years | Years of data before that falls below Sharpe 1.0 |
| --- | ---: | ---: |
| 1 | 0.00 | 0.0 |
| 5 | 0.60 | 1.4 |
| 15 | 0.89 | 3.1 |
| 24 | 0.99 | 3.9 |
| 50 | 1.14 | 5.2 |
| 100 | 1.27 | 6.4 |

The Sharpe estimates of the 15 trials themselves have a standard deviation of 0.48 (annualised), which gives
`SR0` = **0.94** at N = 24 (and 0.73 at N = 9.3). The best gross strategy is at 1.13.

### 10.4 Result

Gross Sharpe, and what is left of it after deflation and family-wise control (all 15 series; the six PCA configurations first):

| Strategy (gross) | Sharpe | PSR vs 0 | DSR, N = 24 | DSR, N = 9.3 | DSR, N = 3.0 | Romano–Wolf p | BH-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA reversal, k = 5 | +1.13 | 0.988 | 0.646 | 0.786 | 0.925 | 0.115 | 0.228 |
| PCA s-score, k = 5 | +0.89 | 0.961 | 0.454 | 0.619 | 0.828 | 0.166 | 0.228 |
| PCA reversal, k = 15 | +0.76 | 0.935 | 0.357 | 0.521 | 0.757 | 0.321 | 0.318 |
| PCA reversal, k = 10 | +0.75 | 0.931 | 0.353 | 0.515 | 0.750 | 0.296 | 0.318 |
| PCA s-score, k = 10 | +0.53 | 0.855 | 0.203 | 0.341 | 0.594 | 0.414 | 0.422 |
| PCA s-score, k = 15 | +0.26 | 0.696 | 0.084 | 0.169 | 0.378 | 0.639 | 0.595 |
| expanding, entry 1.5 | +0.30 | 0.728 | 0.101 | 0.196 | 0.416 | 0.609 | 0.595 |
| static, entry 1.5 | +0.23 | 0.676 | 0.077 | 0.157 | 0.358 | 0.639 | 0.595 |
| expanding, entry 2.0 | +0.01 | 0.510 | 0.031 | 0.075 | 0.213 | 0.726 | 0.740 |
| static, entry 2.0 | -0.16 | 0.377 | 0.014 | 0.037 | 0.128 | 0.824 | 0.740 |
| static, entry 2.5 | -0.32 | 0.259 | 0.006 | 0.017 | 0.071 | 0.849 | 0.740 |
| expanding, entry 2.5 | -0.30 | 0.277 | 0.006 | 0.020 | 0.078 | 0.849 | 0.740 |
| Kalman, entry 1.5 | -0.25 | 0.306 | 0.008 | 0.024 | 0.092 | 0.849 | 0.740 |
| Kalman, entry 2.0 | -0.11 | 0.414 | 0.018 | 0.046 | 0.150 | 0.813 | 0.740 |
| Kalman, entry 2.5 | -0.18 | 0.363 | 0.013 | 0.035 | 0.121 | 0.824 | 0.740 |

PSR is the probability that the true Sharpe is positive. Romano–Wolf and BH are computed on the bootstrap p-values of the 15-strategy family.
Every net-of-cost Sharpe is negative, so their deflated Sharpe is 0.000 and there is nothing to deflate.

**Family tests.** The null is "no strategy in the family has a positive mean return".

| Family (gross) | Best | Reality Check p | SPA p: lower / consistent / upper | Romano–Wolf rejections at 5 % | BH / BY rejections at 5 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 15 strategies | PCA reversal, k = 5 | 0.027 | 0.096 / 0.115 / 0.115 | 0 | 0 / 0 |
| 9 pair configurations | expanding, entry 1.5 | 0.443 | 0.395 / 0.492 / 0.492 | 0 | 0 / 0 |
| 6 PCA configurations | PCA reversal, k = 5 | 0.025 | 0.067 / 0.067 / 0.067 | 0 | 0 / 0 |

Different block lengths (5 and 20 days instead of 10) give Reality Check p = 0.025 / 0.025 and SPA p = 0.113 / 0.105
for all 15. Net of costs, nothing is close to rejecting:

| Family (net of costs) | Reality Check p | SPA p (consistent) | Strategies with positive net Sharpe |
| --- | ---: | ---: | ---: |
| All 15 | 0.998 | 1.000 | 0 |
| 9 pair configurations | 0.974 | 1.000 | 0 |
| 6 PCA configurations | 1.000 | 1.000 | 0 |

**Probability of backtest overfitting** (CSCV over the gross series):

| Family (gross) | PBO, 16 blocks (12,870 splits) | PBO, 8 blocks | Mean Sharpe of the in-sample winner: in-sample → out-of-sample | Splits where the winner loses money out of sample |
| --- | ---: | ---: | ---: | ---: |
| All 15 | 0.218 | 0.071 | 1.28 → 0.60 | 16 % |
| 9 pair configurations | 0.263 | 0.257 | 0.35 → 0.04 | 47 % |
| 6 PCA configurations | 0.491 | 0.500 | 1.26 → 0.72 | 10 % |

**False discovery on the pair screens.** The calibrated p-values (§5) of every candidate pair, with Benjamini–Hochberg and Benjamini–Yekutieli:

| Screen | Candidates | Calibrated p ≤ 0.05 | Expected if all null | Smallest p (floor) | BH at q = 0.05 / 0.10 / 0.20 | BY at q = 0.05 / 0.10 / 0.20 | Storey π₀ (λ = 0.3 / 0.5 / 0.7) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 3 window (train 2011–2015) | 1259 | 43 | 63 | 8.3e-04 (1.0e-04) | 0 / 0 / 0 | 0 / 0 / 0 | 1.00 / 0.97 / 0.97 |
| Fold, test 2015 | 1222 | 68 | 61 | 4.9e-03 (1.1e-04) | 0 / 0 / 0 | 0 / 0 / 0 | 0.96 / 0.88 / 0.75 |
| Fold, test 2016 | 1275 | 47 | 64 | 1.0e-04 (1.0e-04) | 0 / 0 / 1 | 0 / 0 / 0 | 1.00 / 1.00 / 1.00 |
| Fold, test 2017 | 1345 | 52 | 67 | 9.6e-05 (9.6e-05) | 0 / 0 / 1 | 0 / 0 / 0 | 1.00 / 1.00 / 0.86 |
| Fold, test 2018 | 1401 | 54 | 70 | 6.3e-04 (9.1e-05) | 0 / 0 / 0 | 0 / 0 / 0 | 1.00 / 1.00 / 1.00 |

*Post-hoc addition, made after the first run:* with 10 null panels the smallest attainable p-value is ≈ 1e-4 while BH at q = 0.05 with ≈ 1,300
candidates needs the first rejection at ≈ 4e-5, so "zero discoveries at 5 %" is partly forced by the null's resolution. Storey's π₀ (the estimated
share of true nulls) does not depend on it: 0.88–1.00 at λ = 0.5.

**Decision rule** (fixed before running): a positive gross result survives only if the best strategy has DSR ≥ 0.95 at N = 24 **and** the SPA
p-value of its family is ≤ 0.05. The best, PCA reversal, k = 5, has a DSR of **0.646** and an SPA p of **0.067** (PCA family).
**It does not survive**: the gross reversal effect is not distinguishable from selection luck at the standard the project set itself.

### 10.5 Hypotheses

* **H8a — no strategy has a DSR ≥ 0.95 at N = 24: confirmed.** The best is 0.646; it is still below 0.95 at N = 9.3 (0.786) and at N = 3.0
  (0.925, the most generous count). Its PSR alone is 0.988: without any correction it would look significant (p ≈ 0.012); the
  observed 1.13 is just above the 0.94 that the best of 24 luck-only strategies is expected to show. Believing it against zero would take
  2.1 years of data with no selection at all, at N = 1.
* **H8b — the Reality Check and SPA do not reject: split, and I was partly wrong.** SPA does not reject (p = 0.115 for all 15; 0.067 within the PCA family), but the
  **Reality Check does** (0.027 and 0.025). The Reality Check is not studentised, and the best strategy is also the *most volatile* PCA book
  (6.05 % a year, against 3.75–5.13 % for the other five), so its raw mean is credited for its own
  volatility. A post-hoc check (`stage8_rc_scale_check.py`, added to test an explanation I had first stated wrongly: the pair books are *not* much quieter than the PCA books) rescales every
  series to the same volatility: the Reality Check p moves from 0.027 to 0.074 (all 15) and from 0.025 to
  0.046 (PCA family) while SPA, being scale-free, is unchanged. Scale explains most of the gap, not all of it. SPA is the pre-specified primary.
  The honest reading is *borderline*: a p-value of about 0.03–0.12 for the best of the family, depending on the test.
* **H8c — no strategy has a Romano–Wolf adjusted p ≤ 0.05: confirmed** (smallest 0.115, PCA reversal k = 5; BH and BY reject nothing at 5 %). No pair configuration comes near
  (0.44 for the family).
* **H8d — no cointegrated pairs survive FDR control at q = 0.05: confirmed** in all five screens, by BH and BY. At q = 0.20 BH finds one pair in each of the 2016 and 2017 folds and BY none.
  π₀ of 0.88–1.00 says the p-values are consistent with almost every candidate being a null; in the Stage 3 window the calibrated discoveries (43) were *fewer* than chance (63).
* **H8e — within the PCA family the in-sample winner is not reliably the out-of-sample winner: confirmed**, PBO 0.49 (a coin flip). Across all 15 the PBO is 0.22, but that reflects the
  gap between the two families (PCA above pairs in every split), not skill in ranking configurations within a family.

### 10.6 What this does and does not show

* Stage 7's gross result was the strongest in the project (best Sharpe 1.13, single-test p ≈ 0.012). After accounting for having looked at 24 configurations it is
  **suggestive but not established**: DSR 0.65, family-adjusted p of 0.03–0.12. Combined with §9 (costs are at least four times the gross) the conclusion for this framework is negative on both counts.
* The pair strategy has **no** detectable gross edge at all (Reality Check p 0.44), and the calibrated pair screens find nothing beyond chance, at any FDR level tested.
* This is **not** proof that no relationship exists: the sample is four years, the tests have limited power at this length (`min backtest length` for a Sharpe of 1 at N = 24 is 3.9 years),
  and a real but modest effect would be missed. Failing to reject is not evidence of absence.

### 10.7 Limits of Stage 8

* **The trial count is a lower bound**, and the effective count is uncertain (a range of 3.0–9.3 independent trials, against 24 registered): none of the choices reaches a DSR of 0.95, but the answer
  moves from 0.65 to 0.93 across them.
* The joint tests use the 15 trials that share a sample; Stage 3's screen and Stage 4's 2016 variants enter only through N. The tests are on mean returns (not Sharpe ratios) and assume
  stationary, weakly dependent returns; the block bootstrap length was fixed (10 days; 5 and 20 give the same conclusions).
* The Reality Check and SPA answer "is the best of this family positive?", not "is this family profitable at capacity" — costs are handled in §8–9, and the net series have no positive Sharpe to test.
* BH assumes independence or positive dependence and the pair candidates overlap (a stock appears in several pairs); BY is valid under any dependence and is reported alongside. The null's resolution limits BH at small q (§10.4).
* The deflated Sharpe assumes the trial Sharpe estimates are roughly independent draws with the observed variance; that is an approximation for correlated strategies of two different families.

## 11. Robustness: sensitivity, stability, regimes and capacity (Stage 9)

`statarb/statistics/breaks.py`, `statarb/statistics/regimes.py`, `statarb/backtest/capacity.py`, `reselect_pairs` in `selection/screening.py`;
experiments `stage9_sensitivity.py`, `stage9_regimes_breaks.py`, `stage9_capacity.py` and the post-hoc `stage9_regime_followup.py`. Everything is the research
phase (2015–2018); validation and the holdout are untouched (the ledger is still empty). Numbers are generated from `experiments/results/stage9_*.json`.

**Ground rules.** Every variant evaluated on this data is a *look*. The sensitivity surfaces are therefore registered as diagnostics and reported whole; nothing
is chosen from them, **no variant can advance** (Stages 6–7 found no qualifier and a surface cannot create one), and a variant that looked good would only be a
hypothesis for a future, registered trial on data not yet used. If the surface *were* treated as trials (93 configurations were evaluated across the PCA and pair
surfaces), the deflated Sharpe of its best point (1.21) would be **0.38** at N = 24 + 93 (the luck benchmark rising to 1.35), against 0.65 at N = 24 in §10.

### 11.1 Validation

* **Sup-F break test:** the F statistic against a brute-force loop; size ≈ 5 % under no break (i.i.d. and AR(1) data), power > 90 % for a one-standard-deviation shift, the break located
  within a few per cent of the sample, the p-value never zero and 1 for a flat series; **subspace overlap** on its identities (1 for the same span in any basis, 0 for orthogonal, ≈ k/N for random subspaces).
* **Regimes:** the labels pass the leakage detectors (truncation and future perturbation) and a day's own return cannot move its label; the Sharpe-difference bootstrap has ≈ 5 % size and > 90 % power.
* **Capacity:** the analytic scaling (impact per unit of capital ∝ √capital, participation ∝ capital) **reproduces a real re-run of the PCA engine at other capitals to 1e-13**, holds for the pair cost model,
  and, in the experiment, reproduces Stage 7's own re-runs at $10M and $1B to 1e-12. `reselect_pairs` reproduces the screen's own `selected` column on all four folds.
* **Mutation sweep:** 31 single-bug mutants; 30 applicable to the final code (one clause, a no-op re-centring, was deleted); the first sweep left nine survivors, all real test gaps
  (the p-value floor and tie rule, the reported break index, the trend window, the dispersion smoothing, the boundary conventions of the re-selection, break-even at exactly zero room), now all killed.
* **Deviations and misses, in the order they happened:** the first draft of the PCA fit-window grid had 252 days; it stopped at once because 252 days is fewer than the 308–361 names (a singular correlation
  matrix), so it became 378 before any result existed. My prior for the regime tests (H9e, below) was wrong.

### 11.2 Parameter sensitivity (hypothesis C)

Anchors fixed a priori: PCA reversal k = 5 (the Stage 7 result under test) and the class defaults (s-score, k = 10); pairs: the `StrategyConfig` defaults (static hedge, entry 2, exit 0.5, stop 4, max hold 60,
z-window 60). One dimension is moved at a time (gross / net Sharpe, four folds, Stage 6 costs at $100M):

| Variation (one dimension at a time) | Reversal k = 5: gross | net | s-score k = 10: gross | net |
| --- | ---: | ---: | ---: | ---: |
| **anchor** | +1.13 | -3.54 | +0.53 | -4.82 |
| entry = 1.0 | +1.04 | -4.38 | +0.93 | -5.45 |
| entry = 1.5 | +0.96 | -3.03 | +0.41 | -4.22 |
| entry = 2.0 | +0.61 | -2.68 | +0.91 | -2.76 |
| exit = 0.25 | +1.21 | -3.01 | +0.68 | -3.95 |
| exit = 0.75 | +1.20 | -4.02 | +0.60 | -5.67 |
| stop = 3.0 | +1.13 | -3.84 | +0.47 | -5.19 |
| stop = 6.0 | +1.12 | -3.40 | +0.45 | -4.82 |
| max_hold = 20 | +1.12 | -3.56 | +0.54 | -4.83 |
| name_size = 0.01 | +1.13 | -2.56 | +0.53 | -3.73 |
| name_size = 0.04 | +1.13 | -4.93 | +0.53 | -6.33 |
| n_factors = 3 | +1.01 | -2.92 | +0.87 | -2.57 |
| n_factors = 8 | +0.85 | -4.34 | +0.81 | -3.87 |
| n_factors = 10 | +0.75 | -5.09 | n/a | n/a |
| n_factors = 15 | +0.76 | -5.82 | +0.26 | -6.18 |
| n_factors = 20 | +0.28 | -7.33 | +0.48 | -7.31 |
| fit_window = 378 | +1.03 | -3.64 | +0.44 | -4.75 |
| fit_window = 756 | +0.69 | -3.88 | +0.36 | -4.96 |
| refit_every = 5 | +1.05 | -3.61 | +0.69 | -4.81 |
| refit_every = 63 | +1.08 | -3.60 | +0.87 | -4.42 |
| beta_window = 40 | +1.10 | -3.72 | +0.85 | -6.82 |
| beta_window = 90 | +1.16 | -3.59 | +0.64 | -3.43 |
| reversal_days = 3 | +0.92 | -5.37 | n/a | n/a |
| reversal_days = 10 | +0.82 | -2.24 | n/a | n/a |
| n_factors = 5 | n/a | n/a | +0.89 | -3.18 |
| kappa_min = 4.2 | n/a | n/a | +0.29 | -5.27 |
| kappa_min = 16.8 | n/a | n/a | +0.72 | -5.16 |

Reading it: **rule for "robust"** (fixed before running): at least 70 % of neighbours with positive gross Sharpe and a median at least half the anchor's. Reversal k = 5: **100 %**
positive, median +1.04 against +1.13, range +0.28 to +1.21, the anchor ranked 5 of 24: **robust**. S-score k = 10:
100 % positive, median +0.60: **robust**. It is not an isolated peak. Two structures are visible: the gross Sharpe falls steadily as factors are added
(k = 3 → 20: +1.01 → +0.28), and every knob that lowers turnover (entry 2.0, a 10-day horizon,
1 % names) improves the *net* result without ever making it positive: the best net Sharpe anywhere on either surface is **-2.24**. Entry × factors, gross / net:

| k \ entry | 1 | 1.25 | 1.5 | 2 |
| --- | ---: | ---: | ---: | ---: |
| k = 3 | +0.92 / -3.70 | +1.01 / -2.92 | +0.97 / -2.53 | +0.92 / -1.92 |
| k = 5 | +1.04 / -4.38 | +1.13 / -3.54 | +0.96 / -3.03 | +0.61 / -2.68 |
| k = 8 | +0.72 / -5.45 | +0.85 / -4.34 | +0.63 / -3.92 | +0.49 / -2.98 |
| k = 10 | +0.78 / -6.02 | +0.75 / -5.09 | +0.70 / -4.42 | +0.44 / -3.50 |
| k = 15 | +0.55 / -7.12 | +0.76 / -5.82 | +0.40 / -5.27 | +0.60 / -3.71 |

**Pairs** (33 variants around the defaults: the anchor has gross -0.16 / net -0.84, turnover 19×):

| Pair variation | Gross | Net | Turnover / yr |
| --- | ---: | ---: | ---: |
| hedge = static | -0.16 | -0.84 | 19× |
| hedge = expanding | +0.01 | -0.71 | 20× |
| hedge = kalman | -0.11 | -0.71 | 17× |
| hedge = rolling | -0.08 | -0.67 | 18× |
| entry = 1.0 | +0.47 | -0.74 | 40× |
| entry = 1.5 | +0.23 | -0.64 | 28× |
| entry = 2.0 | -0.16 | -0.84 | 19× |
| entry = 2.5 | -0.32 | -0.85 | 13× |
| entry = 3.0 | -0.03 | -0.44 | 7× |
| exit = 0.0 | -0.20 | -0.75 | 16× |
| exit = 0.25 | +0.12 | -0.53 | 19× |
| exit = 0.5 | -0.16 | -0.84 | 19× |
| exit = 1.0 | +0.04 | -0.86 | 22× |
| stop = 3.0 | -0.60 | -1.44 | 19× |
| stop = 4.0 | -0.16 | -0.84 | 19× |
| stop = 6.0 | -0.33 | -0.99 | 19× |
| max_hold = 20 | -0.05 | -0.89 | 19× |
| max_hold = 60 | -0.16 | -0.84 | 19× |
| max_hold = 120 | -0.24 | -0.90 | 19× |
| z_window = 30 | -0.18 | -1.21 | 31× |
| z_window = 60 | -0.16 | -0.84 | 19× |
| z_window = 120 | -0.05 | -0.54 | 12× |
| z_window = 250 | -0.10 | -0.52 | 9× |
| screen_alpha = 0.01 | -0.66 | -1.24 | 9× |
| screen_alpha = 0.05 | -0.16 | -0.84 | 19× |
| screen_alpha = 0.1 | -0.44 | -1.04 | 20× |
| screen_alpha = 0.2 | +0.19 | -0.41 | 22× |
| screen_half_life = 5-60 | -0.16 | -0.84 | 19× |
| screen_half_life = 2-120 | -0.13 | -0.80 | 19× |
| screen_half_life = 10-40 | -0.18 | -0.88 | 19× |
| portfolio_size = 10 | -0.31 | -1.00 | 22× |
| portfolio_size = 20 | -0.16 | -0.84 | 19× |
| portfolio_size = 40 | -0.06 | -0.79 | 19× |

Only 18 % of the pair variants have a positive gross Sharpe (best +0.47; none reaches 0.5), the median is -0.16, and none is positive net (best -0.41). There is no region of the
parameter space with a stable gross edge: the six positive variants (entry 1.0 and 1.5, exit 0.25 and 1.0, the expanding hedge, α = 0.20) are scattered with no pattern and none is above 0.5.

### 11.3 Stability (hypothesis B)

**Pair relationships.** For every selected pair and for the baseline pool, the training hedge ratio and mean are frozen and the spread is examined over the test year (Stage 3's metrics), across all four folds:

| Frozen-hedge spread out of sample | Pairs | ADF rejects at 5 % | Median sd ratio (test / train) | Median mean shift (train sd) | Median half-life (days) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected pairs (20 per fold, 4 folds) | 80 | 3.8 % [1.3, 10.5] | 1.25 | 1.79 | 53 |
| Baseline pool (calibrated p > 0.05, beta > 0) | 4807 | 7.1 % [6.4, 7.8] | 0.78 | 1.23 | 50 |

The selected pairs are **not** more likely to stay cointegrated than pairs picked at random (3.8 % against 7.1 % reject; the pool's own rate is above the nominal 5 %, consistent with §4.3's finding that the test is not calibrated),
and their out-of-sample spread is *wider* than in training (median ratio 1.25, against 0.78 for the pool), which is what selecting on in-sample tightness would produce through regression to the mean (a reading, not something separately tested). The rank correlation between a pair's in-sample
calibrated p-value and its out-of-sample ADF p-value is **+0.016** (p = 0.26, 4887 pairs): in-sample cointegration strength carries no information about out-of-sample strength.

**The PCA factor space** is much more stable than any pair. On the 302 names present in every fold, refitted monthly on 504 days (72 fits, 2013-01-07 to 2018-12-07):

| Overlap of the leading-k factor space between fits this many months apart | 1 | 3 | 6 | 12 | 24 | 36 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k = 5 (random subspaces: 0.017) | 0.98 | 0.94 | 0.88 | 0.76 | 0.64 | 0.62 | 0.59 |
| k = 10 (random subspaces: 0.033) | 0.97 | 0.93 | 0.87 | 0.77 | 0.63 | 0.60 | 0.55 |
| First eigenvector (market), abs cosine | 1.000 | 0.999 | 0.997 | 0.993 | 0.985 | 0.981 | 0.961 |

The market direction is essentially fixed (|cos| 0.993 at a year), the leading five- and ten-dimensional spaces stay close to themselves for a few months and drift steadily thereafter, settling at a
persistent overlap of about 0.6 (far above the random level). This is consistent with refitting monthly rather than once a year (at twelve months only ≈ 76 % of the space is shared); the refit frequency itself is a sensitivity dimension in §11.2 and matters little.

**Structural breaks in performance.** A sup-F mean-shift test on each gross series (15 % trimming, bootstrap p): 0 of 15 have p < 0.05 (smallest 0.22); none survives BH. Six illustrative rows:

| Strategy (gross) | Estimated break | Sharpe before → after | sup-F p | BH-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| expanding, entry 1.5 | 2015-08-26 | -1.45 → +0.60 | 0.744 | 0.933 |
| Kalman, entry 2.0 | 2016-02-11 | -1.16 → +0.43 | 0.504 | 0.933 |
| static, entry 1.5 | 2015-08-26 | -1.63 → +0.54 | 0.706 | 0.933 |
| PCA s-score, k = 5 | 2016-04-25 | +2.67 → +0.11 | 0.224 | 0.933 |
| PCA reversal, k = 5 | 2018-05-10 | +0.79 → +2.42 | 0.562 | 0.933 |
| PCA reversal, k = 15 | 2017-10-25 | +0.18 → +1.89 | 0.584 | 0.933 |

For seven of the nine pair configurations the estimated break falls in 2015–16 and is an *improvement* (a Sharpe of about −1.5 before, +0.5 after; the two entry-2.5 configurations show a 2017 decline instead), the PCA s-score's is a decline (2.7 → 0.1) and the reversal's a late improvement; none is distinguishable from
chance. With 1,006 days the test cannot see a fall from a Sharpe of 1 to 0, so **not rejecting is not evidence of stability**.

### 11.4 Regimes (hypothesis G)

Three regimes fixed in advance, each labelled from data through the previous day against an expanding median: market volatility (36 % of days "high"), the market's 126-day trend (85 % "up") and cross-sectional dispersion
(56 % "high"); 15 strategies × 3 regimes = 45 tests of "the Sharpe ratio is the same in both states" (stationary bootstrap, BH):

| Gross Sharpe, first state / second state (bootstrap p) | High vs low volatility | Uptrend vs downtrend | High vs low dispersion |
| --- | ---: | ---: | ---: |
| PCA reversal, k = 5 | +1.69 / +0.74 (0.422) | +0.83 / +2.50 (0.198) | +2.32 / -0.92 (0.001) |
| PCA reversal, k = 10 | +1.11 / +0.51 (0.595) | +0.75 / +0.79 (0.980) | +1.79 / -1.12 (0.004) |
| PCA reversal, k = 15 | +1.34 / +0.37 (0.406) | +0.73 / +0.90 (0.905) | +1.84 / -1.09 (0.004) |
| PCA s-score, k = 5 | +1.51 / +0.48 (0.303) | +0.59 / +2.29 (0.162) | +1.60 / -0.33 (0.048) |
| PCA s-score, k = 10 | +0.80 / +0.34 (0.633) | +0.25 / +1.83 (0.211) | +0.91 / -0.12 (0.289) |
| PCA s-score, k = 15 | +0.31 / +0.22 (0.930) | +0.11 / +1.00 (0.406) | +0.40 / +0.01 (0.675) |
| Pair configurations (9): median difference (smallest p) | +0.39  (0.486) | +0.46  (0.296) | +1.29  (0.050) |

Of the 45 tests 5 have p < 0.05 (2.3 expected by chance) and **3 survive BH at 10 %: the three PCA reversal configurations against the dispersion regime**, which is one effect seen three times (the configurations are highly correlated). It is the only one:
volatility and trend show nothing, and the pair strategies' differences are individually weak (smallest p 0.050), although *all fifteen* strategies have a higher Sharpe when dispersion is high. **This contradicts my prior (H9e: nothing survives).**

Because it was unexpected, it was probed with a post-hoc script (`stage9_regime_followup.py`; labelled as such, not part of the pre-specified verdict, and any further look at these regimes would be another trial):

* the label is not too persistent for the test: 19 high-dispersion episodes with a median length of 26 days; the p-value for reversal k = 5 is stable across mean block lengths of 5 to 63 days:

| Mean bootstrap block (days) | Reversal k = 5 | k = 10 | k = 15 | Strategies of 15 with p < 0.05 |
| --- | ---: | ---: | ---: | ---: |
| 5.0 | 0.0012 | 0.0046 | 0.0046 | 5 |
| 10.0 | 0.0008 | 0.0026 | 0.0048 | 4 |
| 21.0 | 0.0008 | 0.0014 | 0.0024 | 5 |
| 63.0 | 0.0004 | 0.0004 | 0.0022 | 5 |

* a **circular-shift test** (keeps the label's persistence and pattern exactly, breaks only its alignment with the returns; validated on noise, size 5 %) gives p = 0.0034 / 0.0057 / 0.0079 for reversal k = 5 / 10 / 15 and again 3 BH rejections;
* it is **not one episode**: the contrast has the same sign in every year.

| Reversal k = 5, gross Sharpe | 2015 | 2016 | 2017 | 2018 |
| --- | ---: | ---: | ---: | ---: |
| High dispersion | +3.26 | +0.44 | +6.06 | +2.75 |
| Low dispersion | -1.25 | -1.25 | -0.18 | -2.15 |
| Days (high / low) | 132 / 120 | 185 / 67 | 64 / 187 | 182 / 69 |

Interpretation, with care: short-horizon residual reversal earns more when there is more idiosyncratic movement to absorb — a plausible mechanism (liquidity provision is paid when order-flow imbalances are large) — and the effect is strong and consistent in this sample.
But it is one regime split found after looking, in four years, and **it does not rescue the strategy**: even in the favourable regime the *net* Sharpe of reversal k = 5 is -2.33 (against -5.83 otherwise). Trading only in high-dispersion states would be a new, unregistered configuration
chosen from a look at these data, so it was not tried.

### 11.5 Capacity (hypothesis H)

Impact per unit of capital scales with √capital and participation with capital, so the saved cost components at $100M give the net return at any capital exactly (§11.1):

| Strategy | Gross, bps/yr | Spread + commission + borrow | Impact at $100M | Net at zero impact | Break-even capital (central) | … fixed costs × 0.5 | … × 0.25 | Fixed-cost multiple for break-even at vanishing size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA reversal, k = 5 | 684 | 808 | 1992 | -124 | none | $2.0M | $5.9M | 0.85 |
| PCA reversal, k = 10 | 386 | 871 | 2116 | -484 | none | none | $634K | 0.44 |
| PCA reversal, k = 15 | 369 | 939 | 2250 | -569 | none | none | $358K | 0.39 |
| PCA s-score, k = 5 | 380 | 529 | 1211 | -150 | none | $898K | $4.2M | 0.72 |
| PCA s-score, k = 10 | 204 | 625 | 1431 | -421 | none | none | $110K | 0.33 |
| PCA s-score, k = 15 | 96 | 736 | 1691 | -640 | none | none | none | 0.13 |
| expanding, entry 1.5 | 141 | 93 | 314 | 48 | $2.3M | $9.0M | $14.1M | 1.51 |
| static, entry 1.5 | 109 | 93 | 315 | 16 | $247K | $3.9M | $7.4M | 1.17 |
| static, entry 2.0 | -66 | 68 | 219 | -134 | none | none | none | 0.00 |

*Break-even capital* is where the mean net return reaches zero ("none" = negative even at vanishing size). Under the **central costs no PCA configuration breaks even at any capital**: spread, commission and borrow alone (529–939 bps a year)
exceed the gross (96–684); the best, reversal k = 5, would need fixed costs 15 % lower than the tiers assumed and then could hold about $2.0M with fixed costs halved.
**Two pair configurations do break even** at a tiny scale (expanding, entry 1.5: $2.3M; static, entry 1.5: $247K), on a gross edge whose Sharpe is 0.23–0.30 and whose Romano–Wolf adjusted p-value is ≈ 0.6 (§10): a capacity of a few million dollars for something not distinguishable from zero.
**My prior (H9h: zero for all) was wrong for these two.** Participation:

| PCA strategy | Days with a trade above 10 % of ADV at $1M / $10M / $100M / $1B | Capital at which 5 % of days breach |
| --- | ---: | ---: |
| PCA reversal, k = 5 | 0 % / 0 % / 29 % / 100 % | $35.5M |
| PCA reversal, k = 10 | 0 % / 0 % / 29 % / 100 % | $37.2M |
| PCA reversal, k = 15 | 0 % / 0 % / 32 % / 100 % | $38.0M |
| PCA s-score, k = 5 | 0 % / 0 % / 19 % / 100 % | $43.0M |
| PCA s-score, k = 10 | 0 % / 0 % / 22 % / 100 % | $42.3M |
| PCA s-score, k = 15 | 0 % / 0 % / 23 % / 100 % | $38.0M |

At the central $100M, 19–32 % of days breach 10 % of some name's ADV, and 5 % of days already breach at $35.5M–$43.0M. But size is not what stops these strategies: they lose money even at vanishing size (the column "net at zero impact"), because of how much they trade (§9.5).

### 11.6 Hypotheses

* **H9a — the PCA reversal gross effect is not an isolated peak: confirmed** (robust by the pre-specified rule for both anchors; 100 % of one-at-a-time neighbours positive). Its size depends mostly on the number of factors.
* **H9b — no variant has a positive net Sharpe at $100M: confirmed** (best −2.2 for the PCA surface, −0.4 for the pairs).
* **H9c — no stable positive region for the pairs: confirmed** (18 % of variants positive, best +0.47).
* **H9d — pair relationships are unstable out of sample: confirmed** (3.8 % vs 7.1 % ADF rejections, rank correlation +0.016).
* **H9e — no regime difference survives BH: refuted** for the dispersion regime (3 PCA reversal configurations; robust to block length, a persistence-preserving shift test and year by year), with the caveats above.
* **H9f — no break survives BH: confirmed**, with the test's low power. **H9g** (factor space): the market direction and the k = 5 space are stable at short lags as predicted, but *k = 10 is not less stable than k = 5*.
* **H9h — no strategy has positive break-even capital: partly refuted** (two pair configurations, at ≤ $2.3M).

### 11.7 What this shows and does not show

* Within this sample the PCA residual-reversal gross effect is **not a fragile artefact of one parameter setting** (§11.2) and has an economically coherent regime dependence (§11.4). §10 still says it is not distinguishable
  from selection luck after 24 trials; this stage widens the burden (the surface as trials: DSR 0.38) rather than lifting it.
* **Neither family can be made profitable by any tested robustness lever.** Lowering turnover helps the net result; nothing tested makes it positive. The pair relationships do not persist out of sample and their screen has no predictive value.
* Limits: four years and one universe; the surfaces are one-at-a-time (interactions are seen only in the entry × k grid); regimes were split three ways and found one effect after the fact; the break test has little power; capacity is a model of impact
  (§8), and the analytic scaling holds only *within* that model; and every look here is a look at research data that a future registered trial cannot reuse for confirmation.

## 12. Statistical-arbitrage methodology *(hypotheses A–H; Stage 10 will summarise)*

Hypotheses to be tested, each with its data, method, assumptions, uncertainty, failure cases and
limitations recorded when run:

* **A** cointegrated pairs show economically meaningful out-of-sample mean reversion;
* **B** cointegration relationships and hedge ratios are stable through time *(tested in §11.3)*;
* **C** performance is robust to entry/exit thresholds and look-back windows *(tested in §11.2)*;
* **D** residuals of a common-factor (PCA) model mean-revert exploitably *(tested in §9)*;
* **E** how much apparent performance survives realistic costs;
* **F** how much survives correction for the number of hypotheses tested *(tested in §10)*;
* **G** whether behaviour differs across market regimes *(tested in §11.4)*;
* **H** how much capital the strategy can deploy before impact removes the edge *(tested in §11.5)*.

The train / validation / final-holdout split, the walk-forward scheme, the cost model, the
multiple-testing framework and the experiment registry will be documented here as each is built.

## 13. Limitations so far

* The reconstructed S&P 500 membership is trustworthy only from 2011-01-01 (§2.1); earlier dates are
  not used for any claim, and even later dates inherit the change log's residual gaps.
* Free data has no history for many delisted names; the price universe is survivor-tilted and the
  return of the missing names is unobservable. The bias is *quantified*, not removed. No backtest built
  on it may be described as free from survivorship bias.
* GICS sectors are today's classification applied to all dates.
* Ticker renames not in `configs/ticker_aliases.csv` appear as reconstruction anomalies and missing
  prices; the alias table is deliberately short (hand-verified only).
* Retroactive vendor restatements of *dividend* history that do not touch prices are invisible to the
  overlap check; the 90-day full refresh bounds, but does not eliminate, that exposure.
* Yahoo is an unofficial source and can change or rate-limit without notice.
* (Stage 2) The Engle–Granger p-values are not calibrated on real prices (§4.3) and the cause is not
  isolated; the OU half-life is biased fast in short samples (§4.2); Johansen's size is distorted in small
  samples and depends on `det_order`; ADF/Engle–Granger assume homoskedastic errors and read a structural
  break as a unit root. None of these are corrected for yet beyond what §4 states.
* (Stage 2) Critical-value tables come from statsmodels; a defect there would propagate.
* (Stage 9) Everything is on four research years and one universe; the sensitivity surfaces are one-at-a-time and every point is a look
  (93 configurations, none selectable); the dispersion-regime finding is post-hoc in its follow-up and is one split of three found after
  the fact; the break test has little power; capacity is a property of the impact model, not measured (§11.7).
* (Stage 8) The trial count (24) is a lower bound and the effective count uncertain (3–9); the joint tests use the 15 trials
  that share a sample; the tests are on mean returns over four years, so a modest real effect would be missed; the Reality
  Check's answer depends on the strategies' scale (§10.5); BH on the pair screens is limited by the null's resolution (§10.4).
* (Stage 7) The PCA family's gross result is on four research blocks with a Sharpe interval of about ±1, before any
  correction for six registered configurations; the universe is survivor-tilted in the direction that flatters
  reversal; execution is at the signal's own close; the placebo does not preserve the score–volatility link and its
  net-Sharpe percentile is not interpretable (§9.3, §9.7).
* (Stage 6) The cost model is a documented model, not measured execution (no quotes; impact coefficient
  and market-on-close fills are assumptions); costs were evaluated on the four research folds only.
* (Stage 5) Four research-phase test blocks with 20-pair portfolios (Sharpe intervals of about ±1);
  everything gross; validation and the holdout have not been evaluated; the holdout is unopened.
* (Stage 4) All P&L so far is gross of costs and frictionless, on one year of data for the real-pair look;
  the hedge-ratio comparison rests on a simulator that favours the Kalman filter; nothing has been
  evaluated on a holdout.
* (Stage 3) The screen was run on one window with one configuration; the empirical null uses independent
  circular shifts (a small share of pairs keep aligned volatility regimes) and has no factor structure
  (checked only on synthetic data); candidate generation depends on `k` and on today's sector labels.
