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
| 2 | Statistical toolkit: ADF, Engle–Granger, Johansen, OU, half-life | **done** (this commit) |
| 3 | Pair discovery on training data only | not started |
| 4 | Pair strategy: hedge ratios, spread, z-score, entry/exit/stop | not started |
| 5 | Walk-forward backtest, train/validation/holdout | not started |
| 6 | Transaction costs: spread, commissions, impact | not started |
| 7 | PCA statistical arbitrage | not started |
| 8 | Multiple-testing correction, deflated Sharpe, reality check | not started |
| 9 | Robustness: sensitivity, regimes, structural breaks, capacity | not started |
| 10 | Final research report | not started |

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
  selection/           adf, cointegration (Engle-Granger), johansen, hurst (exploratory only)
  models/              ou (OU fit, half-life, simulation), rolling (causal rolling statistics)
tests/                 pytest + hypothesis
experiments/           runnable experiments and their JSON results (registry: Stage 8+)
```

Later stages add `signals/`, `portfolio/`, `backtest/`, `statistics/`, `research/` beside `data/`
inside `statarb/`.

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

## 5. Statistical-arbitrage methodology *(Stages 3–9: specification only)*

Hypotheses to be tested, each with its data, method, assumptions, uncertainty, failure cases and
limitations recorded when run:

* **A** cointegrated pairs show economically meaningful out-of-sample mean reversion;
* **B** cointegration relationships and hedge ratios are stable through time;
* **C** performance is robust to entry/exit thresholds and look-back windows;
* **D** residuals of a common-factor (PCA) model mean-revert exploitably;
* **E** how much apparent performance survives realistic costs;
* **F** how much survives correction for the number of hypotheses tested;
* **G** whether behaviour differs across market regimes;
* **H** how much capital the strategy can deploy before impact removes the edge.

The train / validation / final-holdout split, the walk-forward scheme, the cost model, the
multiple-testing framework and the experiment registry will be documented here as each is built.

## 6. Limitations so far

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
