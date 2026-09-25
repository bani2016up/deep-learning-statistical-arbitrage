# Theory of Deep Learning Statistical Arbitrage: residuals, factor models and the OU baseline

Study guide for the oral defense. It covers the theory behind Guijarro-Ordonez, Pelger, Zanotti,
*Deep Learning Statistical Arbitrage* (arXiv 2106.04028v2, Sep 2022) and the OU+Threshold
baseline.

**Conventions**

- **Page numbers.** "p. N" is the paper's *printed* page number (the PDF page index minus 1).
- **Other documents.**
  - `paper_summary.md` = `docs/paper_summary.md` (the three-step table, data and protocol,
    headline numbers). This guide does not repeat them.
  - `OU_BASELINE.md` = `docs/OU_BASELINE.md` (our replication numbers).
  - `full_report.md` = `docs/full_report.md` (the team's CNN+Transformer results).
- **Code.** File:line references are to the official repository `dlsa-public/`
  (clone into `references/dlsa-public`, see README).
- **Sources outside the paper.** Anything not in the paper is labelled **[general knowledge]** or
  **[ours]**.

## 30-second answers

1. **What is the residual?** $\epsilon_t=\Phi_{t-1}R_t$: long stock *n*, short a diversified
   portfolio with the same factor exposure. It is itself a traded long-short portfolio.
2. **Why residuals and not returns?** Returns are driven by a few factors, so a K = 0 book is a
   market bet. Residuals are close to uncorrelated, so many independent series carry signal.
   K = 0 OU: SR −0.18 (paper), −0.23 (ours).
3. **What is OU+Thresh?** The 30-day cumulative residual is fitted as an OU, which is exactly an
   AR(1). Short if $s>1.25$, long if $s<-1.25$, provided $R^2>0.25$.
4. **Why does OU fail?**
   - With 30 points, $\hat b$ is biased down, so even a random walk looks like it reverts in about
     4 days.
   - $R^2\approx\hat b^2$, so the $R^2$ filter only sets a minimum half-life.
   - It is a pure contrarian bet.
   - Turnover of about 1 per day means costs of about 12% a year.
5. **What did we replicate?**
   - K = 0 matches the paper, which is the clean test.
   - At K = 5 our SRs of 0.34 / 0.93 / 0.64 (FF / PCA / IPCA) bracket the paper's 0.38 / 0.73 /
     0.97.
   - $\Phi$ only reweights days.

---

## 1. Statistical arbitrage in the authors' framing

**What "arbitrage" means here** [general knowledge, except where a page is cited]:

- **Textbook arbitrage** is a zero-cost portfolio with a riskless, non-negative, sometimes positive
  payoff.
- **APT's asymptotic arbitrage** (Ross 1976; Huberman 1982) is a sequence of zero-cost portfolios
  whose variance goes to 0 while the mean stays bounded away from 0.
- **Statistical arbitrage** has a finite Sharpe ratio and can lose money:
  - at an annual SR of 4 the daily SR is $4/\sqrt{252}\approx0.25$, so about 40% of days still
    lose;
  - the August 2007 "quant quake" hit stat-arb funds (p. 28).
- **Self-financing.** $R$ are excess returns, so any $w^R$ is a self-financing position. The
  $\lVert w^R\rVert_1=1$ constraint bounds gross exposure, not net investment (p. 23).
- **Limits to arbitrage** (Shleifer-Vishny 1997): capital constraints, noise-trader risk, forced
  liquidation, crowding, and short-sale costs (the paper's 1 bp holding cost, p. 35).

**The three steps.** Every stat-arb strategy answers three questions (p. 1, Sec. I; p. 7,
Sec. II):

1. **Arbitrage portfolios:** which long-short portfolios of similar assets to trade?
2. **Signal:** what in the recent path of such a portfolio shows a *temporary* deviation?
3. **Allocation:** given the signals, what weights maximize a trading objective under constraints?

`paper_summary.md` §1 has the full table.

**Math.**

- Step 1 fixes the residual portfolios $\epsilon_t=\Phi_{t-1}R_t$.
- Step 2 is a signal map $\theta:\epsilon^L_{n,t-1}\in\mathbb R^L\to\theta_{n,t-1}\in\mathbb R^p$.
- Step 3 is an allocation map $w^\epsilon:\theta_{n,t-1}\to w^\epsilon_{n,t-1}$.
- For the **deep models**, steps 2 and 3 are solved jointly (eq. 4-5, p. 11):

$$\max_{w^\epsilon\in\mathcal W,\ \theta\in\Theta}\ \frac{E[w^{R\prime}_{t-1}R_t]}{\sqrt{\mathrm{Var}(w^{R\prime}_{t-1}R_t)}}
\quad\text{s.t.}\quad w^R_{t-1}=\frac{\Phi'_{t-1}w^\epsilon_{t-1}}{\lVert\Phi'_{t-1}w^\epsilon_{t-1}\rVert_1},\qquad
w^\epsilon_{t-1}=w^\epsilon(\theta(\epsilon^L_{t-1})).$$

- **Classical models** (OU) estimate the signal separately and then solve only for the allocation
  (eq. 6, p. 11). See §7.

**Why the steps can be separated.**

- Step 1 is fixed before trading, so all models trade the same residuals.
- The signal is a sufficient statistic: two residuals with the same signal get the same weight
  (p. 9).
- **Caveat:** under joint training the split into $\theta$ and $w^\epsilon$ is *not uniquely
  identified* (p. 10). The authors choose a split that matches the classical models. For the
  CNN+Transformer, the "signal" is the transformer output at the last time step,
  $h^{proj}_L$, and an FFN does the allocation (p. 19-20).

**What the CNN+Transformer improves.** Step 2. It replaces a parametric filter (OU) or a fixed
basis (FFT) with a learned filter: local CNN patterns combined by attention (p. 16-20). The
allocation is an FFN, as in Fourier+FFN. The paper says "same type of network" (p. 31), but the
architectures differ in detail (p. 60, p. 65).

**Why the authors call the signal the key element** (Sec. III.G, p. 30-31; Table A.IX, p. 66;
p. 3-4, p. 11):

| Test (K = 5; SR) | What varies | Result |
|---|---|---|
| OU+Thresh vs OU+FFN: FF 0.38 vs 0.17, PCA 0.73 vs 0.68, IPCA 0.97 vs 0.66 | allocation only | a flexible allocation is similar or worse |
| IPCA: FFN on raw window 1.55, Fourier+FFN 1.90, CNN+Trans 4.16 | signal only | a better signal more than doubles SR |
| CNN+Trans: FF 3.21, PCA 3.36, IPCA 4.16 | factor model only | smaller spread than across signals |

In the paper's words, a flexible allocation "is not sufficient to compensate for an uninformative
signal" (p. 31).

**Likely examiner questions**

- *Why not predict returns and sort?* Forecast-then-sort mixes a risk premium with arbitrage and is
  not orthogonal to factors (p. 6). The paper optimizes a trading objective instead.
- *Is the split between signal and allocation identified?* Only for the classical models. Under
  joint training it is a convention (p. 10).
- *Is it really arbitrage?* No. It is a high-Sharpe, factor-neutral, self-financing bet that can
  lose (see the box above).

---

## 2. Mimicking portfolios and the factor-model decomposition

**Intuition.**

- Pairs trading compares stock A with stock B.
- Modern stat-arb compares stock *n* with a **mimicking portfolio**: a *well-diversified* portfolio
  with the **same factor exposure** (p. 2, p. 8).
- The residual portfolio is "long 1 unit of stock *n*, short its mimicking portfolio".
- A mimicking portfolio exists even when no single stock resembles *n* (p. 8).

**Math.** The conditional factor model is (Sec. II.A, p. 7)
$R_{n,t}=\beta_{n,t-1}'F_t+\epsilon_{n,t}$, with $\beta_{t-1}\in\mathbb R^{N_t\times K}$ known at
$t-1$. Factors are traded portfolios, $F_t=w^{F\prime}_{t-1}R_t$ (p. 8). Hence (eq. 1, p. 8):

$$\epsilon_t=R_t-\beta_{t-1}w^{F\prime}_{t-1}R_t=\underbrace{\big(I_{N_t}-\beta_{t-1}w^{F\prime}_{t-1}\big)}_{\Phi_{t-1}}R_t .$$

**Row *n* of $\Phi_{t-1}$ is a portfolio.**

- It holds $+1$ in stock *n* and $-m_n$, with $m_n:=w^F_{t-1}\beta_{n,t-1}$.
- The return of $m_n$ is $m_n'R_t=\beta_{n,t-1}'F_t$, exactly the factor part of stock *n*.
- So $\epsilon_{n,t}$ is a traded portfolio fixed at $t-1$ (p. 22). The PCA code violates this;
  see §6.

**Neutrality.** The paper calls residual portfolios "projections on the return space that
annihilate systematic asset risk" (p. 8). The algebra:

- If $w^{F\prime}\hat\beta=I_K$, then $\Phi\hat\beta=\hat\beta-\hat\beta(w^{F\prime}\hat\beta)=0$
  and $\Phi^2=\Phi$.
- **PCA.** The loadings are OLS of $R$ on $F=Rw^F$ over the same 60 days, so
  $\hat\beta'=(F'F)^{-1}F'R$ and $w^{F\prime}\hat\beta=F'F(F'F)^{-1}=I_K$. This is an **identity of
  the matrices used at *t***. $\Phi$ is an *oblique* projector (not symmetric).
- **IPCA.** $w^F=\beta(\beta'\beta)^{-1}$, so $\Phi=I-\beta(\beta'\beta)^{-1}\beta'$ is the
  *orthogonal* projector (`ipca.py` L605, L633).
- **Only own factors.** $\Phi$ removes exposure to the *estimated* loadings of the model's *own*
  factors. Neutrality to the true loadings, or to other economic factors, is only approximate (§3).

**Code note.** In `pca.py`, `MatrixReduced` is $\Phi'$ (L85), but the saved array is
`MatrixFull[...].T` $=\Phi$ (L100).

**From residual weights to stock weights** (eq. 3, p. 10):

$$w^R_{t-1}=\frac{\Phi'_{t-1}w^\epsilon_{t-1}}{\lVert\Phi'_{t-1}w^\epsilon_{t-1}\rVert_1},\qquad
\Phi'w^\epsilon=w^\epsilon-w^F\beta'w^\epsilon .$$

In words: hold the residual weights in the stocks, minus the aggregated hedge in the mimicking
portfolios.

**Why $\ell_1$ normalization and a Sharpe objective.**

- The Sharpe ratio is invariant to a *constant* rescaling of the strategy, but not to a
  *time-varying* one.
- Without a daily normalization, the network could lever up on "confident" days. The mean-variance
  objective would also be unbounded in scale.
- $\lVert w^R\rVert_1=1$ fixes the daily scale and bounds short positions (p. 23). This is exactly
  why $\Phi$ only reweights days (§8.1).
- The mean-variance version actually penalizes the std, not the variance (fn. 3, p. 11).
- For $K\ge1$ each residual is hedged, so the stock book is factor-neutral whatever the sign
  balance of $w^\epsilon$. The $\ell_1$ norm does not need to impose dollar-neutrality. At K = 0
  nothing hedges (§4).

**What "similar" means in each family** (p. 8):

- **PCA:** the stocks most correlated with *n*, like a clustering algorithm (fn. 2).
- **IPCA:** stocks with similar characteristics.
- **FF:** stocks with the same exposure to the Fama-French factors.

**Likely examiner questions**

- *Why is the residual tradeable?* Factors are portfolios with weights known at $t-1$, so
  $\Phi_{t-1}R_t$ is too (p. 8). The PCA caveat is in §6.
- *Why "well-diversified"?* With weights of order $1/N$, the mimicking portfolio's idiosyncratic
  *variance* is $O(1/N)$ (std $O(1/\sqrt N)$).
- *What is $\Phi$ at K = 0?* $\Phi=I$ and $\epsilon=R$: trading raw stocks.

---

## 3. Why the residual is a temporary deviation from a "fair price"

**Intuition.**

- APT: a diversified portfolio with no factor exposure cannot earn a systematic premium.
- So the mimicking portfolio is the stock's "fair price", and the residual is a deviation from it.
- The paper says APT "provides a justification for mean reversion patterns" (p. 2).

**Math / argument.**

- **APT [general knowledge, Ross 1976].** Take $R=\beta F+\epsilon$ with weakly correlated
  $\epsilon$. No asymptotic arbitrage gives $E[R_n]\approx\beta_n'\lambda$ for most *n*, so
  $E[\epsilon_n]\approx0$.
- **The paper's two expected properties** (p. 8): (i) $E[\epsilon_{n,t}]=0$; (ii) only weak
  cross-sectional dependence.
- **Empirical check** (Sec. III.F, p. 28-30; Tables A.VI-A.VII, p. 64). Equal-weighted residual
  portfolio, $K\ge3$:

  | Family | Mean return | Volatility | Other |
  |---|---|---|---|
  | PCA | ≤ 0.7% a year | 0.4-0.9% | – |
  | FF | ≤ 0.8% a year | 2.9-3.7% | – |
  | IPCA | 1.3-2.0% ($t_\mu\approx2.4$-2.9) | 2.0-2.7% | 84-89% of variance explained by FF8 |

  - For IPCA the paper says so itself: IPCA leaves "a component in the residuals that is highly
    correlated with conventional risk factors" (p. 29-30).
  - Residuals are neutral only to the factors that were removed.
  - Only *timing* the residuals earns real money, up to 50× the unconditional mean (p. 30).

**Omitted factors.**

- **The paper** (p. 9): the approach remains valid if some factors are omitted. The trades then
  exploit deviations "from the risk factors captured by the mimicking portfolios".
- **[ours]** If a factor $g$ is omitted, then $\epsilon=\gamma g+u$. Two consequences:
  - residual cross-correlation rises, so there is less diversification;
  - part of the PnL may be a premium on $g$ rather than arbitrage.
- **Table II (p. 26)** checks this only against FF8 (FF5 + momentum + short- and long-term
  reversal). A priced factor outside FF8 would go undetected.
- For CNN+Trans the FF8 $R^2$ is ≤ 2.4% for FF/PCA residuals and 3.9-9.5% for IPCA.

**Critical point: zero mean is not mean reversion [ours].**

- **Neither stated premise implies mean reversion.**
  - APT gives only $E[\epsilon]=0$.
  - The p. 10 assumption is "a stationary distribution conditioned on its lagged returns". That
    concerns residual *returns*, and it is satisfied by i.i.d. residuals, i.e. by a
    **random-walk** residual price.
- **The paper still presents APT as the justification** ("theoretical argument for mean reversion",
  p. 9). Its real support is empirical: the cumulative residuals "exhibit consistent and relatively
  regular mean-reverting behavior" (p. 23).
- **What stationarity of the level requires.** For $X=\sum\epsilon$ to be stationary, the spectral
  density of $\Delta X$ at frequency 0 must vanish:
  $$\gamma_0+2\sum_{k\ge1}\gamma_k=0,\ \text{i.e.}\ \sum_{k\ge1}\rho_k=-\tfrac12 .$$
  For an AR(1) level, $\rho_k=-\tfrac{1-b}{2}b^{k-1}$, which sums to $-\tfrac12$.
- **Negative autocorrelation is not enough.** Negative lag-1 autocorrelation alone gives only
  *partial* reversion. Example: MA(1) increments $\epsilon_t=u_t+\vartheta u_{t-1}$ with
  $-1<\vartheta<0$ give $\sum_k\rho_k=\vartheta/(1+\vartheta^2)>-\tfrac12$.

**Likely examiner questions**

- *Does no-arbitrage imply mean reversion?* No. It implies zero *unconditional* mean. Mean
  reversion is about conditional dynamics and is here an empirical regularity.
- *Is the profit a risk premium?* For CNN+Trans the FF8 alphas ≈ the means, with low $R^2$
  (Table II). But only FF8 is tested, and IPCA residuals themselves load heavily on FF8.
- *Why does IPCA win if its residuals are less "clean"?* Its $\Phi$ hedges characteristic-based
  exposure, which gives a better notion of "similar" for timing. Its CNN strategy is still
  orthogonal-ish to FF8 ($R^2$ 3.9-9.5%). The residual *level* matters less than its *dynamics*.

---

## 4. Why trade residuals rather than raw returns

**Two different mechanisms, for two kinds of models.**

- **Networks** (FFN, Fourier+FFN, CNN+Trans) share one parameter vector across all assets and a
  1,000-day window. Pooling helps only if the series are many, independent, and identically
  distributed. Raw returns fail on each count:
  - **Correlation.** About a third of return variation is explained by a latent 4-factor model
    (fn. 13, p. 25). The model therefore effectively sees "a few factor time series" (p. 25).
  - **Heterogeneity.** Risk premia differ across stocks with their characteristics. Residuals are
    "close to uncorrelated and locally cross-sectionally stationary" (p. 6) and remove the level
    component (p. 4).
- **OU** fits $\kappa,\mu,\sigma$ separately for every asset-window. Only the functional form and
  the two thresholds are shared. For OU the K = 0 penalty is the **missing hedge**: the unhedged
  book carries market risk. Table II (p. 26) gives OU K = 0 $R^2$ = 13.4% on FF8, with
  $\alpha=-4.5\%$ ($t=-1.4$).

**Evidence.**

- **Table I K = 0 (p. 24):** CNN+Trans 1.64, Fourier+FFN 0.36, OU+Thresh **−0.18**. Every $K\ge1$
  cell beats K = 0 within its model.
- **Our K = 0 check** (`OU_BASELINE.md`): SR −0.23, $\mu=-2.9\%$, $\sigma=12.7\%$ vs the paper's
  −0.18 / −2.4% / 13.3%.
  - With $\Phi=I$ there is no mapping ambiguity.
  - This is a clean test of timing, the universe filter, normalization and trade direction.
- **Hypothesis [ours, not verified].** When all stocks co-move, a rally pushes most s-scores up, so
  the OU rule goes net short the market. We have not measured the K = 0 book's net exposure. The
  moderate 13.4% FF8 $R^2$ supports this only partly.

**Likely examiner questions**

- *Why is K = 0 still 1.64 for CNN+Trans?* It learns some patterns, but 30% of its variance is
  explained by FF8 (Table II). Part of the return is factor exposure.
- *Does more K always help?* No. Performance plateaus around K ≈ 5 (p. 25). See §6.
- *How many stocks do you trade?* About 870 names pass our 30-day window filter on the official
  residuals, and about 155 are held per day. The shipped PCA K = 1 panel has 613-1,002 non-zero
  names per day (median 890), vs the paper's stated ~550 (p. 21).

---

## 5. Where mean reversion comes from, and why it is an assumption

**Economic sources.** These are [general knowledge]; the paper speaks only of "temporary
deviations" and the "law of one price" (p. 1-2, p. 50).

- **Temporary price pressure / liquidity provision.** Uninformed order flow pushes prices, and
  liquidity providers are paid when prices recover (Campbell, Grossman and Wang 1993; Nagel 2012).
- **Overreaction to firm-specific news** relative to peers (short-term reversal: Jegadeesh 1990;
  Lehmann 1990).
- **Relative mispricing.** Assets with the same risk exposure should have the same value (p. 7).
- **Microstructure.** Bid-ask bounce creates spurious negative autocorrelation in close-to-close
  returns. The paper trades at the close (p. 21) with no execution lag.

**Why it is an assumption to test, not a law.**

- **The paper claims stable profits.** It reports "non-declining profitability" and reads the
  profits as compensation "to enforce the law of one price" (p. 50). Fig. 5 shows CNN+Trans nearly
  immune to the 2007 quant quake and to 2011-12 (p. 28).
- **Our evidence questions this.**
  - On the **official** residuals, the OU paper rule earns SR 0.09-0.28 after 2010
    (`OU_BASELINE.md`).
  - On our **own** WIKI top-500 residuals (a different universe and different residuals, not
    CRSP), the CNN+Trans gross SR falls from ~5 (2003-06) to 0.3-0.5 (2015-16)
    (`full_report.md`).
- **Misspecification.** A parametric model is misspecified for trend patterns and for "multiple mean
  reversion patterns of different frequencies" (p. 14; Fig. A.2, p. 61). The learned model trades
  asymmetric **trend and reversion** patterns (p. 4, Sec. III.N).
- **Simple reversal is weak.** High-minus-low reversal on lagged residuals reaches only SR ≈ 0.3
  (Sec. III.L, p. 40-41).

**Likely examiner questions**

- *What would falsify the mean-reversion assumption?* Residual-return autocorrelations that do not
  sum to about −1/2 at the traded horizon (§3). In our synthetic study the OU edge vanishes at lag-1
  autocorrelation $\phi\approx+0.05$ and turns to losses beyond that (§8.4).
- *Why did Avellaneda-Lee find OU profitable?* An earlier sample, less competition, and
  pre-decimalization spreads, whose bid-ask bounce inflates measured close-to-close reversal. AL
  themselves report weaker results after the early 2000s [general knowledge].

---

## 6. Fama-French vs PCA vs IPCA

**Intuition.**

- **FF:** observed, economically named factors.
- **PCA:** statistical factors, the directions of maximal recent co-movement.
- **IPCA:** latent factors whose loadings are driven by characteristics.

K = 0 is the zero-factor control. The K grids and windows are listed in `paper_summary.md` §2.

**Fama-French** (p. 22; `famafrench.py`):

- Factors from the Kenneth French library.
- 60-day rolling OLS (Carhart 1997):
  - the window is $[t-60,t-1]$ (L77), so it is genuinely out of sample;
  - **no intercept** (`fit_intercept=False`, L79).
- The factors are built from a **broader universe** and treated "as tradeable assets in our
  universe" (p. 22). In code, $\Phi=[\,I\mid-\beta\,]$ acts on the stacked vector $(R_t,F_t)$
  (L88-89).
- For this reason FF is **excluded from the cost analysis**: the factor portfolios have different
  trading costs (p. 35, fn. 17).

**PCA: the paper's description** (p. 22; following Avellaneda-Lee, fn. 11). At $t-1$:

1. Estimate the correlation matrix on the last 252 days of standardized returns (population std
   $\sigma_i$).
2. Take the top-K eigenvectors $V$.
3. Form eigenportfolios $w^F=\mathrm{diag}(1/\sigma)V$, so $F_s=(R_s/\sigma)'V$.
4. Regress each stock on $F$ over the last 60 days (OLS, no intercept), giving $\hat\beta$.
5. The residual is $\epsilon_t=R_t-\hat\beta F_t$, so $\Phi=I-\hat\beta V'\mathrm{diag}(1/\sigma)$.

The universe is market cap ≥ 0.01% of the total in the previous month, with no missing return in
the window. Correlation is used rather than covariance so high-volatility names do not dominate
[general knowledge, AL 2010].

**Our finding: the shipped PCA residuals are in-sample on day *t*.**

- **Which function wrote them.** The files `residuals/pca/AvPCA_OOSresiduals_{K}_factors_...npy`
  are written only by `OOSRollingWindowPermnosVectorized` (L172).
  `OOSRollingWindowPermnos` saves that file name only when `factor == 20` (L117-118).
- **What that function does.**
  - Its correlation window is $[t-251,t]$ (L143).
  - Its 60-day loading regression runs on `res_cov_window[-60:]` (L161).
  - Both **include day *t***, and the residual is the last-row OLS residual (L163).
- **Consequence.** With hat matrix $h_{ts}=F_t'(F'F)^{-1}F_s$ over the 60-day window,
  $$\hat\epsilon_t=(1-h_{tt})u_t-\sum_{s<t}h_{ts}u_s .$$
  - $|u_t|$ is shrunk by $1-h_{tt}$, where $h_{tt}\approx K/60$ on average.
  - $\hat\epsilon_t$ loads mechanically, with weights $-h_{ts}$, on the window's past residuals.
    This can create **artificial short-horizon mean reversion**, which is exactly what the signal
    models trade.
  - $\Phi_t$ is not known at $t-1$, contrary to p. 22.
- **Scope.** If these files are the ones behind Table I, this affects the PCA rows of every model,
  including our OU replication. The **magnitude is being quantified**. We do *not* claim it
  explains our PCA5 0.93 vs the paper's 0.73.
- **FF and IPCA are clean.** FF residuals are out of sample ($[t-60,t-1]$). IPCA uses loadings from
  month-$(t-1)$ characteristics.

**IPCA** (p. 22-23; Kelly, Pruitt, Su 2019; `ipca.py`):

- Loadings are linear in 46 rank-transformed characteristics: $\beta_{n,t-1}=\Gamma'z_{n,t-1}$
  (p. 21, Table A.I).
- $\Gamma$ is re-estimated yearly on 240 months (p. 22).
- The daily factor is the cross-sectional OLS $F_t=(\beta'\beta)^{-1}\beta'R_t$, with $\beta$ from
  the previous month's characteristics (`ipca.py` L581, L605; `run_factor_model.py` L37
  `weighted=False`).
- So $\epsilon_t$ is the cross-sectional regression residual, and $\Phi$ is the orthogonal
  projector (L633).

**Trade-offs.**

| | FF | PCA | IPCA |
|---|---|---|---|
| Factors | observed | latent, from returns | latent, conditional on characteristics |
| Loadings | 60-day OLS, noisy | 60-day OLS on eigenportfolios | $\Gamma'z$, smooth |
| Interpretability | high | low (eigenvectors rotate, flip sign) | medium ($\Gamma$) |
| Data needed | factor + stock returns | stock returns only | returns + 46 characteristics |
| Tradeable within universe | no | yes | yes |
| Out of sample in shipped code | yes | **no** (day *t* in-sample) | yes |
| CNN+Trans K = 5 SR | 3.21 | 3.36 | 4.16 |

**How many factors** (Table I, p. 24-25):

- About 5 factors suffice (p. 25).
- CNN+Trans PCA peaks at K = 3 (3.56) and falls to 2.30 at K = 15. Our reading [ours]: extra
  statistical factors absorb tradable relative moves.
- IPCA CNN+Trans stays at ≈ 4.
- OU+Thresh is roughly flat for $K\ge3$ for PCA (0.62-0.87) and IPCA (0.86-0.97). For FF it jumps
  to **1.16 at K = 8**, the only OU cell above 1. That is also the cell we fail to replicate (0.04),
  see §8.

**Likely examiner questions**

- *Are the PCA residuals out of sample?* In the paper's description, yes (p. 22). In the shipped
  files, no: day *t* is in the estimation window, and the magnitude is being quantified. FF and IPCA
  residuals are out of sample.
- *Why do FF and PCA give similar Sharpe?* They explain a similar amount of co-movement, but they
  are different factors with different means (p. 25).
- *Why yearly IPCA but daily PCA?* Characteristics change at most monthly. PCA must track the
  recent correlation structure (p. 22).
- *Why is FF missing from Table IX?* Its factors are not tradeable within the universe at
  comparable cost (p. 35).

---

## 7. The Ornstein-Uhlenbeck model and the threshold rule

**Intuition.**

- The cumulative residual $X_l=\sum_{j\le l}\epsilon_{t-L-1+j}$, $l=1..L$, is a residual "price"
  (p. 12).
- Model it as a spring pulled toward $\mu$ with strength $\kappa$ and shaken by noise $\sigma$.
- If the price is far above $\mu$ relative to its typical spread, short it.

**SDE and exact discretization** (p. 12; Appendix B, p. 54-55):
$dX_t=\kappa(\mu-X_t)\,dt+\sigma\,dB_t$, with $\kappa>0$.

*Derivation.* By Itô,
$d(e^{\kappa t}X_t)=e^{\kappa t}(\kappa X_t\,dt+dX_t)=e^{\kappa t}(\kappa\mu\,dt+\sigma\,dB_t)$.
Integrating over $[t,t+\Delta t]$ gives

$$X_{t+\Delta t}=\underbrace{\mu(1-e^{-\kappa\Delta t})}_{a}+\underbrace{e^{-\kappa\Delta t}}_{b}X_t+\underbrace{\sigma\int_t^{t+\Delta t}e^{-\kappa(t+\Delta t-s)}dB_s}_{e_t},\qquad
\mathrm{Var}(e_t)=\sigma^2\!\int_0^{\Delta t}\!e^{-2\kappa u}du=\frac{\sigma^2(1-b^2)}{2\kappa}.$$

The variance follows from the Itô isometry, and $e_t$ is Gaussian and i.i.d. So the sampled OU is
**exactly** an AR(1), with $b=e^{-\kappa\Delta t}\in(0,1)$ and $\Delta t=1$ day.

**Stationary variance.** $\mathrm{Var}(X)=b^2\mathrm{Var}(X)+\mathrm{Var}(e)$, so
$\mathrm{Var}(X)=\mathrm{Var}(e)/(1-b^2)=\sigma^2/(2\kappa)$. Hence

$$\sigma_{eq}:=\sigma/\sqrt{2\kappa}=\sqrt{\mathrm{Var}(e)/(1-b^2)}.$$

**Estimators** (p. 55; `preprocess.py` L80-101). OLS on the 29 pairs $(X_l,X_{l+1})$, with
population moments:

$$\hat b=\frac{\widehat{\mathrm{Cov}}(X_l,X_{l+1})}{\widehat{\mathrm{Var}}(X_l)},\quad \hat a=\bar X_{+}-\hat b\bar X_{-},\quad
\hat\kappa=-\ln\hat b,\quad \hat\mu=\frac{\hat a}{1-\hat b},\quad \hat\sigma_{eq}=\sqrt{\frac{\widehat{\mathrm{Var}}(\hat e)}{1-\hat b^2}},\quad
R^2=\frac{\widehat{\mathrm{Cov}}^2}{\widehat{\mathrm{Var}}(X_l)\widehat{\mathrm{Var}}(X_{l+1})}.$$

**The signal.**

- The paper writes $\theta^{OU}=(\hat\kappa,\hat\mu,\hat\sigma,X_L,R^2)$ (p. 13).
- Only the ratio $\hat\sigma/\sqrt{2\hat\kappa}$ is used (fn. 5, p. 14).
- The code's 4-dim signal for OU+FFN is $(X_L,\hat\mu,\hat\sigma_{eq},R^2)$ (`preprocess.py`
  L98-101), with **no $\hat\kappa$**. The speed enters only through $R^2\approx\hat b^2=e^{-2\hat\kappa}$
  (§8.2).

**Meaning of the parameters.**

- **Half-life.** $E[X_{t+h}-\mu\mid X_t]=e^{-\kappa h}(X_t-\mu)$, so the half-life is
  $\ln2/\kappa=\ln2/(-\ln b)$ days.
- **$\mu$:** the equilibrium level.
- **$\sigma_{eq}$:** the typical distance from $\mu$, i.e. the natural unit for "far".

**s-score and the rule** (p. 13):

$$s=\frac{X_L-\hat\mu}{\hat\sigma/\sqrt{2\hat\kappa}},\qquad
w^\epsilon=\begin{cases}-1 & s>c_{thresh},\ R^2>c_{crit}\\ +1 & s<-c_{thresh},\ R^2>c_{crit}\\ 0&\text{otherwise.}\end{cases}$$

**Why short when $s>c$.**

- $E[X_{L+1}-X_L\mid X_L]=(1-b)(\mu-X_L)=-(1-b)\sigma_{eq}s$.
- The expected residual return has sign $-\mathrm{sgn}(s)$. So sell: short stock *n*, long its
  mimicking portfolio.
- **The per-bet edge is small.** One-day expected move / one-day noise std
  $=(1-b)\sigma_{eq}s/(\sigma_{eq}\sqrt{1-b^2})=s\sqrt{(1-b)/(1+b)}$. At $s=1.25$, $b=0.85$ this
  is ≈ 0.36. The strategy relies on diversifying across ~155 names.

**Why $R^2$ enters** (Yeo-Papanicolaou 2017, p. 13): a poor fit makes the s-score unreliable, so
the rule does not trade. §8.2 shows what the filter actually selects.

**Validity.**

- The paper requires $b<1$ (mean reversion, p. 55). The code adds $b>0$ (`preprocess.py` L91).
- An OU sampled at $\Delta t>0$ always has $b\in(0,1)$.
- $b\le0$: $\ln b$ is undefined.
- $b=1$: $\hat\mu$ is undefined.
- $b>1$: $\hat\mu$ is a repelling level and $\kappa<0$.

**Thresholds.**

- $c_{thresh}=1.25$ and $c_{crit}=0.25$ were chosen from $\{1,1.25,1.5\}\times\{0.25,0.5,0.75\}$
  on validation data. The paper says they coincide with the AL 2010 and YP 2017 optima (p. 55).
- Fn. 4 (p. 13): the rule is "derived by maximizing an expected trading profit". The paper gives
  no derivation.
- **Enter-at-*c*, exit-at-$\mu$ rule.** Profit per unit time is maximized at an interior $c$ of
  order 1 [general knowledge: Bertram 2010; Leung & Li 2015, cited on p. 5].
- **The paper's rule is memoryless.** It exits as soon as $|s|<c$, not at $\mu$. Each trade
  captures only part of the move, which is consistent with turnover of about 1 per day.
- **AL's original rule has hysteresis** (`OU_BASELINE.md`): open at $|s|>1.25$, close a long at
  $s>-0.50$, close a short at $s<0.75$.

**Where OU sits in eq. (4)-(6).**

- OU+Thresh is the *separate* case, eq. (6) (p. 11):
  - the signal comes from its own estimator (OLS);
  - $\mathcal W$ is a two-parameter threshold family.
- The two thresholds are **not** fitted to eq. (6). They are picked by validation Sharpe on a 3×3
  grid (p. 55).
- OU has no 1,000-day training and no rolling re-estimation: it is estimated fresh on each 30-day
  window.
- **Our out-of-sample check.** Choosing thresholds on 1998-2001 gives OOS SR 0.40 / 0.93 / 0.76
  (FF / PCA / IPCA) against validation SR of 2.6-3.4 (`OU_BASELINE.md`). Tuning does not rescue OU,
  and we did not tune on the test set.

**Why L = 30.**

- The paper: "the last 30 trading days seem to capture the relevant information" (p. 4). L = 30 is
  the main setting (p. 23).
- L = 60 gives essentially the same CNN+Trans results (Table V, Sec. III.I, p. 32). The paper
  argues this separates the signal from long-history momentum or reversal (p. 32).
- **For OU,** L = 30 means $n=29$ pairs, hence the Kendall bias of about $-(1+3b)/n$ (§8.3). AL used
  60-day windows [general knowledge].
- **Trade-off:** a longer L lowers the bias but assumes a constant $\kappa$ for longer and adapts
  more slowly.

**Likely examiner questions**

- *Why model the cumulative residual?* The OU describes a reverting *level*. Its increments (the
  returns) are then negatively autocorrelated, with the autocorrelations summing to −1/2 (§3).
- *What does $\hat\kappa=0.2$ mean?* Half-life $\ln2/0.2\approx3.5$ trading days.
- *Why divide by $\sigma/\sqrt{2\kappa}$ and not $\sigma$?* The equilibrium spread depends on both
  noise and pull. A fast reverter stays closer to $\mu$.
- *Is the AR(1) an approximation?* No. It is the exact discretization. Only the estimation is
  approximate.

---

## 8. What our empirical work adds to the theory (`OU_BASELINE.md`)

### 8.1 $\Phi$ only rescales daily leverage

- We trade $w^\epsilon$ with $\lVert w^\epsilon\rVert_1=1$. The paper trades
  $w^R=\Phi'w^\epsilon/\lVert\Phi'w^\epsilon\rVert_1$, whose daily return is
  $$w^{R\prime}R_t=\frac{w^{\epsilon\prime}\Phi_{t-1}R_t}{\lVert\Phi'_{t-1}w^\epsilon\rVert_1}=\frac{w^{\epsilon\prime}\epsilon_t}{\lVert\Phi'_{t-1}w^\epsilon\rVert_1}.$$
- Ours is $w^{\epsilon\prime}\epsilon_t$. The factor $1/\lVert\Phi'w^\epsilon\rVert_1>0$ is known at
  $t-1$. So the sign of every day's PnL is the same, and only the day's leverage differs.
- **Why the numbers still differ.** $\Phi$ is not published with the residuals, so the day
  weighting and the Sharpe ratio cannot be matched exactly. K = 0 ($\Phi=I$) matches: −0.23 vs
  −0.18.
- **K = 5:** FF 0.34 / PCA 0.93 / IPCA 0.64 vs the paper's 0.38 / 0.73 / 0.97.
  - Our IPCA-PCA gap is −0.29 (block-bootstrap SE 0.28). The paper's gap is +0.24.
  - The gap-vs-gap difference of −0.53 is about 1.9 of our SEs, so it is borderline, not clearly
    noise.
  - Excluding 2008-09 restores the paper's ranking: IPCA 0.60 > PCA 0.51 > FF 0.40.

### 8.2 $R^2\approx\hat b^2$: the $R^2$ filter is a minimum-half-life filter

- From the estimators,
  $R^2=\hat b^2\,\widehat{\mathrm{Var}}(X_l)/\widehat{\mathrm{Var}}(X_{l+1})\approx\hat b^2$,
  since the two sub-windows differ by one point.
- So $R^2>c_{crit}$ roughly means $\hat b>\sqrt{c_{crit}}$:

| $c_{crit}$ | min $\hat b$ | min half-life |
|---|---|---|
| 0.25 | 0.50 | 1.0 day |
| 0.50 | 0.71 | 2.0 days |
| 0.75 | 0.87 | 4.8 days |

- A high $R^2$ selects slow, near-unit-root windows and discards the fastest reversion.
- **Evidence.**
  - Simulation: correlation between $R^2$ and $\hat b^2$ is 0.93-0.98.
  - Real data: every $c_{crit}=0.75$ cell loses money.
- **Our best variant *net of costs*** is AL with a $\kappa$ filter and no $R^2$ filter (PCA 1.06,
  IPCA 0.79 gross).
- **The best *gross* cell** is the paper rule at 1.0 / 0.25 (PCA 1.28).

### 8.3 Small-sample (Kendall) bias of $\hat b$ with L = 30

- OLS on 29 AR(1) pairs is biased down.
  - Stationary case: $E[\hat b]-b\approx-(1+3b)/n$ (Kendall 1954; Marriott-Pope 1954), about
    −0.14 near $b=1$.
  - Unit root with intercept: $E[n(\hat b-1)]\approx-5.4$ (Dickey-Fuller 1979; Fuller 1976), about
    −0.19.
- So a **random walk** gives $\hat b\approx0.81$-$0.86$, an apparent half-life of about 3-5 days.
  - Our simulation: median 4.3 days, and 18% of windows trade.
  - A true half-life of 20 days is estimated at 3.8 days.
- The downward bias in $\hat b$ also shrinks $\hat\sigma_{eq}$, which inflates $|s|$.
- Under a random walk, $\hat\mu\approx\bar X_-+\bar d/(1-\hat b)$ stays near the window mean. The rule
  becomes "fade the deviation from the 30-day mean".
- **One-liner.** Real residuals have lag-1 autocorrelation ≈ −0.02. If that came from an OU,
  $-(1-b)/2=-0.02$ gives $b\approx0.96$ and a half-life of about 17 days. The 30-day fit reports
  about 3-4 days.

### 8.4 The rule is a contrarian bet on negative return autocorrelation

- It profits only if residual returns revert. In simulation the edge vanishes at momentum
  $\phi\approx+0.05$, and $\phi=0.3$ gives SR ≈ −9.
- **The paper's example** (Fig. A.2, p. 14, p. 61): no long position during the uptrend, then stuck
  short through a mean-reversion phase.
- **Ours** (`OU_BASELINE.md` illustration): it shorts the start of an uptrend, stays short while $s$
  explodes, then buys the top (−11%).

### 8.5 Trends: with or against? It depends on the noise

- **Algebra.** Let $\bar d=(X_L-X_1)/(L-1)$. Since $\bar X_+-\bar X_-=\bar d$,
  $$\hat a=(1-\hat b)\bar X_-+\bar d\quad\Rightarrow\quad \hat\mu=\bar X_-+\frac{\bar d}{1-\hat b}.$$
- **When $\hat\mu$ lands ahead of $X_L$.** For an up-trend, $\hat\mu>X_L$ iff
  $1-\hat b<\bar d/(X_L-\bar X_-)$. For a clean linear trend, $X_L-\bar X_-\approx\bar dL/2$, so this
  needs $\hat b\gtrsim1-2/L\approx0.93$.
- **A noiseless trend** gives $\hat b=1$ exactly, which is masked: no trade.
- **Simulation** [ours, `scripts/ou_trend_sim.py`]: 1,500 assets × 300 windows, paper rule.
  "d" is the trend per day; "with" and "against" are the shares of windows with a position in each
  direction relative to the trend.

  | Level = trend $d\,t$ + | d / day | $\hat\mu$ ahead | with / against | PnL bp/day (trend part) |
  |---|---|---|---|---|
  | white noise 2% | 0.25% | 4% | 0% / 23% | +37 (−6) |
  | white noise 2% | 0.5% | 1% | 0% / 18% | +22 (−9) |
  | white noise 2% | 1.0% | 75% | 24% / 0% | −6 (+24) |
  | white noise 1% | 0.25% | 1% | 0% / 18% | +11 (−5) |
  | white noise 1% | 1.0% | 100% | 90% / 0% | +101 (+90) |
  | random walk 2% | 0.5% | 57% | 18% / 6% | +6 (+6) |
  | OU (hl 5, sd 2%) + RW 2% (our design) | 0.5% | 53% | 16% / 6% | +4 (+5) |
  | OU (hl 5, sd 2%) + RW 2% (our design) | 1.0% | 78% | 37% / 2% | +35 (+35) |

- **How to read it.**
  - **Stationary noise with a moderate trend** (roughly $d/\sigma_{noise}\le0.25$): $\hat b$ is well
    below 0.93, $\hat\mu$ sits near the window mean, and $X_L$ sits above it. The rule **shorts
    against the trend**. It still earns money here only because a white-noise level reverts
    completely.
  - **Trend dominating stationary noise** ($d/\sigma_{noise}\gtrsim0.5$): the rule goes long, with
    the trend.
  - **Persistent (random-walk) noise:** $\hat b$ is already near 1. The rule is net *with* the trend
    (about 3:1), and its PnL is almost exactly the drift it harvests.
- **Correction to `OU_BASELINE.md` point 4.** "The rule trades with the trend" holds for our
  synthetic design (trend + OU + random walk). It is *not* a general property.

### 8.6 Costs kill OU

- **Cost model** (p. 35): $0.0005\lVert w^R_{t-1}-w^R_{t-2}\rVert_1+0.0001\lVert\min(w^R_{t-1},0)\rVert_1$,
  charged in the paper on *stock* weights, in training and evaluation.
- **Turnover.** With $\lVert w\rVert_1=1$, turnover ranges over [0, 2]. Ours is 0.97: about half
  the gross book is replaced every day.
- **Drag vs gross.** The cost drag is $0.97\times5\,\text{bp}\times252\approx12\%$ a year, plus
  about 1% for the short leg. Gross $\mu$ is a few percent (IPCA5: 2.6% ours, 3.8% paper).
- **Net SR:**
  - paper rule: −2.7 to −4.2;
  - best AL + $\kappa$-filter variant: −0.74 (IPCA) to −1.66 (FF).
- Break-even cost is ≤ ~2 bp.
- **Caveats.**
  - Our costs are charged on residual weights.
  - OU's fixed rule cannot adapt to costs.
  - The paper reports no OU net figure. Its CNN+Trans net SR of 1.11 (Table IX) comes from a
    cost-*trained* model, so the comparison is not like for like.

**Likely examiner questions**

- *Is OU's positive gross SR real?*
  - Ours: IPCA5 Newey-West $t=2.68$. As a consistency check, $SR\sqrt{15}=0.64\times3.9\approx2.5$.
  - Paper, same cell: $\alpha=2.8\%$, $t_\alpha=3.0$, $R^2=18\%$ on FF8 (Table II).
  - FF5 and PCA5 OU alphas are not significant at 1% ($t=0.9$, $2.4$).
  - The paper itself says parametric strategies are "largely explained by conventional risk
    factors" (p. 28), though its own PCA K = 8 cell has $t_\alpha=3.0$.
  - Net of costs, every variant is negative.
- *So is the $R^2$ filter useless?* It is a speed filter in disguise, and it works the wrong way at
  high cutoffs. A direct $\kappa$ filter works better.
- *Why FF8 0.04 vs the paper's 1.16?* An untested hypothesis: FF8 contains a short-term reversal
  factor, and hedging it removes the reversion the rule trades.

---

## 9. Cheat sheet

1. **Three steps:** residual portfolios → signal → allocation. The paper improves the signal
   (jointly trained with the allocation). OU uses eq. (6) with separate estimation (p. 11).
2. **The signal is the key element.** OU+FFN is similar to or worse than OU+Thresh. OU → FFT →
   CNN roughly doubles SR at each step (Table A.IX, Table I).
3. **Residual:** $\Phi=I-\beta w^{F\prime}$. Row *n* = stock *n* minus the mimicking portfolio.
   $\Phi\hat\beta=0$ when $w^{F\prime}\hat\beta=I$ (PCA: oblique; IPCA: orthogonal projector).
4. **$\Phi$ and leverage:**
   $w^{R\prime}R_t=w^{\epsilon\prime}\epsilon_t/\lVert\Phi'w^\epsilon\rVert_1$. $\Phi$ only
   reweights days, which matters because the $\ell_1$ scale is fixed daily.
5. **APT gives $E[\epsilon]=0$, not mean reversion.** Level stationarity needs return
   autocorrelations summing to −1/2. The paper's support is empirical (p. 23), though presented as
   APT-motivated (p. 2, 9).
6. **The shipped PCA residuals include day *t*** in the loadings, so they are in-sample (magnitude
   being quantified). FF ($[t-60,t-1]$) and IPCA are out of sample.
7. **OU as AR(1):** $b=e^{-\kappa}$, $a=\mu(1-b)$, $\mathrm{Var}(e)=\sigma^2(1-b^2)/(2\kappa)$,
   $\sigma_{eq}=\sqrt{\mathrm{Var}(e)/(1-b^2)}$, half-life $\ln2/\kappa$.
8. **Rule:** $E[\Delta X]=-(1-b)\sigma_{eq}s$, so short at $s>1.25$ and long at $s<-1.25$, if
   $R^2>0.25$ and $0<b<1$.
9. **Why OU fails:**
   - $R^2\approx\hat b^2$ is a minimum-half-life filter;
   - Kendall bias makes a random walk look like a 4-day reverter;
   - it is a contrarian bet that dies at $\phi\approx+0.05$;
   - trends are faded under stationary noise and followed under random-walk noise.
10. **Costs:** turnover ≈ 1 per day means ~12% a year at 5 bp vs a few % gross, so OU is negative net
    in every variant.

---

### Open points

- **PCA look-ahead magnitude.** Not yet quantified. Nor is its effect on the paper's PCA rows.
- **Typo in $\theta^{OU}$.** The p. 13 formula writes $X_L$ as $\sum_{l=1}^L\epsilon_{n,t-1+l}$.
  From the $\mathrm{Int}(\cdot)$ definition (p. 12) it should be $\epsilon_{n,t-L-1+l}$, as in
  `preprocess.py` L78.

### Review responses (rejected or partly applied findings)

- **m14 (partly).** Applied "similar or worse". The K = 0 comparison was left out of the K = 5 table.
- **m17 (drop the §1 table).** Kept it: it is the direct evidence for the key-element claim. Instead
  the §6 protocol details were cut and the cheat sheet shortened to 10 items.
- All other BLOCKER, MAJOR and MINOR findings were verified against `paper.txt` and the code and
  applied.
