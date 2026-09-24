# Research directions

Limitations of _Deep Learning Statistical Arbitrage_ (Guijarro-Ordonez, Pelger, Zanotti, 2022) and possible extensions.

---

## 1. Framework and limitations

The baseline framework decomposes statistical arbitrage into three modular stages:

1. **Arbitrage Portfolio Formation**: Constructing residual returns $r_{i,t}^\epsilon = r_{i,t} - \sum_{k=1}^K \beta_{i,k,t} f_{k,t}$ relative to systematic asset pricing factors (PCA, IPCA, Fama-French).
2. **Temporal Signal Extraction**: Processing cumulative 30-day residual windows $S_{i,t} = \sum_{\tau=t-30}^{t-1} r_{i,\tau}^\epsilon$ using a causal 1D CNN followed by a temporal Transformer encoder.
3. **Portfolio Allocation & Optimization**: Mapping learned representations into gross-exposure-constrained portfolio weights $w_t$ that maximize the out-of-sample Sharpe ratio end-to-end.

Simplifications in the original design:

- **Independent Asset Processing**: The temporal network processes each asset's residual history independently; cross-asset lead-lag relationships, sector clustering, and supply-chain dependencies are discarded.
- **Linear Factor Orthogonalization**: PCA and IPCA assume linear or bilinear exposure structures, failing to capture regime-dependent or nonlinear systematic risk.
- **Simple friction model**: the paper's cost model is linear (5 bp per unit turnover, 1 bp per unit short) and is used only in a separate experiment (§III.J). Our runs show that daily turnover of 70–110% makes net Sharpe strongly negative without costs in the objective.
- **Fixed Lookback Window (30 Days)**: Mean reversion manifests across multiple timescales—from high-frequency intraday microstructure bounces (hours) to intermediate capital reallocation cycles (60–120 days).

Five directions follow.

---

## 2. Dimension 1: Nonlinear & Dynamic Latent Factor Models

### 2.1 Deep Conditional Autoencoders (Deep IPCA)

- **Concept**: Replace linear rolling PCA/IPCA with a Conditional Autoencoder (CAE) architecture (inspired by Gu, Kelly, Xiu, 2021).
- **Formulation**:
  $$r_{i,t} = \beta(z_{i,t-1})^\top f_t + \epsilon_{i,t}$$
  where the factor loading $\beta(z_{i,t-1}) \in \mathbb{R}^K$ is parameterized by a deep neural network taking firm characteristics $z_{i,t-1}$ (momentum, size, book-to-market, volatility, liquidity), and factor returns $f_t$ are extracted via a linear projection of the cross-section $f_t = (B_t^\top B_t)^{-1} B_t^\top r_t$.
- **Advantage**: Residuals $\epsilon_{i,t}$ are strictly orthogonal to both linear and complex interaction effects of firm characteristics, preventing the temporal arbitrageur from merely rediscovering known equity risk premia.

### 2.2 Kalman Variational Autoencoders (Kalman VAE)

- **Concept**: Model latent factor states probabilistically with explicit continuous-time transition dynamics.
- **Formulation**:
  $$z_t \sim \mathcal{N}(A z_{t-1}, Q), \quad r_t \sim \mathcal{N}(g_\theta(z_t), \Sigma_\epsilon)$$
- **Advantage**: Captures time-varying factor volatility (heteroskedasticity) and sudden macroeconomic regime shifts (e.g., market crashes, liquidity dry-ups) without requiring arbitrary rolling window lookbacks.

### 2.3 Robust Low-Rank plus Sparse Matrix Decomposition

- **Concept**: Formulate the return matrix $R \in \mathbb{R}^{T \times N}$ via robust principal component pursuit:
  $$\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad R = L + S$$
- **Advantage**: Automatically isolates idiosyncratic outliers, earnings surprises, and structural breaks into the sparse matrix $S$, producing stable, clean low-rank systematic factors $L$ and robust residual series.

---

## 3. Dimension 2: Next-Generation Sequence & Spatio-Temporal Architectures

### 3.1 Spatio-Temporal Graph Neural Networks (Graph Transformers)

- **Limitation of Baseline**: Treats each stock's residual series in complete isolation with weight sharing.
- **Proposed Solution**: Model the universe as a dynamic graph $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$, where nodes represent assets and edges represent structural/statistical affinity:
  - _Statistical Edges_: Rolling residual correlation matrix $\rho_{i,j,t}$.
  - _Fundamental Edges_: Supply-chain links (customer-supplier), shared SEC 10-K industry codes, or analyst co-coverage.
- **Architecture**:
  1. _Temporal Stage_: Causal 1D CNN or State-Space layer per node to extract intra-asset temporal momentum/reversal.
  2. _Spatial Stage_: Multi-Head Graph Attention (GATv2) layer:
     $$h_{i,t}^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij,t} W_V h_{j,t}^{(l)} \right)$$
- **Impact**: Enables the model to exploit **lead-lag cross-asset arbitrage** (e.g., downstream supplier lagging upstream customer announcement by 1–3 days).

### 3.2 Selective State Space Models (Mamba / S6)

- **Limitation of Baseline**: Self-attention in Transformers has quadratic compute complexity $O(L^2)$ and lacks inductive bias for exponentially decaying temporal signals.
- **Proposed Solution**: Replace the Transformer encoder with a bidirectional or causal Mamba (Selective State Space) layer:
  $$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t$$
- **Advantage**:
  - $O(L)$ linear scaling with window length $L$, enabling lookback windows of 90, 180, or 252 days.
  - Time-varying selection mechanism allows the network to dynamically filter out high-frequency noise while remembering long-horizon mean-reverting equilibrium levels.

### 3.3 Temporal Fusion Transformer (TFT) with Macro Conditioning

- **Concept**: Incorporate market-wide context into the asset-level arbitrage decision:
  - _Macro Inputs_: VIX, Treasury yield curve 10Y-2Y slope, credit spread (HY OAS), aggregate cross-sectional dispersion.
  - _Asset Residuals_: 30-day cumulative residual series.
- **Mechanism**: Gated Residual Networks (GRNs) modulate signal confidence based on market volatility regimes (e.g., scaling down bets when market-wide dispersion explodes).

---

## 4. Dimension 3: Friction-Aware & Risk-Constrained Objectives

### 4.1 Differentiable Turnover-Penalized Sharpe Loss

- **Baseline Objective**:
  $$\mathcal{L}_{\text{gross}}(\theta) = - \frac{\hat{\mu}_r(\theta)}{\hat{\sigma}_r(\theta) + \epsilon}$$
- **Friction-Aware Objective**:
  Directly optimize net return after transaction costs:
  $$\hat{r}_t^{\text{net}}(\theta) = \sum_{i=1}^N w_{i,t}(\theta) r_{i,t}^\epsilon - \sum_{i=1}^N \left( c_{\text{prop}} |\Delta w_{i,t}(\theta)| + c_{\text{quad}} \frac{|\Delta w_{i,t}(\theta)|^2}{\text{ADV}_{i,t}} + c_{\text{borrow}} \max(-w_{i,t}(\theta), 0) \right)$$
  $$\mathcal{L}_{\text{net}}(\theta) = - \frac{\mathbb{E}[\hat{r}_t^{\text{net}}]}{\sqrt{\text{Var}(\hat{r}_t^{\text{net}}) + \epsilon}} + \lambda_{\text{turnover}} \frac{1}{T} \sum_{t=1}^T \|\Delta w_t(\theta)\|_1$$
- **Advantage**: Prevents the optimizer from generating microscopic edge trades that generate high gross Sharpe but collapse under real broker commission and bid-ask spreads.

### 4.2 Explicit Portfolio Constraints via Differentiable Optimization Layers

- Instead of simple $L_1$ normalization ($w_t = \frac{s_t}{\|s_t\|_1}$), use a differentiable convex layer (`CvxpyLayers` or `OptNet`):
  $$\min_w \quad \frac{1}{2} \|w - s_t\|_2^2$$
  $$\text{s.t.} \quad \sum_{i=1}^N w_i = 0 \quad (\text{Dollar Neutrality})$$
  $$\beta_{\text{mkt}}^\top w = 0 \quad (\text{Beta Neutrality})$$
  $$\sum_{i \in \text{Sector}_k} w_i = 0 \quad (\text{Sector Neutrality})$$
  $$\|w\|_1 \le 1, \quad |w_i| \le w_{\max}$$

### 4.3 Differentiable Sortino & Downside Risk Objectives

- Penalize only downside volatility rather than symmetric variance:
  $$\text{SemiDev} = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, r_t - \tau)^2 + \epsilon}$$
  $$\mathcal{L}_{\text{Sortino}}(\theta) = - \frac{\mathbb{E}[r_t] - \tau}{\text{SemiDev} + \epsilon}$$

---

## 5. Dimension 4: Multi-Horizon Execution & Market Microstructure

### 5.1 Multi-Task Multi-Horizon Signal Heads

- Rather than optimizing solely for next-day residual realization $t+1$, train a multi-task head predicting multiple forward horizons:
  - 1-day ahead residual return: $r_{i, t+1}^\epsilon$ (microstructure reversal)
  - 5-day cumulative residual return: $\sum_{k=1}^5 r_{i, t+k}^\epsilon$ (swing arbitrage)
  - 20-day cumulative residual return: $\sum_{k=1}^{20} r_{i, t+k}^\epsilon$ (fundamental convergence)
- Combines the outputs via uncertainty weighting (Kendall et al.) to establish a multi-speed holding portfolio.

### 5.2 Dynamic No-Trade Bands (Hysteresis Execution)

- To slash turnover without sacrificing signal freshness, deploy dynamic buffer zones around target weights:
  $$w_{i,t}^{\text{exec}} = \begin{cases} w_{i,t}^{\text{target}} & \text{if } |w_{i,t}^{\text{target}} - w_{i,t-1}^{\text{exec}}| > \delta_{i,t} \\ w_{i,t-1}^{\text{exec}} & \text{otherwise} \end{cases}$$
  where threshold $\delta_{i,t} \propto \frac{\text{Spread}_{i,t}}{\hat{\alpha}_{i,t}}$.

---

## 6. Dimension 5: Cross-Asset & High-Frequency Domains

### 6.1 Intraday Statistical Arbitrage (1-Minute / 5-Minute Bars)

- Apply the causal CNN + Transformer architecture to intraday ETF basket residuals (e.g., S&P 500 constituents vs. SPY / sector ETFs).
- Incorporate order flow imbalance (OFI), book pressure, and volume-weighted average price (VWAP) deviations as auxiliary input channels alongside price residuals.

### 6.2 Cryptocurrency Perpetual Futures & Funding Arbitrage

- **Characteristics**: 24/7 continuous trading, massive cross-sectional retail noise, high retail leverage.
- **Model Opportunity**: Extract statistical factors across top 100 perpetual swaps; trade residual basis while delta-hedging or funding-rate hedging against extreme spikes.

---

## 7. Prioritized Research Roadmap

| Priority | Initiative                                | Complexity | Expected Alpha Impact            | Target Metric                   |
| -------- | ----------------------------------------- | ---------- | -------------------------------- | ------------------------------- |
| **P1**   | **Friction-Aware Loss Function**          | Low        | Very High (survives costs)       | Net Sharpe @ 10 bps > 1.0       |
| **P1**   | **Turnover Penalty & L1 Damping**         | Low        | High (halves turnover)           | Daily turnover < 0.40           |
| **P2**   | **Multi-Horizon Lookback (60d & 90d)**    | Medium     | Moderate                         | Sharpe stability across regimes |
| **P2**   | **Mamba / State-Space Temporal Backbone** | Medium     | High (computational & long-term) | Better drawdown recovery        |
| **P3**   | **Dynamic Factor Models / Deep IPCA**     | High       | High (truer residuals)           | Lower factor correlation        |
| **P3**   | **Cross-Asset Graph Attention (GNN)**     | High       | Very High (lead-lag alpha)       | Stat-arb alpha $t$-stat > 3.0   |

---

## 8. Summary & Next Steps for Team

1. Run benchmark comparisons across baseline models (Reversal vs. Raw FFN vs. CNN-Transformer).
2. Integrate the net-of-costs Sharpe optimization into training.
3. Test lookback window scaling ($L=15, 30, 60, 90$) using the comparison framework.
