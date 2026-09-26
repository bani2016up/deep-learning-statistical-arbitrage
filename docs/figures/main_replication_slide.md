# Main Replication Result

**Metric on the slide:** gross annualized Sharpe ratio (SR), calculated from out-of-sample daily returns **before transaction costs**. Higher SR means better risk-adjusted performance. Our OOS period: February 2003 to December 2016. Our CNN+Transformer values are the **mean SR of two separate seeds**; the other model values use one seed. Paper values are the reported Table I benchmarks. The square-bracketed ensemble results in the source report are **not** plotted here.

| Model | Our FF5 | Paper FF5 | Our PCA5 | Paper PCA5 |
| :-- | --: | --: | --: | --: |
| CNN+Transformer | **1.95** | **3.21** | **2.66** | **3.36** |
| Fourier+FFN | 0.94 | 1.66 | 1.32 | 1.98 |
| OU+Threshold | 0.15 | 0.38 | 0.62 | 0.73 |

## Slide text (English)

- **Ranking reproduced:** CNN+Transformer > Fourier+FFN > OU+Threshold for both FF5 and PCA5.
- **Lower absolute performance:** our CNN+Transformer SR is 1.95 vs 3.21 (paper) on FF5, and 2.66 vs 3.36 on PCA5.
- **PCA5 beats FF5:** our CNN+Transformer SR increases from 1.95 to 2.66; the same direction holds for all three models.

**Speaker note:** This is a *gross* signal replication, not evidence of profitability after trading costs. Reversal is omitted because there is no paper benchmark for it in the comparison table. The two-seed ensemble SR is 2.12 (FF5) and 2.84 (PCA5), but those figures are not the mean-of-seeds figures shown on the slide.

Source: `docs/full_report.md`, section 1, and `results/report_tables.md` (`full_paper` and `ensemble`). Visual: `docs/figures/main_replication_slide.svg` (1600 x 900, 16:9).
