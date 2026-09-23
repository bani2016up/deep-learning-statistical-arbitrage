# Limitations

- The free 50-stock public universe is not the CRSP universe.
- A present-day liquid-stock list creates survivorship and selection bias.
- Compustat/IPCA and the paper's 46 firm characteristics are absent.
- Returns are not reduced by the risk-free rate.
- Rolling PCA is a provisional approximation, not the final factor-model dataset.
- Residuals are traded directly instead of mapped to stocks and normalized through `Phi`.
- One chronological split replaces the paper's rolling 1,000-day/125-day schedule.
- Transaction, financing, borrow, and holding costs are not modeled.
- Corporate-action quality is delegated to the free adjusted-price source.
- Missing observations are handled with a rectangular panel, unlike a dynamic CRSP universe.
- Hyperparameters and one fixed seed were used without tuning or seed selection.
- Public-data metrics are engineering smoke tests, not a scientific reproduction or a claim
  of deployable financial performance.
