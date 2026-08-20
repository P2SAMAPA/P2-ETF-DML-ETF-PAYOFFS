# P2-DML-ETF-PAYOFFS

## Differential Machine Learning for Discontinuous ETF Payoffs & Liquidity Barriers

### Concept

This repository implements **Differential Machine Learning** for financial applications, specifically handling:
- Discontinuous payoff functions (barriers, knockouts)
- Execution cost discontinuities (bid-ask spread, price impact)
- Liquidity barriers

Instead of standard automatic differentiation through numerical approximations, the model uses **analytical derivatives** of execution costs and price impact directly in the training loss.

### Architecture
