# P2-DML-ETF-PAYOFFS

## Differential Machine Learning for Discontinuous ETF Payoffs & Liquidity Barriers

### Concept

This repository implements **Differential Machine Learning** for financial applications, specifically handling:
- Discontinuous payoff functions (barriers, knockouts)
- Execution cost discontinuities (bid-ask spread, price impact)
- Liquidity barriers

Instead of standard automatic differentiation through numerical approximations, the model uses **analytical derivatives** of execution costs and price impact directly in the training loss.

### Architecture
Market Data → Feature Engineering → Differential Layers → Payoff Module → Differential Loss
↓
Analytical Derivatives
(No numerical smoothing)

text

### Key Features

1. **Discontinuity-Aware Layers**: Custom layers that handle tick size barriers with analytical derivatives
2. **Execution Cost Models**: Square-root impact, Almgren-Chriss, and liquidity barriers
3. **Barrier Payoff Functions**: Analytical derivatives for barrier and knockout options
4. **Differential Loss**: Loss function that penalizes high execution cost sensitivity
