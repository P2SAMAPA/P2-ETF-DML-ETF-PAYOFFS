# P2-DML-ETF-PAYOFFS

## Differential ML for ETF Payoffs & Liquidity

### Concept

Differential Machine Learning for discontinuous ETF payoffs and execution costs.

### Architecture

```text
Market Data
  → Feature Engineering
  → Differential Layers
  → Payoff Module
  → Differential Loss
  ↓
Analytical Derivatives (No numerical smoothing)
```

### Key Features

1. **Discontinuity-Aware Layers**: Custom layers that handle tick size barriers with analytical derivatives
2. **Execution Cost Models**: Square-root impact, Almgren-Chriss, and liquidity barriers
3. **Barrier Payoff Functions**: Analytical derivatives for barrier and knockout options
4. **Differential Loss**: Loss function that penalizes high execution cost sensitivity

### Installation

```bash
git clone https://github.com/P2SAMAPA/P2-DML-ETF-PAYOFFS
cd P2-DML-ETF-PAYOFFS
pip install -r requirements.txt
```
