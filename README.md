# 2 the MOON 🚀 — KN Hack 2026

**Estrategia:** Dual Momentum ETF Rotation  
**Señal:** mom_12m (Jegadeesh & Titman 1993)  
**Filtro:** EMA 200  
**Selección:** TOP-1 concentrado, siempre invertido  
**Universo:** 13 ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, DBC, XLK, XLE, XLF, SMH)

---

## Resultados KaxaNuk (Mar 2007 – Dic 2025)

| Métrica | 2 the MOON | SPY |
|---------|-----------|-----|
| CAGR | **15.26%** | 10.33% |
| Alpha | **+4.92%** | — |
| Max Drawdown | **-41.27%** | -55.18% |
| Sharpe | **0.6056** | 0.5132 |
| Sortino | **0.8036** | 0.6888 |
| $1M → | **$16.3M** | $6.9M |

---

## Cómo correrlo

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Correr el backtest (genera portfolio_weights.csv y gráficas)
python dual_momentum_backtest.py
```

Los datos de cada ETF están en `Output/<TICKER>.csv` (provistos por KaxaNuk Data Curator).  
El archivo `portfolio_weights.csv` es el que se sube a la plataforma KaxaNuk para el backtest oficial.

---

## Archivos principales

| Archivo | Descripción |
|---------|-------------|
| `dual_momentum_backtest.py` | Código principal de la estrategia |
| `portfolio_weights.csv` | Pesos mensuales (228 rebalanceos, 2007–2025) |
| `Output/<ETF>.csv` | Datos históricos por ETF (KaxaNuk) |
| `Output/dm_dashboard.png` | Dashboard de resultados |
| `Output/dm_analytics.png` | Analytics profundos |

---

## Equipo

KN Hack 2026 — Joserhamón Silva
