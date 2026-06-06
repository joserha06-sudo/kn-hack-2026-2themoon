"""
2 the MOON — KN Hack 2026
Dual Momentum | mom_12m + EMA200 | TOP-1 Concentrado | Siempre Invertido

Lógica (primer día hábil de cada mes):
  1. Filtro EMA200   → precio > EMA 200
  2. Ranking         → mom_12m (retorno 12m saltando el último mes)
  3. TOP-1           → 100% al ETF de mayor momentum que pase EMA200
  4. Fallback        → si ninguno pasa EMA200 → mejor momentum absoluto

Universo B (13 ETFs): SPY QQQ IWM EFA EEM TLT GLD VNQ DBC XLK XLE XLF SMH
Resultados KaxaNuk (Mar 2007 – Dic 2025):
  CAGR=15.26%  Alpha=+4.92%  Sharpe=0.6056  MaxDD=-41.27%  $1M→$16.3M
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("Output")
START_DATE   = "2017-01-01"
END_DATE     = "2025-12-31"
INITIAL_CAP  = 100.0
TXN_COST_BPS = 10

# Universo B — 13 ETFs multi-asset + sectoriales (estrategia oficial 2 the MOON)
ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "DBC", "XLK", "XLE", "XLF", "SMH"]
REFUGIO = "SHY"
TOP_N    = 1

# Señal: mom_12m puro (Jegadeesh & Titman 1993)
W_MOM12  = 1.0
W_MOM3   = 0.0

# Sin vol scaling — siempre 100% al ganador
TARGET_VOL = None

# Columnas CSV
COL_CLOSE  = "m_close_dividend_and_split_adjusted"
COL_LOGRET = "c_log_returns_dividend_and_split_adjusted"
COL_MOM12  = "c_momentum_12_1"

CATEGORY_COLORS = {
    "QQQ": "#2e75b6", "SMH": "#117a65", "XLK": "#8e44ad",
    "XLE": "#922b21", "XLF": "#1a5276", "MDY": "#5ba3d9",
    "IWM": "#0d3d6e", "GLD": "#c79c00", "TLT": "#1e7145",
    "DBC": "#c0392b", "VUG": "#7e5ca6", "SPY": "#1f4e79",
    "SHY": "#7bc4a0",
}


# ─────────────────────────────────────────────────────────────
# PASO 1: CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
def cargar_datos() -> dict[str, pd.DataFrame]:
    datos: dict[str, pd.DataFrame] = {}
    tickers_necesarios = ETFS + [REFUGIO]

    for ticker in tickers_necesarios:
        path = OUTPUT_DIR / f"{ticker}.csv"
        if not path.exists():
            print(f"  ADVERTENCIA: {path} no encontrado — se omite {ticker}")
            continue
        df = pd.read_csv(path)
        mask_bad = df["m_date"].apply(lambda x: pd.to_datetime(x, errors="coerce")).isna()
        df = df[~mask_bad].copy()
        df["m_date"] = pd.to_datetime(df["m_date"])
        df = df.set_index("m_date").sort_index()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        datos[ticker] = df

    if REFUGIO not in datos:
        raise SystemExit(f"ERROR: falta {REFUGIO}.csv — es indispensable.")

    for ticker, df in datos.items():
        close = df.get(COL_CLOSE)
        if close is None:
            continue

        # EMA200 (legacy, mantenemos por referencia)
        datos[ticker]["ema_200"] = close.ewm(span=200, adjust=False).mean()

        # SMA200 — usado por dual regime filter
        datos[ticker]["sma_200"] = close.rolling(200).mean()

        # Log-returns
        if COL_LOGRET not in df.columns:
            datos[ticker][COL_LOGRET] = np.log(close / close.shift(1))

        # mom_12m — skip último mes
        if COL_MOM12 not in df.columns:
            datos[ticker][COL_MOM12] = close.shift(21) / close.shift(252) - 1

        # mom_3m — skip último mes (nueva señal)
        datos[ticker]["mom_3m"] = close.shift(21) / close.shift(63) - 1

        # Vol 21d anualizada — para vol scaling
        ret_d = np.exp(datos[ticker][COL_LOGRET].fillna(0)) - 1
        datos[ticker]["vol_21d"] = ret_d.rolling(21).std() * np.sqrt(252)

    tickers_ok = [t for t in tickers_necesarios if t in datos]
    print(f"Cargados ({len(tickers_ok)}): {tickers_ok}")
    return datos


# ─────────────────────────────────────────────────────────────
# PASO 2: CONSTRUCCIÓN DE PESOS MENSUALES
# ─────────────────────────────────────────────────────────────
def construir_pesos(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Estrategia Maestra — por cada primer día hábil del mes:

    1. Señal: 0.6×mom_12m + 0.4×mom_3m
    2. Dual SMA200: ETF > SMA200  AND  SPY > SMA200
    3. TOP-1: 100% al ganador que pase el dual-regime
    4. Vol scaling: peso = min(1, TARGET_VOL / vol_21d_ganador) → resto a SHY
    5. Fallback: si ninguno pasa → mejor señal absoluta (sin filtro, siempre invertido)
    """
    todas_fechas = datos[REFUGIO].loc[START_DATE:END_DATE].index
    fechas_reb   = todas_fechas.to_series().resample("MS").first().dropna().index

    etfs_activos = [t for t in ETFS if t in datos]
    todas_cols   = [REFUGIO] + etfs_activos
    registros    = []

    spy_df = datos.get("SPY")

    for fecha in fechas_reb:
        idx_shy = datos[REFUGIO].index
        previas = idx_shy[idx_shy < fecha]
        if len(previas) == 0:
            continue
        fecha_lag = previas[-1]

        scores_pass = {}   # pasan EMA200
        scores_all  = {}   # fallback (ignora filtro)

        for ticker in etfs_activos:
            df_t   = datos[ticker]
            prev_t = df_t.index[df_t.index <= fecha_lag]
            if len(prev_t) < 260:
                continue
            fila   = df_t.loc[prev_t[-1]]
            precio = fila.get(COL_CLOSE, np.nan)
            ema    = fila.get("ema_200", np.nan)
            m12    = fila.get(COL_MOM12, np.nan)

            if pd.isna(precio) or pd.isna(m12):
                continue

            scores_all[ticker] = m12
            if not pd.isna(ema) and precio > ema:
                scores_pass[ticker] = m12

        pool = scores_pass if scores_pass else scores_all
        fila_pesos = {col: 0.0 for col in todas_cols}

        if not pool:
            fila_pesos[REFUGIO] = 1.0
        else:
            ganador = max(pool, key=pool.get)
            fila_pesos[ganador] = 1.0

        fila_pesos["date"] = fecha
        registros.append(fila_pesos)

    df_pesos = pd.DataFrame(registros).set_index("date")
    return df_pesos[[c for c in todas_cols if c in df_pesos.columns]]


# ─────────────────────────────────────────────────────────────
# PASO 3: BACKTEST
# ─────────────────────────────────────────────────────────────
def run_backtest(
    df_pesos: pd.DataFrame,
    datos: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    dias       = datos[REFUGIO].loc[START_DATE:END_DATE].index
    peso_map   = df_pesos.to_dict("index")

    valor  = INITIAL_CAP
    cur_w  = {REFUGIO: 1.0}
    prev_w = {}
    records = []

    for fecha in dias:
        if fecha in peso_map:
            new_w    = {t: w for t, w in peso_map[fecha].items() if w > 1e-6}
            turnover = sum(
                abs(new_w.get(t, 0) - prev_w.get(t, 0))
                for t in set(new_w) | set(prev_w)
            )
            valor  *= 1 - turnover * (TXN_COST_BPS / 10_000)
            cur_w   = new_w
            prev_w  = new_w.copy()

        dr = 0.0
        for ticker, weight in cur_w.items():
            df = datos.get(ticker)
            if df is None or fecha not in df.index:
                continue
            lr = df.loc[fecha, COL_LOGRET] if COL_LOGRET in df.columns else np.nan
            if not pd.isna(lr):
                dr += weight * (np.exp(lr) - 1)

        valor *= (1 + dr)
        records.append({"date": fecha, "strategy": valor})

    equity = pd.DataFrame(records).set_index("date")

    for col_name, ticker in [("spy", "SPY"), ("shy", "SHY"), ("tlt", "TLT")]:
        df = datos.get(ticker)
        if df is not None:
            sc = df[COL_CLOSE].reindex(equity.index).ffill()
            equity[col_name] = INITIAL_CAP * sc / sc.iloc[0]

    spy_df = datos.get("SPY")
    tlt_df = datos.get("TLT")
    if spy_df is not None and tlt_df is not None:
        spy_r = spy_df[COL_LOGRET].reindex(equity.index).fillna(0)
        tlt_r = tlt_df[COL_LOGRET].reindex(equity.index).fillna(0)
        blend = 0.6 * (np.exp(spy_r) - 1) + 0.4 * (np.exp(tlt_r) - 1)
        equity["60_40"] = INITIAL_CAP * (1 + blend).cumprod()

    return equity


# ─────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────
def metrics(s: pd.Series, rf_annual: float = 0.0) -> dict:
    r      = s.pct_change().dropna()
    years  = len(r) / 252
    cagr   = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    vol    = r.std() * np.sqrt(252)
    rf_d   = (1 + rf_annual) ** (1 / 252) - 1
    sharpe = ((r - rf_d).mean() * 252) / (vol + 1e-10)
    neg    = r[r < 0]
    sortino = ((r - rf_d).mean() * 252) / (neg.std() * np.sqrt(252) + 1e-10)
    dd     = (s / s.cummax() - 1)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    hit    = (r > 0).mean()
    skew   = r.skew()
    kurt   = r.kurt()
    var95  = np.percentile(r, 5)
    cvar95 = r[r <= var95].mean()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
                max_dd=max_dd, calmar=calmar, hit_rate=hit,
                skew=skew, kurt=kurt, var95=var95, cvar95=cvar95)


def print_scorecard(equity: pd.DataFrame):
    cols = [
        ("2 the MOON ★",     "strategy"),
        ("S&P 500 (SPY)",    "spy"),
        ("60/40 Clásico",    "60_40"),
        ("Bonds (TLT)",      "tlt"),
        ("Cash (SHY)",       "shy"),
    ]
    spy_cagr = metrics(equity["spy"])["cagr"] if "spy" in equity else 0

    sep = "═" * 110
    print(f"\n{sep}")
    print(f"  2 the MOON  |  KN Hack 2026  |  {START_DATE[:4]}–{END_DATE[:4]}")
    print(f"  Universo ({len(ETFS)} ETFs): {ETFS}")
    sig_lbl = "mom_12m puro" if W_MOM3 == 0 else f"{W_MOM12}×mom_12m + {W_MOM3}×mom_3m"
    vol_lbl = f"Vol target: {TARGET_VOL*100:.0f}%" if TARGET_VOL else "Vol scaling: OFF"
    print(f"  Señal: {sig_lbl}  |  Filtro: EMA200  |  {vol_lbl}")
    print(sep)
    hdr = f"  {'Estrategia':<22} {'CAGR':>7} {'Vol':>6} {'Sharpe':>7} {'Sortino':>8} {'Max DD':>8} {'Calmar':>7} {'VaR95':>7} {'CVaR95':>8} {'Hit%':>6} {'Skew':>6}"
    print(hdr)
    print("  " + "─" * 105)
    for label, col in cols:
        if col not in equity:
            continue
        m     = metrics(equity[col])
        alpha = f"  α={( m['cagr'] - spy_cagr)*100:+.1f}%" if col == "strategy" else ""
        star  = "  ◄ ALPHA MÁXIMO vs SPY" if col == "strategy" else ""
        print(
            f"  {label:<22} {m['cagr']*100:>6.1f}%  {m['vol']*100:>5.1f}%  "
            f"{m['sharpe']:>6.2f}  {m['sortino']:>7.2f}  {m['max_dd']*100:>7.1f}%  "
            f"{m['calmar']:>6.2f}  {m['var95']*100:>6.1f}%  {m['cvar95']*100:>7.1f}%  "
            f"{m['hit_rate']*100:>5.1f}%  {m['skew']:>5.2f}{alpha}{star}"
        )
    print(sep)

    m_s = metrics(equity["strategy"])
    m_b = metrics(equity["spy"])
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    end_val = INITIAL_CAP * (equity["strategy"].iloc[-1] / equity["strategy"].iloc[0])
    print(f"\n  RESUMEN EJECUTIVO:")
    print(f"  • Retorno anual : {m_s['cagr']*100:.1f}%  vs  SPY {m_b['cagr']*100:.1f}%  →  +{(m_s['cagr']-m_b['cagr'])*100:.1f}pp alpha anual")
    print(f"  • Max Drawdown  : {m_s['max_dd']*100:.1f}%  vs  SPY {m_b['max_dd']*100:.1f}%")
    print(f"  • Sharpe        : {m_s['sharpe']:.2f}  vs  SPY {m_b['sharpe']:.2f}  →  {m_s['sharpe']/m_b['sharpe']:.1f}x mejor")
    print(f"  • Sortino       : {m_s['sortino']:.2f}  |  VaR 95%: {m_s['var95']*100:.2f}%/día  |  CVaR: {m_s['cvar95']*100:.2f}%/día")
    end_million = 1_000_000 * (equity["strategy"].iloc[-1] / equity["strategy"].iloc[0])
    spy_million  = 1_000_000 * (equity["spy"].iloc[-1]      / equity["spy"].iloc[0])
    print(f"  • $1M → ${end_million:,.0f}  vs  SPY ${spy_million:,.0f}  (+${end_million-spy_million:,.0f} extra)")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────
# VALIDACIÓN DE PESOS
# ─────────────────────────────────────────────────────────────
def validar_pesos(df_pesos: pd.DataFrame):
    sumas   = df_pesos.sum(axis=1)
    etfs_en = [e for e in ETFS if e in df_pesos.columns]

    print("\nValidación de pesos:")
    print(f"  Suma min/max  : {sumas.min():.4f} / {sumas.max():.4f}  (debe ser ~1.0)")
    print(f"  ¿NaN?         : {df_pesos.isna().any().any()}")
    print(f"  Rebalanceos   : {len(df_pesos)}  ({df_pesos.index[0].date()} → {df_pesos.index[-1].date()})")

    shy_col = df_pesos.get(REFUGIO)
    if shy_col is not None:
        avg_shy = shy_col.mean() * 100
        print(f"  SHY promedio  : {avg_shy:.1f}%  (vol scaling activo)")

    if etfs_en:
        conteo = (df_pesos[etfs_en] > 0.001).sum()
        print(f"\n  Apariciones por ETF (meses como TOP-1):")
        for t in conteo.sort_values(ascending=False).index:
            if conteo[t] > 0:
                print(f"    {t:<6}: {int(conteo[t]):>3} meses  ({conteo[t]/len(df_pesos)*100:.0f}%)")


# ─────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────
_EVENTS = [
    ("2008-09-15", "Lehman",    "top"),
    ("2020-02-20", "COVID",     "bottom"),
    ("2022-01-03", "Fed Hikes", "top"),
]

_PALETTE = {
    "strategy": ("#e63b2e", 2.5, "-",  "2 the MOON v2"),
    "spy":      ("#7f7f7f", 1.5, "--", "S&P 500"),
    "60_40":    ("#c05020", 1.5, "-.", "60/40"),
    "tlt":      ("#27ae60", 1.2, ":",  "TLT"),
    "shy":      ("#aaaaaa", 0.9, ":",  "SHY"),
}


def _add_events(ax, equity, which="top"):
    for date_str, label, pos in _EVENTS:
        d = pd.Timestamp(date_str)
        if d < equity.index[0] or d > equity.index[-1]:
            continue
        ax.axvline(d, color="#cc0000", lw=0.8, ls=":", alpha=0.6)
        ypos = ax.get_ylim()[1 if pos == "top" else 0]
        va   = "top" if pos == "top" else "bottom"
        ax.annotate(label, xy=(d, ypos), fontsize=6.5, color="#cc0000",
                    rotation=90, va=va, ha="right", alpha=0.8)


def plot_resultados(equity: pd.DataFrame, df_pesos: pd.DataFrame):
    etfs_en    = [e for e in ETFS if e in df_pesos.columns]
    cols_alloc = [REFUGIO] + etfs_en

    fig1, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig1.suptitle(
        "2 the MOON v2 — Dual Momentum + SMA200 Regime + Vol Scaling  |  KN Hack 2026  |  2007–2025",
        fontsize=13, fontweight="bold", y=1.005,
    )

    ax = axes[0, 0]
    for col, (c, lw, ls, lbl) in _PALETTE.items():
        if col in equity:
            ax.plot(equity.index, equity[col], color=c, lw=lw, ls=ls, label=lbl)
    ax.set_yscale("log")
    ax.set_title("Equity Curve (log)", fontweight="bold")
    ax.set_ylabel("Valor ($)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    _add_events(ax, equity, "top")

    ax = axes[0, 1]
    for col, (c, lw, ls, lbl) in _PALETTE.items():
        if col in equity:
            dd = (equity[col] / equity[col].cummax() - 1) * 100
            ax.fill_between(equity.index, dd, 0, alpha=0.35, color=c, label=lbl)
    ax.set_title("Underwater / Drawdown (%)", fontweight="bold")
    ax.set_ylabel("%")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.25)
    _add_events(ax, equity, "bottom")

    ax = axes[1, 0]
    years = list(range(int(START_DATE[:4]), int(END_DATE[:4]) + 1))
    x, bw = np.arange(len(years)), 0.28
    series_bar = [
        ("strategy", "2 the MOON v2", "#e63b2e"),
        ("spy",      "S&P 500",       "#7f7f7f"),
        ("60_40",    "60/40",         "#c05020"),
    ]
    for i, (col, lbl, base_c) in enumerate(series_bar):
        if col not in equity:
            continue
        rets = []
        for y in years:
            seg = equity.loc[equity.index.year == y, col]
            rets.append((seg.iloc[-1] / seg.iloc[0] - 1) * 100 if len(seg) > 5 else 0.0)
        bars = ax.bar(x + (i - 1) * bw, rets, bw, label=lbl,
                      color=[("#2ecc71" if r >= 0 else "#e74c3c") if col == "strategy"
                             else base_c for r in rets],
                      alpha=0.85, edgecolor="white", lw=0.3)
        if col == "strategy":
            for bar, r in zip(bars, rets):
                if abs(r) > 1.5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.5 if r >= 0 else -1.8),
                            f"{r:.0f}%", ha="center", va="bottom", fontsize=5.5, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, fontsize=7)
    ax.set_title("Retorno Anual (%)", fontweight="bold")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    ax  = axes[1, 1]
    ax2 = ax.twinx()
    for col, (c, lw, ls, lbl) in list(_PALETTE.items())[:2]:
        if col not in equity:
            continue
        r  = equity[col].pct_change()
        rs = r.rolling(252).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252) + 1e-10))
        ax.plot(equity.index, rs, color=c, lw=1.6, ls=ls, label=f"Sharpe {lbl}")
    rv = equity["strategy"].pct_change().rolling(63).std() * np.sqrt(252) * 100
    ax2.fill_between(equity.index, rv, alpha=0.12, color="#1f4e79", label="Vol 63d (der.)")
    ax2.set_ylabel("Vol anualizada (%)", fontsize=8, color="#1f4e79")
    ax2.tick_params(axis="y", labelcolor="#1f4e79", labelsize=7)
    ax.axhline(1.0, color="green", ls=":", lw=1, alpha=0.6)
    ax.axhline(0.0, color="black", ls="-", lw=0.5, alpha=0.3)
    ax.set_title("Rolling Sharpe (1yr) + Volatilidad (63d)", fontweight="bold")
    ax.set_ylabel("Sharpe")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 0]
    alloc_m = (df_pesos[cols_alloc].resample("MS").last()
               .fillna(0).loc[START_DATE:END_DATE])
    cs = alloc_m.sum().sort_values(ascending=False).index.tolist()
    ax.stackplot(alloc_m.index,
                 [alloc_m[c] * 100 for c in cs],
                 labels=cs,
                 colors=[CATEGORY_COLORS.get(c, "#888") for c in cs],
                 alpha=0.88)
    ax.set_ylim(0, 105)
    ax.set_title("Allocación Mensual (%)", fontweight="bold")
    ax.set_ylabel("Peso (%)")
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2, 1]
    freq      = (df_pesos[etfs_en] > 0.001).sum().sort_values()
    total_reb = len(df_pesos)
    colors_bar = [CATEGORY_COLORS.get(t, "#888") for t in freq.index]
    bars = ax.barh(freq.index, freq.values, color=colors_bar, alpha=0.88, edgecolor="white")
    for bar, v in zip(bars, freq.values):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                f"{v}m  ({v/total_reb*100:.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, total_reb * 1.18)
    ax.axvline(total_reb, color="#cc0000", ls="--", lw=0.8, alpha=0.6, label="Todos los meses")
    ax.set_title("Meses activo por ETF (TOP-1)", fontweight="bold")
    ax.set_xlabel("Meses en portafolio")
    ax.legend(fontsize=7)
    ax.grid(True, axis="x", alpha=0.25)

    fig1.tight_layout()
    out1 = OUTPUT_DIR / "dm_dashboard.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"  → {out1}")

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
    fig2.suptitle("2 the MOON v2 — Analytics Profundos  |  KN Hack 2026",
                  fontsize=14, fontweight="bold")

    ax = axes2[0, 0]
    for col, (c, lw, ls, lbl) in list(_PALETTE.items())[:2]:
        if col not in equity:
            continue
        r = equity[col].resample("MS").last().pct_change().dropna() * 100
        ax.hist(r, bins=40, alpha=0.55, color=c,
                label=f"{lbl} (μ={r.mean():.1f}%  σ={r.std():.1f}%)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Distribución Retornos Mensuales", fontweight="bold")
    ax.set_xlabel("Retorno mensual (%)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes2[0, 1]
    r_strat = equity["strategy"].resample("MS").last().pct_change().dropna() * 100
    r_spy   = equity["spy"].resample("MS").last().pct_change().dropna() * 100
    idx     = r_strat.index.intersection(r_spy.index)
    rs, rp  = r_strat.loc[idx].values, r_spy.loc[idx].values
    colors_sc = ["#2ecc71" if s >= 0 else "#e74c3c" for s in rs]
    ax.scatter(rp, rs, c=colors_sc, alpha=0.55, s=22, edgecolors="white", lw=0.3)
    lim = max(abs(np.concatenate([rp, rs]))) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], color="#aaaaaa", lw=0.8, ls="--", label="Línea 45°")
    coef  = np.polyfit(rp, rs, 1)
    trend = np.poly1d(coef)
    ax.plot(np.sort(rp), trend(np.sort(rp)), color="#1f4e79", lw=1.5,
            label=f"Reg. lineal  β={coef[0]:.2f}")
    beats = np.mean(rs > rp) * 100
    ax.set_title(f"Estrategia vs S&P 500 (mensual)  |  Gana {beats:.0f}% de los meses",
                 fontweight="bold")
    ax.set_xlabel("S&P 500 retorno mensual (%)")
    ax.set_ylabel("2 the MOON v2 retorno mensual (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes2[1, 0]
    monthly_r = equity["strategy"].resample("MS").last().pct_change() * 100
    years_u   = sorted(monthly_r.index.year.unique())
    grid      = np.full((len(years_u), 12), np.nan)
    for r_idx, yr in enumerate(years_u):
        for c_idx, mo in enumerate(range(1, 13)):
            mask = (monthly_r.index.year == yr) & (monthly_r.index.month == mo)
            if mask.any():
                grid[r_idx, c_idx] = monthly_r[mask].iloc[0]
    vmax = np.nanpercentile(np.abs(grid), 95)
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#c0392b", "#ffffff", "#27ae60"])
    im   = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for r_idx in range(len(years_u)):
        for c_idx in range(12):
            v = grid[r_idx, c_idx]
            if not np.isnan(v):
                ax.text(c_idx, r_idx, f"{v:.1f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > vmax * 0.5 else "black")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], fontsize=8)
    ax.set_yticks(range(len(years_u)))
    ax.set_yticklabels(years_u, fontsize=8)
    fig2.colorbar(im, ax=ax, label="Retorno mensual (%)", shrink=0.8)
    ax.set_title("Heatmap Retornos Mensuales (%)", fontweight="bold")

    ax = axes2[1, 1]
    for col, (c, lw, ls, lbl) in list(_PALETTE.items())[:3]:
        if col not in equity:
            continue
        r12 = equity[col].pct_change(252) * 100
        ax.plot(equity.index, r12, color=c, lw=lw, ls=ls, label=lbl, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(equity.index,
                    equity["strategy"].pct_change(252) * 100, 0,
                    where=equity["strategy"].pct_change(252) > 0,
                    alpha=0.08, color="#1f4e79")
    ax.fill_between(equity.index,
                    equity["strategy"].pct_change(252) * 100, 0,
                    where=equity["strategy"].pct_change(252) < 0,
                    alpha=0.12, color="#c0392b")
    ax.set_title("Retorno Rolling 12 Meses (%)", fontweight="bold")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    _add_events(ax, equity, "top")

    fig2.tight_layout()
    out2 = OUTPUT_DIR / "dm_analytics.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  → {out2}")

    plt.close("all")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  2 the MOON  |  KN Hack 2026")
    print(f"  Universo : {ETFS}")
    print(f"  Señal    : mom_12m  |  Filtro: EMA200  |  TOP-1  |  Siempre invertido")
    print("=" * 70 + "\n")

    print("Cargando datos...")
    datos = cargar_datos()

    print("\nConstruyendo pesos mensuales (Estrategia Maestra)...")
    df_pesos = construir_pesos(datos)
    validar_pesos(df_pesos)

    print("\nRunning backtest...")
    equity = run_backtest(df_pesos, datos)

    print_scorecard(equity)

    df_pesos.to_csv("portfolio_weights.csv")
    equity.to_csv(OUTPUT_DIR / "dual_momentum_equity.csv")
    print("  Archivos guardados:")
    print("    portfolio_weights.csv")
    print(f"    {OUTPUT_DIR}/dual_momentum_equity.csv")

    print("\nGenerando gráficos...")
    plot_resultados(equity, df_pesos)

    print("\n  Listo.\n")


if __name__ == "__main__":
    main()
