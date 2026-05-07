# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Importamos la logica de negocio desde nuestro motor separado.
from engine import (
    baseline_reorder,
    optimized_recommendation,
    simulate_cost_impact,
    fmt_currency,
    fmt_pct,
)

# ======================================================
# 1) Configuracion general.
# ======================================================

st.set_page_config(
    page_title="Supply Chain Control Tower",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# 2) Estilo visual.
# ======================================================

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3 {
        color: #111827;
        font-family: Inter, Arial, sans-serif;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%);
        color: white;
        padding: 22px 24px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }
    .hero-card h1, .hero-card p {
        color: white;
        margin-bottom: 0;
    }
    .metric-card {
        background: #ffffff;
        padding: 16px 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
    }
    .section-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.05);
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
    }
    .kpi-subtext {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }
    .insight-box {
        background: #f9fafb;
        border-left: 4px solid #111827;
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.82rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================
# 3) Deteccion de rutas.
# ======================================================

def detect_project_root() -> Path:
    current_file = Path(__file__).resolve()
    # Busca dinamicamente la carpeta que contiene "data".
    for parent in [current_file.parent, current_file.parent.parent, Path.cwd()]:
        if (parent / "data").exists():
            return parent
    return Path.cwd()

PROJECT_DIR = detect_project_root()
DATA_DIR = PROJECT_DIR / "data" / "processed"

# ======================================================
# 4) Carga de datos con cache y manejo de errores.
# ======================================================

@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = DATA_DIR / "feature_engineered_dataset.csv"
    scoring_path = DATA_DIR / "model_scoring_dataset.csv"

    if not feature_path.exists() or not scoring_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(feature_path, parse_dates=["date"])
    scoring = pd.read_csv(scoring_path, parse_dates=["date"])
    return df, scoring

@st.cache_data(show_spinner=False)
def prepare_dataset() -> pd.DataFrame:
    df, scoring = load_data()
    if df.empty or scoring.empty:
        return pd.DataFrame()

    cols = [
        "date",
        "mine_id",
        "item_id",
        "predicted_stockout_proba_next_day",
        "predicted_stockout_flag_next_day",
        "decision_alert_level",
    ]
    cols = [c for c in cols if c in scoring.columns]

    merged = df.merge(scoring[cols], on=["date", "mine_id", "item_id"], how="left")
    merged["predicted_stockout_proba_next_day"] = merged["predicted_stockout_proba_next_day"].fillna(0.0)
    merged["decision_alert_level"] = merged["decision_alert_level"].fillna("low")
    return merged

# Ejecucion de carga.
merged = prepare_dataset()

if merged.empty:
    st.error("""No se encontraron los datos procesados en la carpeta 'data/processed'. 
             Por favor, ejecuta los notebooks 02 y 03 primero para generar los archivos CSV.""")
    st.stop()

# ======================================================
# 5) Encabezado ejecutivo.
# ======================================================

hero_col1, hero_col2 = st.columns([3, 1])
with hero_col1:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Supply Chain Control Tower</h1>
            <p>Mining Supply Chain Decision Intelligence | ML + RL Optimization</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_col2:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">Cobertura Operativa</div>
            <div class="kpi-value">{merged['mine_id'].nunique()}</div>
            <div class="kpi-subtext">Minas Analizadas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ======================================================
# 6) Sidebar: Filtros y Simulador.
# ======================================================

st.sidebar.header("Filtros Ejecutivos.")

selected_mine = st.sidebar.selectbox("Mina", sorted(merged["mine_id"].dropna().unique()))
selected_item = st.sidebar.selectbox("Material (Item)", sorted(merged["item_id"].dropna().unique()))
selected_weather = st.sidebar.selectbox("Escenario Climatico", ["clear", "cloudy", "white_wind"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulador de Decisiones")
current_stock_sim = st.sidebar.slider("Stock Actual", 0, 1500, 300, 10)
demand_sim = st.sidebar.slider("Demanda Esperada", 0, 500, 120, 5)
reorder_point_sim = st.sidebar.slider("Punto de Reorden", 0, 800, 180, 10)
target_stock_sim = st.sidebar.slider("Stock Objetivo", 0, 1200, 420, 10)
risk_proba_sim = st.sidebar.slider("Riesgo Predictivo (ML)", 0.0, 1.0, 0.35, 0.01)
unit_cost_sim = st.sidebar.slider("Costo Unitario ($)", 1.0, 300.0, 24.0, 0.5)
holding_cost_factor_sim = st.sidebar.slider("Factor Costo Mantenimiento", 0.01, 0.30, 0.14, 0.01)

# Filtramos dataset historico.
filtered = merged[(merged["mine_id"] == selected_mine) & (merged["item_id"] == selected_item)].copy().sort_values("date")

# ======================================================
# 7) Capa de Decision y Costos (Simulacion).
# ======================================================

st.markdown("### Recomendador de Reabastecimiento")

baseline_qty = baseline_reorder(current_stock_sim,
                                reorder_point_sim,
                                target_stock_sim)
optimized_qty = optimized_recommendation(
    current_stock_sim,
    reorder_point_sim,
    target_stock_sim,
    risk_proba_sim,
    selected_weather,
    demand_sim
)

baseline_impact = simulate_cost_impact(current_stock_sim,
                                       demand_sim,
                                       baseline_qty,
                                       unit_cost_sim,
                                       holding_cost_factor_sim)
optimized_impact = simulate_cost_impact(current_stock_sim,
                                        demand_sim,
                                        optimized_qty,
                                        unit_cost_sim,
                                        holding_cost_factor_sim)

rec_cols = st.columns(3)
rec_cols[0].markdown(f"<div class='section-card'><div class='kpi-label'>Recomendacion SAP (Baseline)</div><div class='kpi-value'>{baseline_qty:,} u</div></div>", unsafe_allow_html=True)
rec_cols[1].markdown(f"<div class='section-card'><div class='kpi-label'>Recomendacion Optimizada (ML)</div><div class='kpi-value'>{optimized_qty:,} u</div></div>", unsafe_allow_html=True)
rec_cols[2].markdown(f"<div class='section-card'><div class='kpi-label'>Ahorro/Costo Diferencial</div><div class='kpi-value'>{fmt_currency(baseline_impact['total_cost'] - optimized_impact['total_cost'])}</div></div>", unsafe_allow_html=True)

if optimized_qty > baseline_qty:
    st.warning("La politica optimizada sugiere una postura defensiva para evitar el quiebre de stock al riesgo actual.")
else:
    st.success("La politica optimizada es mas eficiente y previene sobrestock bajo este escenario.")

# ======================================================
# 8) Analisis Historico y KPIs.
# ======================================================

st.markdown("### Historico de Operaciones")

left, right = st.columns([1.2, 1])
with left:
    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(x=filtered["date"], 
                                y=filtered["demand_units"], 
                                name="Demanda", 
                                mode="lines"))
    ts_fig.add_trace(go.Scatter(x=filtered["date"], 
                                y=filtered["ending_stock"], 
                                name="Stock Final", 
                                mode="lines"))
    ts_fig.add_trace(go.Scatter(x=filtered["date"], 
                                y=filtered["predicted_stockout_proba_next_day"], 
                                name="Riesgo", mode="lines", yaxis="y2", 
                                line=dict(dash='dot', 
                                          color='red')))
    ts_fig.update_layout(height=350, 
                         yaxis=dict(title="Unidades"), 
                         yaxis2=dict(title="Prob. Riesgo", 
                                     overlaying="y", 
                                     side="right", 
                                     range=[0, 1]), 
                         margin=dict(l=0, r=0, t=30, b=0), 
                         legend=dict(orientation="h", 
                                     y=1.1))
    st.plotly_chart(ts_fig, width="stretch")

with right:
    # Tabla de costos puente.
    cost_df = pd.DataFrame({
        "Escenario": ["Baseline", "Optimizado"],
        "Holding": [baseline_impact["holding_cost"], optimized_impact["holding_cost"]],
        "Stockout": [baseline_impact["stockout_cost"], optimized_impact["stockout_cost"]],
        "Transporte": [baseline_impact["transport_cost"], optimized_impact["transport_cost"]],
    })
    fig_cost = px.bar(cost_df, x="Escenario", y=["Holding", "Stockout", "Transporte"], title="Estructura de Costos Simulada")
    fig_cost.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_cost, width="stretch")

# ======================================================
# 9) Tabla Ejecutiva y Exportacion.
# ======================================================

st.markdown("### Detalle Operativo")
exec_table = filtered[["date", 
                       "demand_units", 
                       "ending_stock", 
                       "service_level",
                       "predicted_stockout_proba_next_day",
                       "decision_alert_level",
                       "total_cost"]].sort_values("date", ascending=False).head(20)

st.dataframe(exec_table, width="stretch", hide_index=True)

# Funcionalidad de Exportacion para Integracion (Simulado SAP MM)
csv_data = exec_table.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Exportar Resumen Operativo (CSV)",
    data=csv_data,
    file_name=f"SC_Report_{selected_mine}_{selected_item}.csv",
    mime="text/csv",
)