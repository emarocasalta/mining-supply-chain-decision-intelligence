# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import (
    baseline_reorder,
    optimized_recommendation,
    simulate_cost_impact,
    fmt_currency,
    fmt_pct,
)

# ======================================================
# Configuración general de Streamlit.
# ======================================================

st.set_page_config(
    page_title="Mining Supply Chain Decision Intelligence",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# CSS custom.
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
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }

    .hero-card h1,
    .hero-card p {
        color: white;
        margin-bottom: 0;
    }

    .section-card {
        background: white;
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
        font-size: 1.7rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
    }

    .kpi-subtext {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.82rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================
# Rutas.
# ======================================================

def detect_project_root() -> Path:
    """
    Detecta automáticamente la raíz del proyecto.

    Sirve para:
    - ejecución local en VS Code
    - Streamlit Cloud
    - repos con app.py en raíz o en /notebooks
    """
    current_file = Path(__file__).resolve()

    candidates = [
        current_file.parent,
        current_file.parent.parent,
        Path.cwd(),
    ]

    for candidate in candidates:
        if (candidate / "data").exists():
            return candidate

    return Path.cwd()


PROJECT_DIR = detect_project_root()
DATA_DIR = PROJECT_DIR / "data" / "processed"

# ======================================================
# Carga de datos.
# ======================================================

@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga datasets procesados desde data/processed.
    """
    feature_path = DATA_DIR / "feature_engineered_dataset.csv"
    scoring_path = DATA_DIR / "model_scoring_dataset.csv"

    if not feature_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {feature_path}")

    if not scoring_path.exists():
        raise FileNotFoundError(f"No se encontró el scoring: {scoring_path}")

    df = pd.read_csv(feature_path, parse_dates=["date"])
    scoring = pd.read_csv(scoring_path, parse_dates=["date"])
    return df, scoring


@st.cache_data(show_spinner=False)
def prepare_dataset() -> pd.DataFrame:
    """
    Une el dataset principal con el scoring del modelo supervisado.
    """
    df, scoring = load_data()

    cols = [
        "date",
        "mine_id",
        "item_id",
        "predicted_stockout_proba_next_day",
        "predicted_stockout_flag_next_day",
        "decision_alert_level",
    ]
    cols = [c for c in cols if c in scoring.columns]

    merged = df.merge(
        scoring[cols],
        on=["date", "mine_id", "item_id"],
        how="left",
    )

    merged["predicted_stockout_proba_next_day"] = merged["predicted_stockout_proba_next_day"].fillna(0.0)
    merged["decision_alert_level"] = merged["decision_alert_level"].fillna("low")
    return merged


# ======================================================
# Helpers de render.
# ======================================================

def render_header(merged: pd.DataFrame) -> None:
    """Muestra el encabezado ejecutivo."""
    hero_col1, hero_col2 = st.columns([3, 1])

    with hero_col1:
        st.markdown(
            """
            <div class="hero-card">
                <h1>Supply Chain Control Tower</h1>
                <p>Mining Supply Chain Decision Intelligence</p>
                <p style="margin-top:10px; opacity:0.9;">
                    Machine Learning + RL-inspired replenishment optimization.
                </p>
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
                <div class="kpi-subtext">minas monitoreadas</div>

                <div style="height:12px"></div>

                <div class="kpi-label">Horizonte</div>
                <div class="kpi-value">{merged['date'].nunique()}</div>
                <div class="kpi-subtext">días simulados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(merged: pd.DataFrame) -> Tuple[str, str, str, int, int, int, int, float, float, float]:
    """
    Renderiza filtros y simulador, y devuelve las selecciones.
    """
    st.sidebar.header("Executive Filters")

    selected_mine = st.sidebar.selectbox(
        "Mine",
        sorted(merged["mine_id"].dropna().unique()),
    )

    selected_item = st.sidebar.selectbox(
        "Item",
        sorted(merged["item_id"].dropna().unique()),
    )

    selected_weather = st.sidebar.selectbox(
        "Weather Scenario",
        ["clear", "cloudy", "white_wind"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Decision Simulator")

    current_stock_sim = st.sidebar.slider("Current Stock", 0, 1500, 300, 10)
    demand_sim = st.sidebar.slider("Expected Demand", 0, 500, 120, 5)
    reorder_point_sim = st.sidebar.slider("Reorder Point", 0, 800, 180, 10)
    target_stock_sim = st.sidebar.slider("Target Stock", 0, 1200, 420, 10)
    risk_proba_sim = st.sidebar.slider("Predicted Stockout Risk", 0.0, 1.0, 0.35, 0.01)
    unit_cost_sim = st.sidebar.slider("Unit Cost", 1.0, 300.0, 24.0, 0.5)
    holding_cost_factor_sim = st.sidebar.slider("Holding Cost Factor", 0.01, 0.30, 0.14, 0.01)

    return (
        selected_mine,
        selected_item,
        selected_weather,
        current_stock_sim,
        demand_sim,
        reorder_point_sim,
        target_stock_sim,
        risk_proba_sim,
        unit_cost_sim,
        holding_cost_factor_sim,
    )


def render_simulation_panel(
    current_stock_sim: int,
    demand_sim: int,
    reorder_point_sim: int,
    target_stock_sim: int,
    risk_proba_sim: float,
    selected_weather: str,
    unit_cost_sim: float,
    holding_cost_factor_sim: float,
) -> tuple[int, int, dict, dict]:
    """
    Muestra recomendaciones baseline/optimizada y calcula impactos.
    """
    st.markdown("## Replenishment Recommendation Engine")

    baseline_qty = baseline_reorder(
        current_stock_sim,
        reorder_point_sim,
        target_stock_sim,
    )

    optimized_qty = optimized_recommendation(
        stock=current_stock_sim,
        reorder_point=reorder_point_sim,
        target_stock=target_stock_sim,
        risk_proba=risk_proba_sim,
        weather_state=selected_weather,
        demand_units=demand_sim,
    )

    baseline_impact = simulate_cost_impact(
        stock=current_stock_sim,
        demand=demand_sim,
        order_qty=baseline_qty,
        unit_cost=unit_cost_sim,
        holding_cost_factor=holding_cost_factor_sim,
    )

    optimized_impact = simulate_cost_impact(
        stock=current_stock_sim,
        demand=demand_sim,
        order_qty=optimized_qty,
        unit_cost=unit_cost_sim,
        holding_cost_factor=holding_cost_factor_sim,
    )

    rec_cols = st.columns(3)

    rec_cols[0].markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">Baseline Recommendation</div>
            <div class="kpi-value">{baseline_qty:,} u</div>
            <div class="kpi-subtext">Classic reorder-point policy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rec_cols[1].markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">Optimized Recommendation</div>
            <div class="kpi-value">{optimized_qty:,} u</div>
            <div class="kpi-subtext">Risk-aware decision layer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rec_cols[2].markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">Estimated Cost Delta</div>
            <div class="kpi-value">{fmt_currency(baseline_impact['total_cost'] - optimized_impact['total_cost'])}</div>
            <div class="kpi-subtext">Baseline minus optimized</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if optimized_qty > baseline_qty:
        st.warning(
            "The optimized policy recommends a more defensive posture under the current operational risk."
        )
    else:
        st.success(
            "The optimized policy reduces overstock while maintaining operational continuity."
        )

    return baseline_qty, optimized_qty, baseline_impact, optimized_impact


def render_operational_section(
    filtered: pd.DataFrame,
    baseline_impact: dict,
    optimized_impact: dict,
) -> None:
    """Dibuja series temporales, costos y tabla ejecutiva."""
    st.markdown("## Operational Analytics")

    left, right = st.columns([1.3, 1])

    with left:
        ts_fig = go.Figure()

        ts_fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered["demand_units"],
                mode="lines",
                name="Demand",
            )
        )

        ts_fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered["ending_stock"],
                mode="lines",
                name="Ending Stock",
            )
        )

        ts_fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered["predicted_stockout_proba_next_day"],
                mode="lines",
                name="Predicted Risk",
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )

        ts_fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title="Units"),
            yaxis2=dict(
                title="Risk",
                overlaying="y",
                side="right",
                range=[0, 1],
            ),
            legend=dict(orientation="h", y=1.1),
        )

        st.plotly_chart(ts_fig, use_container_width=True)

    with right:
        cost_df = pd.DataFrame(
            {
                "Scenario": ["Baseline", "Optimized"],
                "Holding": [
                    baseline_impact["holding_cost"],
                    optimized_impact["holding_cost"],
                ],
                "Stockout": [
                    baseline_impact["stockout_cost"],
                    optimized_impact["stockout_cost"],
                ],
                "Transport": [
                    baseline_impact["transport_cost"],
                    optimized_impact["transport_cost"],
                ],
            }
        )

        fig_cost = px.bar(
            cost_df,
            x="Scenario",
            y=["Holding", "Stockout", "Transport"],
            title="Simulated Cost Structure",
        )

        fig_cost.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("## Operational Detail")

    exec_table = (
        filtered[
            [
                "date",
                "demand_units",
                "ending_stock",
                "service_level",
                "predicted_stockout_proba_next_day",
                "decision_alert_level",
                "total_cost",
            ]
        ]
        .sort_values("date", ascending=False)
        .head(20)
    )

    st.dataframe(
        exec_table,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = exec_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Export Operational Summary (CSV)",
        data=csv_data,
        file_name="SC_Report.csv",
        mime="text/csv",
    )


def render_footer() -> None:
    """Pie de página."""
    st.markdown("---")
    st.markdown(
        """
        <div class="small-note">
            Mining Supply Chain Decision Intelligence |
            Executive dashboard with ML + RL-inspired decision layer.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================
# Función principal.
# ======================================================

def main() -> None:
    """
    Punto de entrada principal de la app.
    """
    try:
        merged = prepare_dataset()
    except Exception as e:
        st.error(f"Error cargando datasets:\n\n{e}")
        st.stop()

    if merged.empty:
        st.error(
            "No se encontraron datos procesados en data/processed. "
            "Ejecutá primero los notebooks 02 y 03."
        )
        st.stop()

    render_header(merged)

    (
        selected_mine,
        selected_item,
        selected_weather,
        current_stock_sim,
        demand_sim,
        reorder_point_sim,
        target_stock_sim,
        risk_proba_sim,
        unit_cost_sim,
        holding_cost_factor_sim,
    ) = render_sidebar(merged)

    filtered = (
        merged[
            (merged["mine_id"] == selected_mine)
            & (merged["item_id"] == selected_item)
        ]
        .copy()
        .sort_values("date")
    )

    if filtered.empty:
        st.warning(
            f"⚠️ No hay registros históricos para el Material '{selected_item}' "
            f"en la Mina '{selected_mine}'."
        )
        return

    st.markdown("## Executive Summary")

    k1, k2, k3 = st.columns(3)

    k1.metric("Total Cost", fmt_currency(filtered["total_cost"].sum()))
    k2.metric("Avg Service Level", fmt_pct(filtered["service_level"].mean()))
    k3.metric("Stockout Rate", fmt_pct(filtered["is_stockout"].mean()))

    st.markdown("## Key Insights")

    avg_risk = float(filtered["predicted_stockout_proba_next_day"].mean())
    avg_service = float(filtered["service_level"].mean())
    stockout_rate = float(filtered["is_stockout"].mean())

    worst_weather = (
        filtered.groupby("weather_state")["total_cost"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    ins1, ins2, ins3 = st.columns(3)

    ins1.markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">Risk posture</div>
            <div class="kpi-subtext">
                El riesgo promedio para {selected_mine} / {selected_item} es {fmt_pct(avg_risk)}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ins2.markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">Operational service</div>
            <div class="kpi-subtext">
                Servicio promedio: {fmt_pct(avg_service)} | Stockout rate: {fmt_pct(stockout_rate)}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ins3.markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">Weather effect</div>
            <div class="kpi-subtext">
                El clima {worst_weather} tiende a concentrar el mayor costo medio.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    (
        baseline_qty,
        optimized_qty,
        baseline_impact,
        optimized_impact,
    ) = render_simulation_panel(
        current_stock_sim=current_stock_sim,
        demand_sim=demand_sim,
        reorder_point_sim=reorder_point_sim,
        target_stock_sim=target_stock_sim,
        risk_proba_sim=risk_proba_sim,
        selected_weather=selected_weather,
        unit_cost_sim=unit_cost_sim,
        holding_cost_factor_sim=holding_cost_factor_sim,
    )

    render_operational_section(
        filtered=filtered,
        baseline_impact=baseline_impact,
        optimized_impact=optimized_impact,
    )

    render_footer()


# ======================================================
# Ejecución principal.
# ======================================================

if __name__ == "__main__":
    main()
