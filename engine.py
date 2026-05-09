# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

# =========================
# Constantes del motor
# =========================
TRANSPORT_BASE_COST = 180.0
TRANSPORT_COST_PER_UNIT = 1.8
STOCKOUT_PENALTY_MULTIPLIER = 4.5
HIGH_RISK_MULTIPLIER = 1.35
MEDIUM_RISK_MULTIPLIER = 1.15
MAX_CAPACITY = 1200.0


def baseline_reorder(stock: float, reorder_point: float, target_stock: float) -> int:
    """
    Política tradicional tipo reorder point (Punto de pedido).
    Si el stock cae por debajo del punto de reorden, pide hasta alcanzar
    el stock objetivo.
    """
    if stock < reorder_point:
        return int(max(0, target_stock - stock))
    return 0


def optimized_recommendation(
    stock: float,
    reorder_point: float,
    target_stock: float,
    risk_proba: float,
    weather_state: str,
    demand_units: float,
) -> int:
    """
    Regla de negocio inspirada en una política optimizada (RL/ML).
    Ajusta dinámicamente el pedido basado en el riesgo predictivo y el clima.
    """
    weather_multiplier = {
        "clear": 1.00,
        "cloudy": 1.10,
        "white_wind": 1.25,  # Penalización por clima extremo en minería.
    }.get(weather_state, 1.0)

    pressure = max(0.0, (reorder_point - stock) / max(reorder_point, 1))
    risk_boost = 1 + 0.65 * risk_proba
    demand_boost = 1 + min(0.35, demand_units / max(target_stock, 1) * 0.35)

    raw = max(0, target_stock - stock)

    # Escalado defensivo ante alto riesgo de quiebre de stock.
    if risk_proba >= 0.70:
        raw *= HIGH_RISK_MULTIPLIER
    elif risk_proba >= 0.45:
        raw *= MEDIUM_RISK_MULTIPLIER

    raw *= weather_multiplier
    raw *= (1 + 0.5 * pressure)
    raw *= risk_boost
    raw *= demand_boost

    # Supongamos que la capacidad física máxima es 1200
    max_capacity = MAX_CAPACITY
    # Aseguramos que el stock actual + el pedido no superen el almacén
    final_order = min(raw, max_capacity - stock)

    return int(round(max(0, final_order)))


def simulate_cost_impact(
    stock: float,
    demand: float,
    order_qty: float,
    unit_cost: float,
    holding_cost_factor: float,
) -> Dict[str, float]:
    """
    Calcula el impacto financiero simulado (Holding, Stockout y Transporte).
    """
    post_order_stock = stock + order_qty
    sales = min(post_order_stock, demand)
    ending_stock = max(0, post_order_stock - sales)
    stockout_units = max(0, demand - post_order_stock)

    # Costo de mantenimiento de inventario anualizado.
    holding_cost = ending_stock * unit_cost * holding_cost_factor / 365.0
    # Penalización por quiebre de stock (4.5x el costo unitario como heurística).
    stockout_cost = stockout_units * unit_cost * STOCKOUT_PENALTY_MULTIPLIER
    # Costo de transporte logístico.
    transport_cost = TRANSPORT_BASE_COST + TRANSPORT_COST_PER_UNIT * order_qty

    total_cost = holding_cost + stockout_cost + transport_cost
    service_level = 1.0 if demand == 0 else sales / demand

    return {
        "ending_stock": ending_stock,
        "stockout_units": stockout_units,
        "holding_cost": holding_cost,
        "stockout_cost": stockout_cost,
        "transport_cost": transport_cost,
        "total_cost": total_cost,
        "service_level": service_level,
    }


def fmt_currency(x: float) -> str:
    """
    Formatea valores a moneda.
    """
    return f"${x:,.0f}"


def fmt_pct(x: float) -> str:
    """
    Formatea valores a porcentaje.
    """
    return f"{x:.1%}"
