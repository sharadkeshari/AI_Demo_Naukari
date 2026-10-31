from __future__ import annotations

from typing import Any

from utils.models import ToolSpec


ORDERS: dict[str, dict[str, Any]] = {
    "ORD-100": {"status": "in_transit", "city": "Pune"},
    "ORD-200": {"status": "delivered", "city": "Delhi"},
}


def lookup_order(order_id: str) -> dict[str, Any]:
    """Return local order details without contacting an external service."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id, **order}


def estimate_delivery(postal_code: str, express: bool) -> dict[str, Any]:
    """Return a deterministic delivery estimate."""
    days = 1 if express else 3
    return {"postal_code": postal_code, "estimated_days": days}


def draft_ticket(order_id: str, reason: str) -> dict[str, Any]:
    """Create a local draft only. This function sends nothing."""
    return {
        "draft_id": f"DRAFT-{order_id}",
        "order_id": order_id,
        "reason": reason,
        "sent": False,
    }


def build_registry() -> dict[str, ToolSpec]:
    return {
        "lookup_order": ToolSpec(
            lookup_order,
            {"order_id": str},
            description="Look up the current status and city for an order.",
        ),
        "estimate_delivery": ToolSpec(
            estimate_delivery,
            {"postal_code": str, "express": bool},
            description="Estimate delivery time for a postal code.",
        ),
        "draft_ticket": ToolSpec(
            draft_ticket,
            {"order_id": str, "reason": str},
            description="Prepare a support-ticket draft without sending it.",
        ),
    }
