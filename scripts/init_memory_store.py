"""Populate the memory store with past sourcing decisions.

This is the harness capability the Decide node queries for "have we
sourced this before?" — stored under the "sourcing-agent" namespace.

Each entry is a summary of a past sourcing decision:
  - item, chosen supplier, final price, founder feedback

Run this script to populate the memory store with seed data:
    PYTHONPATH=. python scripts/init_memory_store.py
"""

import asyncio
import logging

from app.agentmesh.memory.store import get_memory_store, close_memory_store

logger = logging.getLogger("agentmesh.sourcing_agent.init_memory")

NAMESPACE = "sourcing-agent"

# Seed data: past sourcing decisions
PAST_DECISIONS = [
    {
        "content": "Sourced USB-C cables from SupplierBeta, 500 units at $2.85/unit, total $1425. "
                    "Lead time 5 days, rating 4.8. Founder feedback: excellent quality, fast delivery.",
        "metadata": {
            "item": "USB-C cable",
            "supplier": "SupplierBeta",
            "unit_price": 2.85,
            "quantity": 500,
            "total_cost": 1425.0,
            "lead_time_days": 5,
            "rating": 4.8,
            "feedback": "positive",
        },
    },
    {
        "content": "Sourced USB-C cables from SupplierAlpha, 1000 units at $2.50/unit, total $2500. "
                    "Lead time 7 days, rating 4.5. Founder feedback: good value, slight delay.",
        "metadata": {
            "item": "USB-C cable",
            "supplier": "SupplierAlpha",
            "unit_price": 2.50,
            "quantity": 1000,
            "total_cost": 2500.0,
            "lead_time_days": 7,
            "rating": 4.5,
            "feedback": "neutral",
        },
    },
    {
        "content": "Sourced HDMI cables from SupplierEpsilon, 200 units at $3.90/unit, total $780. "
                    "Lead time 6 days, rating 4.6. Founder feedback: great product, on time.",
        "metadata": {
            "item": "HDMI cable",
            "supplier": "SupplierEpsilon",
            "unit_price": 3.90,
            "quantity": 200,
            "total_cost": 780.0,
            "lead_time_days": 6,
            "rating": 4.6,
            "feedback": "positive",
        },
    },
    {
        "content": "Sourced HDMI cables from SupplierDelta, 50 units at $4.50/unit, total $225. "
                    "Lead time 10 days, rating 4.0. Founder feedback: acceptable but slow.",
        "metadata": {
            "item": "HDMI cable",
            "supplier": "SupplierDelta",
            "unit_price": 4.50,
            "quantity": 50,
            "total_cost": 225.0,
            "lead_time_days": 10,
            "rating": 4.0,
            "feedback": "neutral",
        },
    },
    {
        "content": "Sourced Ethernet cables from SupplierEta, 300 units at $1.50/unit, total $450. "
                    "Lead time 2 days, rating 4.7. Founder feedback: excellent, very fast delivery.",
        "metadata": {
            "item": "Ethernet cable",
            "supplier": "SupplierEta",
            "unit_price": 1.50,
            "quantity": 300,
            "total_cost": 450.0,
            "lead_time_days": 2,
            "rating": 4.7,
            "feedback": "positive",
        },
    },
    {
        "content": "Sourced Ethernet cables from SupplierZeta, 1000 units at $1.20/unit, total $1200. "
                    "Lead time 4 days, rating 4.3. Founder feedback: good price, reliable.",
        "metadata": {
            "item": "Ethernet cable",
            "supplier": "SupplierZeta",
            "unit_price": 1.20,
            "quantity": 1000,
            "total_cost": 1200.0,
            "lead_time_days": 4,
            "rating": 4.3,
            "feedback": "positive",
        },
    },
    {
        "content": "Sourced Power adapters from SupplierTheta, 100 units at $8.75/unit, total $875. "
                    "Lead time 14 days, rating 3.9. Founder feedback: slow but quality acceptable.",
        "metadata": {
            "item": "Power adapter",
            "supplier": "SupplierTheta",
            "unit_price": 8.75,
            "quantity": 100,
            "total_cost": 875.0,
            "lead_time_days": 14,
            "rating": 3.9,
            "feedback": "neutral",
        },
    },
    {
        "content": "Sourced USB-C cables from SupplierGamma, 5000 units at $3.10/unit, total $15500. "
                    "Lead time 3 days, rating 4.2. Founder feedback: fast but expensive.",
        "metadata": {
            "item": "USB-C cable",
            "supplier": "SupplierGamma",
            "unit_price": 3.10,
            "quantity": 5000,
            "total_cost": 15500.0,
            "lead_time_days": 3,
            "rating": 4.2,
            "feedback": "neutral",
        },
    },
]


async def populate_memory_store():
    """Populate the memory store with seed past sourcing decisions."""
    store = await get_memory_store()

    # Clear existing entries
    await store.clear(NAMESPACE)

    # Store each past decision
    for decision in PAST_DECISIONS:
        await store.store(
            namespace=NAMESPACE,
            content=decision["content"],
            metadata=decision["metadata"],
        )

    entries = await store.get_all(NAMESPACE)
    print(f"Memory store populated: {len(entries)} entries under '{NAMESPACE}'")
    for e in entries:
        print(f"  {e['id'][:8]}... {e['content'][:60]}...")

    await close_memory_store()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(populate_memory_store())
