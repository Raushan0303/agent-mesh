MOCK_SUPPLIERS = [
    {
        "name": "SupplierAlpha",
        "item": "USB-C cable",
        "price": 2.50,
        "lead_time_days": 7,
        "rating": 4.5,
    },
    {
        "name": "SupplierBeta",
        "item": "USB-C cable",
        "price": 2.85,
        "lead_time_days": 5,
        "rating": 4.8,
    },
    {
        "name": "SupplierGamma",
        "item": "USB-C cable",
        "price": 3.10,
        "lead_time_days": 3,
        "rating": 4.2,
    },
    {
        "name": "SupplierDelta",
        "item": "HDMI cable",
        "price": 4.50,
        "lead_time_days": 10,
        "rating": 4.0,
    },
    {
        "name": "SupplierEpsilon",
        "item": "HDMI cable",
        "price": 3.90,
        "lead_time_days": 6,
        "rating": 4.6,
    },
    {
        "name": "SupplierZeta",
        "item": "Ethernet cable",
        "price": 1.20,
        "lead_time_days": 4,
        "rating": 4.3,
    },
    {
        "name": "SupplierEta",
        "item": "Ethernet cable",
        "price": 1.50,
        "lead_time_days": 2,
        "rating": 4.7,
    },
    {
        "name": "SupplierTheta",
        "item": "Power adapter",
        "price": 8.75,
        "lead_time_days": 14,
        "rating": 3.9,
    },
]


def query_mock_marketplace(item: str, budget: float, quantity: int) -> list[dict]:
    """Returns suppliers matching the item within budget."""
    return [
        s
        for s in MOCK_SUPPLIERS
        if s["item"].lower() == item.lower() and s["price"] * quantity <= budget * quantity
    ]
