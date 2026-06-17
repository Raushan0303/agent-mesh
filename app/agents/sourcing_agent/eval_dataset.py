from app.agentmesh.evals.models import EvalScenario

# 50 sourcing scenarios — mix of success, no-match, and edge cases.
# Each scenario specifies the input brief, the expected tool sequence,
# and the expected outcome status.
#
# Marketplace reference (per-unit prices):
#   USB-C cable:     SupplierAlpha $2.50, SupplierBeta $2.85, SupplierGamma $3.10
#   HDMI cable:      SupplierDelta $4.50, SupplierEpsilon $3.90
#   Ethernet cable:  SupplierZeta $1.20, SupplierEta $1.50
#   Power adapter:   SupplierTheta $8.75
#
# A supplier matches when its per-unit price <= budget (case-insensitive item).

EVAL_DATASET: list[EvalScenario] = [
    # ── USB-C cable — success cases ──
    EvalScenario(
        scenario_id="sourcing-001",
        input={"item": "USB-C cable", "quantity": 500, "budget": 3.00, "deadline": "2026-08-20"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-002",
        input={"item": "USB-C cable", "quantity": 100, "budget": 3.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-003",
        input={"item": "USB-C cable", "quantity": 1, "budget": 5.00, "deadline": "2026-09-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-004",
        input={"item": "USB-C cable", "quantity": 10000, "budget": 3.50, "deadline": "2026-12-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── HDMI cable — success cases ──
    EvalScenario(
        scenario_id="sourcing-005",
        input={"item": "HDMI cable", "quantity": 200, "budget": 5.00, "deadline": "2026-08-25"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-006",
        input={"item": "HDMI cable", "quantity": 50, "budget": 4.50, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Ethernet cable — success cases ──
    EvalScenario(
        scenario_id="sourcing-007",
        input={"item": "Ethernet cable", "quantity": 300, "budget": 2.00, "deadline": "2026-09-15"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-008",
        input={"item": "Ethernet cable", "quantity": 10, "budget": 2.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Power adapter — success case ──
    EvalScenario(
        scenario_id="sourcing-009",
        input={"item": "Power adapter", "quantity": 100, "budget": 10.00, "deadline": "2026-10-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── No-match cases (impossible budget) ──
    EvalScenario(
        scenario_id="sourcing-010",
        input={"item": "USB-C cable", "quantity": 500, "budget": 0.01, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-011",
        input={"item": "HDMI cable", "quantity": 100, "budget": 0.01, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-012",
        input={"item": "Ethernet cable", "quantity": 500, "budget": 0.01, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    # ── Item not in marketplace at all ──
    EvalScenario(
        scenario_id="sourcing-013",
        input={"item": "DisplayPort cable", "quantity": 100, "budget": 10.00, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-014",
        input={"item": "SATA cable", "quantity": 50, "budget": 5.00, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-015",
        input={"item": "Thunderbolt cable", "quantity": 25, "budget": 50.00, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    # ── Edge quantities ──
    EvalScenario(
        scenario_id="sourcing-016",
        input={"item": "USB-C cable", "quantity": 1, "budget": 3.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-017",
        input={"item": "HDMI cable", "quantity": 1, "budget": 5.00, "deadline": "2026-11-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Case-insensitive item matching ──
    EvalScenario(
        scenario_id="sourcing-018",
        input={"item": "usb-c cable", "quantity": 100, "budget": 3.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-019",
        input={"item": "hdmi cable", "quantity": 100, "budget": 5.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Large quantity, tight budget ──
    EvalScenario(
        scenario_id="sourcing-020",
        input={"item": "Ethernet cable", "quantity": 5000, "budget": 1.50, "deadline": "2026-12-31"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Budget exactly at supplier price boundaries ──
    EvalScenario(
        scenario_id="sourcing-021",
        input={"item": "USB-C cable", "quantity": 1000, "budget": 3.10, "deadline": "2026-08-15"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-022",
        input={"item": "USB-C cable", "quantity": 5000, "budget": 2.50, "deadline": "2027-01-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-023",
        input={"item": "USB-C cable", "quantity": 200, "budget": 2.84, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-024",
        input={"item": "HDMI cable", "quantity": 1000, "budget": 3.90, "deadline": "2026-09-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-025",
        input={"item": "HDMI cable", "quantity": 500, "budget": 4.50, "deadline": "2027-03-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-026",
        input={"item": "HDMI cable", "quantity": 10, "budget": 3.89, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-027",
        input={"item": "Ethernet cable", "quantity": 1000, "budget": 1.20, "deadline": "2026-08-10"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-028",
        input={"item": "Ethernet cable", "quantity": 5000, "budget": 1.49, "deadline": "2027-02-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-029",
        input={"item": "Ethernet cable", "quantity": 10000, "budget": 1.50, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-030",
        input={"item": "Power adapter", "quantity": 50, "budget": 8.75, "deadline": "2026-10-15"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Power adapter — no-match and generous budget ──
    EvalScenario(
        scenario_id="sourcing-031",
        input={"item": "Power adapter", "quantity": 10, "budget": 8.74, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-032",
        input={"item": "Power adapter", "quantity": 500, "budget": 9.00, "deadline": "2027-04-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-033",
        input={"item": "Power adapter", "quantity": 1, "budget": 10.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Case-insensitive matching — lowercase variants ──
    EvalScenario(
        scenario_id="sourcing-034",
        input={"item": "usb-c cable", "quantity": 5000, "budget": 3.00, "deadline": "2027-05-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-035",
        input={"item": "hdmi cable", "quantity": 1000, "budget": 4.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-036",
        input={"item": "ethernet cable", "quantity": 100, "budget": 1.30, "deadline": "2026-08-05"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-037",
        input={"item": "power adapter", "quantity": 200, "budget": 9.00, "deadline": "2027-06-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Case-insensitive matching — uppercase and mixed case ──
    EvalScenario(
        scenario_id="sourcing-038",
        input={"item": "USB-C CABLE", "quantity": 100, "budget": 3.00, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-039",
        input={"item": "HdMi CaBlE", "quantity": 50, "budget": 5.00, "deadline": "2026-09-20"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    # ── Unknown items — no matches ──
    EvalScenario(
        scenario_id="sourcing-040",
        input={"item": "SATA cable", "quantity": 1000, "budget": 1.00, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-041",
        input={"item": "Thunderbolt cable", "quantity": 10, "budget": 100.00, "deadline": "2028-01-01"},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-042",
        input={"item": "DisplayPort cable", "quantity": 500, "budget": 5.00, "deadline": "2026-08-30"},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    # ── Impossible budgets across categories ──
    EvalScenario(
        scenario_id="sourcing-043",
        input={"item": "USB-C cable", "quantity": 10000, "budget": 0.01, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-044",
        input={"item": "HDMI cable", "quantity": 1, "budget": 0.01, "deadline": "2026-08-01"},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-045",
        input={"item": "Ethernet cable", "quantity": 100, "budget": 1.19, "deadline": None},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    EvalScenario(
        scenario_id="sourcing-046",
        input={"item": "USB-C cable", "quantity": 50, "budget": 2.49, "deadline": "2026-09-10"},
        expected_tool_sequence=["query_suppliers"],
        expected_outcome={"status": "no_matches"},
    ),
    # ── Generous budgets, varied deadlines ──
    EvalScenario(
        scenario_id="sourcing-047",
        input={"item": "HDMI cable", "quantity": 200, "budget": 5.00, "deadline": "2027-01-15"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-048",
        input={"item": "Ethernet cable", "quantity": 10, "budget": 2.00, "deadline": "2026-08-10"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-049",
        input={"item": "Power adapter", "quantity": 1000, "budget": 10.00, "deadline": "2027-06-01"},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
    EvalScenario(
        scenario_id="sourcing-050",
        input={"item": "USB-C cable", "quantity": 100, "budget": 3.10, "deadline": None},
        expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
        expected_outcome={"status": "completed"},
    ),
]
