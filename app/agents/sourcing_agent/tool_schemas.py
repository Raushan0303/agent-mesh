from pydantic import BaseModel, Field


# ── query_suppliers ──


class QuerySuppliersInput(BaseModel):
    item: str = Field(..., description="The item to source, e.g., 'USB-C cable'")
    budget: float = Field(..., gt=0, description="Max price per unit")
    quantity: int = Field(..., gt=0, description="Number of units needed")


class SupplierResult(BaseModel):
    name: str
    item: str
    price: float
    lead_time_days: int
    rating: float


class QuerySuppliersOutput(BaseModel):
    suppliers: list[SupplierResult]


# ── get_price_quote ──


class GetPriceQuoteInput(BaseModel):
    supplier_name: str = Field(..., description="Name of the supplier")
    item: str = Field(..., description="The item to quote")
    quantity: int = Field(..., gt=0, description="Number of units")


class GetPriceQuoteOutput(BaseModel):
    supplier_name: str
    item: str
    unit_price: float
    total_price: float
    lead_time_days: int
    in_stock: bool


# ── check_seller_rating ──


class CheckSellerRatingInput(BaseModel):
    supplier_name: str = Field(..., description="Name of the supplier")


class CheckSellerRatingOutput(BaseModel):
    supplier_name: str
    rating: float
    total_orders: int
    on_time_rate: float


# ── create_purchase_order ──


class CreatePurchaseOrderInput(BaseModel):
    supplier_name: str
    item: str
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)


class CreatePurchaseOrderOutput(BaseModel):
    po_id: str
    supplier_name: str
    status: str  # "created" | "failed"


# ── initiate_payment ──


class InitiatePaymentInput(BaseModel):
    po_id: str
    amount: float = Field(..., gt=0)


class InitiatePaymentOutput(BaseModel):
    payment_id: str
    po_id: str
    status: str  # "initiated" | "failed"
