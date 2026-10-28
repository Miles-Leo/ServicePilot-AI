"""JSON-ready, explicitly allowlisted order-query response types."""

from typing import Literal, TypedDict


ErrorCode = Literal[
    "INVALID_ARGUMENT", "UNAUTHENTICATED", "ORDER_NOT_FOUND", "TEMPORARY_UNAVAILABLE",
    "TICKET_TYPE_NOT_ALLOWED", "CONFIRMATION_REQUIRED", "IDEMPOTENCY_CONFLICT", "ACTIVE_TICKET_EXISTS",
    "PREPARATION_EXPIRED",
    "FORBIDDEN", "TICKET_NOT_FOUND", "INVALID_TICKET_STATUS",
]


class ErrorDetail(TypedDict):
    code: ErrorCode
    message: str
    retryable: bool


class ProductSnapshot(TypedDict):
    product_id: str
    name: str
    sku: str
    unit_price_cents: int
    quantity: int


class OrderData(TypedDict):
    order_no: str
    status: str
    product: ProductSnapshot
    total_amount_cents: int
    currency: str
    created_at: str
    paid_at: str | None
    shipped_at: str | None
    completed_at: str | None
    available_ticket_types: list[str]


class OrderQueryResponse(TypedDict):
    ok: bool
    data: OrderData | None
    error: ErrorDetail | None


class TicketData(TypedDict):
    ticket_no: str
    order_no: str
    type: str
    status: str
    created_at: str
    created: bool


class TicketResponse(TypedDict):
    ok: bool
    data: TicketData | None
    error: ErrorDetail | None


class PreparedTicketData(TypedDict):
    draft_id: str
    order_no: str
    type: str
    description: str
    expires_at: str
    requires_confirmation: bool
    notice: str


class PreparedTicketResponse(TypedDict):
    ok: bool
    data: PreparedTicketData | None
    error: ErrorDetail | None


class TicketReviewData(TypedDict):
    ticket_no: str
    status: str
    reviewed_by: str
    reviewed_at: str
    review_note: str


class TicketReviewResponse(TypedDict):
    ok: bool
    data: TicketReviewData | None
    error: ErrorDetail | None


class ReviewTicketView(TypedDict):
    ticket_no: str
    order_no: str
    type: str
    description: str
    created_at: str
    status: str
    reviewed_at: str | None
    review_note: str | None


class ReviewListResponse(TypedDict):
    ok: bool
    data: list[ReviewTicketView] | None
    error: ErrorDetail | None


class ReviewDetailResponse(TypedDict):
    ok: bool
    data: ReviewTicketView | None
    error: ErrorDetail | None
