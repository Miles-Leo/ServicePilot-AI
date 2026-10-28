"""Four core entities. State-transition authorization belongs to future services."""

from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import (
    Boolean, CheckConstraint, Enum as SAEnum, ForeignKey, ForeignKeyConstraint,
    Index, Integer, String, Text, UniqueConstraint, text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from business.database import Base, UTCDateTime, utc_now


class UserRole(str, Enum):
    CUSTOMER = "customer"
    STAFF = "staff"


class OrderStatus(str, Enum):
    PENDING_PAYMENT = "pending_payment"
    PAID = "paid"
    SHIPPED = "shipped"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TicketType(str, Enum):
    RETURN_REQUEST = "return_request"
    EXCHANGE_REQUEST = "exchange_request"
    REPAIR_REQUEST = "repair_request"


class TicketStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


def enum_column(enum_type: type[Enum], name: str) -> SAEnum:
    return SAEnum(
        enum_type, name=name, native_enum=False, create_constraint=True,
        validate_strings=True, values_callable=lambda members: [m.value for m in members],
    )


def new_id() -> str:
    return str(uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    auth_subject: Mapped[str] = mapped_column(String(255), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    role: Mapped[UserRole] = mapped_column(
        enum_column(UserRole, "user_role"), default=UserRole.CUSTOMER
    )
    is_active: Mapped[bool] = mapped_column(Boolean(create_constraint=True), default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    orders: Mapped[list["Order"]] = relationship(back_populates="user")
    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="user", foreign_keys="Ticket.user_id"
    )


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("typeof(price_cents) = 'integer'", name="ck_product_price_integer"),
        CheckConstraint("price_cents >= 0", name="ck_product_price_nonnegative"),
        CheckConstraint("currency = 'CNY'", name="ck_product_currency"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    sku: Mapped[str] = mapped_column(String(100), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    price_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="CNY")
    is_active: Mapped[bool] = mapped_column(Boolean(create_constraint=True), default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now)

    orders: Mapped[list["Order"]] = relationship(back_populates="product")


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("id", "user_id", name="uq_order_id_user"),
        CheckConstraint("typeof(unit_price_cents) = 'integer'", name="ck_order_unit_price_integer"),
        CheckConstraint("typeof(quantity) = 'integer'", name="ck_order_quantity_integer"),
        CheckConstraint("typeof(total_amount_cents) = 'integer'", name="ck_order_total_integer"),
        CheckConstraint("unit_price_cents >= 0", name="ck_order_unit_price_nonnegative"),
        CheckConstraint("quantity > 0", name="ck_order_quantity_positive"),
        CheckConstraint("total_amount_cents >= 0", name="ck_order_total_nonnegative"),
        CheckConstraint(
            "total_amount_cents = unit_price_cents * quantity", name="ck_order_total_matches"
        ),
        CheckConstraint("currency = 'CNY'", name="ck_order_currency"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_no: Mapped[str] = mapped_column(String(64), unique=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), index=True)
    product_name_snapshot: Mapped[str] = mapped_column(String(255))
    sku_snapshot: Mapped[str] = mapped_column(String(100))
    unit_price_cents: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)
    total_amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="CNY")
    status: Mapped[OrderStatus] = mapped_column(
        enum_column(OrderStatus, "order_status"), default=OrderStatus.PENDING_PAYMENT
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now)
    paid_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    shipped_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    user: Mapped[User] = relationship(back_populates="orders")
    product: Mapped[Product] = relationship(back_populates="orders")
    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="order", foreign_keys="Ticket.order_id"
    )


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name="uq_ticket_user_idempotency"),
        # A ticket cannot claim a different owner from its referenced order.
        ForeignKeyConstraint(
            ["order_id", "user_id"], ["orders.id", "orders.user_id"],
            name="fk_ticket_order_owner",
        ),
        CheckConstraint(
            "length(trim(description)) BETWEEN 1 AND 2000", name="ck_ticket_description"
        ),
        CheckConstraint("length(trim(idempotency_key)) > 0", name="ck_ticket_idempotency_key"),
        CheckConstraint("length(trim(request_fingerprint)) > 0", name="ck_ticket_fingerprint"),
        CheckConstraint(
            "status != 'rejected' OR "
            "(review_note IS NOT NULL AND length(trim(review_note)) > 0)",
            name="ck_ticket_rejection_note",
        ),
        CheckConstraint(
            "status != 'resolved' OR "
            "(resolution_note IS NOT NULL AND length(trim(resolution_note)) > 0)",
            name="ck_ticket_resolution_note",
        ),
        Index(
            "uq_ticket_active_order_type", "user_id", "order_id", "type", unique=True,
            sqlite_where=text("status IN ('pending_review', 'approved', 'in_progress')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    ticket_no: Mapped[str] = mapped_column(String(64), unique=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    order_id: Mapped[str] = mapped_column(String(36), index=True)
    type: Mapped[TicketType] = mapped_column(enum_column(TicketType, "ticket_type"))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[TicketStatus] = mapped_column(
        enum_column(TicketStatus, "ticket_status"), default=TicketStatus.PENDING_REVIEW
    )
    idempotency_key: Mapped[str] = mapped_column(String(255))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    reviewed_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    review_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    resolution_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now)

    user: Mapped[User] = relationship(back_populates="tickets", foreign_keys=[user_id])
    order: Mapped[Order] = relationship(back_populates="tickets", foreign_keys=[order_id])
    reviewer: Mapped[User | None] = relationship(foreign_keys=[reviewed_by])
