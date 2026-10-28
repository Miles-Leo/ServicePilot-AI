"""Order queries and confirmed ticket requests, independent of Agent and UI."""

from collections.abc import Callable
from datetime import datetime, timezone
import logging
import hashlib
import json
import re
from uuid import uuid4

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from business.models import Order, OrderStatus, Ticket, TicketStatus, TicketType, UserRole
from business.repositories import OrderRepository, TicketRepository
from business.schemas import ErrorCode, OrderData, OrderQueryResponse, TicketResponse, TicketReviewResponse


logger = logging.getLogger(__name__)
ORDER_NO_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def _failure(code: ErrorCode, message: str, *, retryable: bool = False) -> OrderQueryResponse:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message, "retryable": retryable},
    }


def _available_ticket_types(status: OrderStatus) -> list[str]:
    if status == OrderStatus.PAID:
        return [TicketType.RETURN_REQUEST.value]
    if status in (OrderStatus.SHIPPED, OrderStatus.COMPLETED):
        return [
            TicketType.RETURN_REQUEST.value,
            TicketType.EXCHANGE_REQUEST.value,
            TicketType.REPAIR_REQUEST.value,
        ]
    return []


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _order_data(order: Order) -> OrderData:
    # Build a whitelist instead of serializing an ORM object or relationships.
    return {
        "order_no": order.order_no,
        "status": order.status.value,
        "product": {
            "product_id": order.product_id,
            "name": order.product_name_snapshot,
            "sku": order.sku_snapshot,
            "unit_price_cents": order.unit_price_cents,
            "quantity": order.quantity,
        },
        "total_amount_cents": order.total_amount_cents,
        "currency": order.currency,
        "created_at": order.created_at.astimezone(timezone.utc).isoformat(),
        "paid_at": _utc_iso(order.paid_at),
        "shipped_at": _utc_iso(order.shipped_at),
        "completed_at": _utc_iso(order.completed_at),
        "available_ticket_types": _available_ticket_types(order.status),
    }


class OrderService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def get_order(self, *, current_user_id: str, order_no: str) -> OrderQueryResponse:
        """Identity MUST come from authenticated backend context, never the model.

        This method verifies account existence/availability, not login credentials.
        Staff receive no cross-user access through this customer-facing service.
        It owns a fresh read-only session and never commits application writes.
        """
        if (
            not isinstance(current_user_id, str)
            or not current_user_id.strip()
            or current_user_id != current_user_id.strip()
            or len(current_user_id) > 36
        ):
            return _failure("UNAUTHENTICATED", "登录身份无效，请重新登录。")
        if not isinstance(order_no, str) or ORDER_NO_PATTERN.fullmatch(order_no) is None:
            return _failure("INVALID_ARGUMENT", "订单号须为1–64位字母、数字、下划线或连字符，且以字母或数字开头。")

        try:
            with self.session_factory() as session:
                repository = OrderRepository(session)
                user = repository.get_user(current_user_id)
                if user is None or not user.is_active:
                    return _failure("UNAUTHENTICATED", "登录身份无效，请重新登录。")
                order = repository.get_by_order_no_for_user(
                    order_no=order_no, current_user_id=current_user_id
                )
                if order is None:
                    return _failure("ORDER_NOT_FOUND", "未找到该订单。")
                data = _order_data(order)
            return {"ok": True, "data": data, "error": None}
        except SQLAlchemyError as exc:
            # Exception text/tracebacks may contain SQL, parameters and local paths.
            logger.error("Order query database failure (%s)", type(exc).__name__)
            return _failure("TEMPORARY_UNAVAILABLE", "订单查询暂时不可用，请稍后重试。", retryable=True)


def _ticket_failure(code: ErrorCode, message: str, *, retryable: bool = False) -> TicketResponse:
    return {"ok": False, "data": None, "error": {
        "code": code, "message": message, "retryable": retryable,
    }}


def _ticket_response(ticket: Ticket, order_no: str, *, created: bool) -> TicketResponse:
    return {"ok": True, "data": {
        "ticket_no": ticket.ticket_no, "order_no": order_no, "type": ticket.type.value,
        "status": ticket.status.value,
        "created_at": ticket.created_at.astimezone(timezone.utc).isoformat(), "created": created,
    }, "error": None}


def _ticket_fingerprint(order_no: str, ticket_type: TicketType, description: str) -> str:
    payload = json.dumps({
        "version": 1, "order_no": order_no, "type": ticket_type.value, "description": description,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TicketService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def create_ticket(
        self, *, current_user_id: str, order_no: str, ticket_type: str,
        description: str, idempotency_key: str, confirmed: bool,
    ) -> TicketResponse:
        """Backend supplies identity, confirmation and a stable per-submission key.

        Only records a pending-review request; no payment or fulfillment occurs.
        No writes (or database connection) occur without literal boolean True.
        """
        if (
            not isinstance(current_user_id, str) or not current_user_id.strip()
            or current_user_id != current_user_id.strip() or len(current_user_id) > 36
        ):
            return _ticket_failure("UNAUTHENTICATED", "登录身份无效，请重新登录。")
        if not isinstance(order_no, str) or ORDER_NO_PATTERN.fullmatch(order_no) is None:
            return _ticket_failure("INVALID_ARGUMENT", "订单号格式不合法。")
        if not isinstance(ticket_type, str):
            return _ticket_failure("INVALID_ARGUMENT", "售后申请类型不合法。")
        try:
            kind = TicketType(ticket_type)
        except ValueError:
            return _ticket_failure("INVALID_ARGUMENT", "售后申请类型不合法。")
        if not isinstance(description, str) or "\x00" in description or not 1 <= len(description.strip()) <= 2000:
            return _ticket_failure("INVALID_ARGUMENT", "问题描述须为1–2000个字符。")
        try:
            description.encode("utf-8")
        except UnicodeError:
            return _ticket_failure("INVALID_ARGUMENT", "问题描述包含无效字符。")
        if not isinstance(idempotency_key, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}", idempotency_key
        ) is None:
            return _ticket_failure("INVALID_ARGUMENT", "提交请求标识格式不合法。")
        if confirmed is not True:
            return _ticket_failure("CONFIRMATION_REQUIRED", "请明确确认后再提交售后申请。")

        normalized_description = description.strip()
        fingerprint = _ticket_fingerprint(order_no, kind, normalized_description)
        try:
            with self.session_factory() as session, session.begin():
                tickets = TicketRepository(session)
                tickets.begin_sqlite_write()
                failure, order = self._authorized_order(session, current_user_id, order_no)
                if failure is not None:
                    return failure
                assert order is not None
                existing = tickets.get_by_idempotency_key(user_id=current_user_id, key=idempotency_key)
                if existing is not None:
                    return self._replay(existing, order_no, fingerprint)
                if kind.value not in _available_ticket_types(order.status):
                    return _ticket_failure("TICKET_TYPE_NOT_ALLOWED", "当前订单状态不允许该售后申请类型。")
                if tickets.get_active(user_id=current_user_id, order_id=order.id, ticket_type=kind):
                    return _ticket_failure("ACTIVE_TICKET_EXISTS", "该订单已有同类型的未结束售后工单。")
                ticket = Ticket(
                    ticket_no=f"SPT-{uuid4().hex.upper()}", user_id=current_user_id, order_id=order.id,
                    type=kind, description=normalized_description, status=TicketStatus.PENDING_REVIEW,
                    idempotency_key=idempotency_key, request_fingerprint=fingerprint,
                )
                tickets.add(ticket)
                response = _ticket_response(ticket, order_no, created=True)
            # The transaction (including commit) must succeed before reporting creation.
            return response
        except IntegrityError:
            # The context manager rolled back. Re-read durable records in a fresh
            # session; never parse or return SQLite's exception text.
            return self._resolve_constraint_conflict(current_user_id, order_no, kind, idempotency_key, fingerprint)
        except SQLAlchemyError:
            return self._unavailable()

    @staticmethod
    def _authorized_order(
        session: Session, user_id: str, order_no: str
    ) -> tuple[TicketResponse | None, Order | None]:
        orders = OrderRepository(session)
        user = orders.get_user(user_id)
        if user is None or not user.is_active:
            return _ticket_failure("UNAUTHENTICATED", "登录身份无效，请重新登录。"), None
        order = orders.get_by_order_no_for_user(order_no=order_no, current_user_id=user_id)
        if order is None:
            return _ticket_failure("ORDER_NOT_FOUND", "未找到该订单。"), None
        return None, order

    @staticmethod
    def _replay(ticket: Ticket, order_no: str, fingerprint: str) -> TicketResponse:
        if ticket.request_fingerprint != fingerprint:
            return _ticket_failure("IDEMPOTENCY_CONFLICT", "同一提交请求标识不能用于不同申请内容。")
        return _ticket_response(ticket, order_no, created=False)

    def _resolve_constraint_conflict(
        self, user_id: str, order_no: str, kind: TicketType, key: str, fingerprint: str
    ) -> TicketResponse:
        try:
            with self.session_factory() as session:
                failure, order = self._authorized_order(session, user_id, order_no)
                if failure is not None:
                    return failure
                assert order is not None
                tickets = TicketRepository(session)
                existing = tickets.get_by_idempotency_key(user_id=user_id, key=key)
                if existing is not None:
                    return self._replay(existing, order_no, fingerprint)
                if tickets.get_active(user_id=user_id, order_id=order.id, ticket_type=kind):
                    return _ticket_failure("ACTIVE_TICKET_EXISTS", "该订单已有同类型的未结束售后工单。")
        except SQLAlchemyError:
            return self._unavailable()
        return self._unavailable()

    @staticmethod
    def _unavailable() -> TicketResponse:
        logger.error("Ticket request database failure; details omitted")
        return _ticket_failure("TEMPORARY_UNAVAILABLE", "售后申请暂时不可用，请稍后重试。", retryable=True)


def _review_failure(code: ErrorCode, message: str, *, retryable: bool = False) -> TicketReviewResponse:
    return {"ok": False, "data": None, "error": {
        "code": code, "message": message, "retryable": retryable,
    }}


class TicketReviewService:
    """One final human review; trusted backend supplies the reviewer identity.

    This is role authorization, not a login/authentication implementation.
    No payment or fulfillment operation is performed here.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def review_ticket(
        self, *, current_reviewer_id: str, ticket_no: str, decision: str, review_note: str,
    ) -> TicketReviewResponse:
        if (
            not isinstance(current_reviewer_id, str) or not current_reviewer_id.strip()
            or current_reviewer_id != current_reviewer_id.strip() or len(current_reviewer_id) > 36
        ):
            return _review_failure("UNAUTHENTICATED", "审核身份无效，请重新登录。")
        if (
            not isinstance(ticket_no, str) or ORDER_NO_PATTERN.fullmatch(ticket_no) is None
            or not isinstance(decision, str) or decision not in {"approve", "reject"}
            or not isinstance(review_note, str) or "\x00" in review_note
            or not 1 <= len(review_note.strip()) <= 2000
        ):
            return _review_failure("INVALID_ARGUMENT", "请提供有效工单号、审核决定及1–2000字审核意见。")
        try:
            review_note.encode("utf-8")
        except UnicodeError:
            return _review_failure("INVALID_ARGUMENT", "审核意见包含无效字符。")
        note = review_note.strip()
        status = TicketStatus.APPROVED if decision == "approve" else TicketStatus.REJECTED
        try:
            with self.session_factory() as session, session.begin():
                tickets = TicketRepository(session)
                # Serialize SQLite writers, including concurrent account changes.
                tickets.begin_sqlite_write()
                reviewer = OrderRepository(session).get_user(current_reviewer_id)
                if reviewer is None or not reviewer.is_active:
                    return _review_failure("UNAUTHENTICATED", "审核身份无效，请重新登录。")
                if reviewer.role is not UserRole.STAFF:
                    return _review_failure("FORBIDDEN", "当前账号没有工单审核权限。")
                if tickets.get_by_ticket_no(ticket_no) is None:
                    return _review_failure("TICKET_NOT_FOUND", "未找到该工单。")
                reviewed_at = datetime.now(timezone.utc)
                if not tickets.review_if_pending(
                    ticket_no=ticket_no, status=status, reviewer_id=current_reviewer_id,
                    note=note, reviewed_at=reviewed_at,
                ):
                    return _review_failure("INVALID_TICKET_STATUS", "工单已审核或不处于待审核状态，不能重复审核。")
                response: TicketReviewResponse = {"ok": True, "data": {
                    "ticket_no": ticket_no, "status": status.value,
                    "reviewed_by": current_reviewer_id, "reviewed_at": reviewed_at.isoformat(),
                    "review_note": note,
                }, "error": None}
            # Return success only after transaction commit has succeeded.
            return response
        except SQLAlchemyError:
            logger.error("Ticket review database failure; details omitted")
            return _review_failure("TEMPORARY_UNAVAILABLE", "工单审核暂时不可用，请稍后重试。", retryable=True)
