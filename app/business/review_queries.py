"""Authorized read-only staff views, independent of Streamlit and Agent."""

import logging

from sqlalchemy.exc import SQLAlchemyError

from business.models import TicketStatus, UserRole
from business.repositories import OrderRepository, TicketRepository
from business.schemas import ReviewDetailResponse, ReviewListResponse
from business.services import ORDER_NO_PATTERN, _utc_iso

logger = logging.getLogger(__name__)


def _failure(code, message, retryable=False):
    return {"ok": False, "data": None, "error": {
        "code": code, "message": message, "retryable": retryable,
    }}


def _view(row):
    ticket, order_no = row
    return dict(ticket_no=ticket.ticket_no, order_no=order_no, type=ticket.type.value,
                description=ticket.description, created_at=_utc_iso(ticket.created_at),
                status=ticket.status.value, reviewed_at=_utc_iso(ticket.reviewed_at),
                review_note=ticket.review_note)


class TicketReviewQueryService:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def list_tickets(self, *, current_reviewer_id, status="pending_review", limit=100) -> ReviewListResponse:
        if (type(limit) is not int or not 1 <= limit <= 100
                or (status is not None and (not isinstance(status, str) or status not in {s.value for s in TicketStatus}))):
            return _failure("INVALID_ARGUMENT", "列表状态或数量限制不合法。")
        return self._read(current_reviewer_id, status=status, limit=limit)

    def get_ticket(self, *, current_reviewer_id, ticket_no) -> ReviewDetailResponse:
        if not isinstance(ticket_no, str) or ORDER_NO_PATTERN.fullmatch(ticket_no) is None:
            return _failure("INVALID_ARGUMENT", "工单号不合法。")
        return self._read(current_reviewer_id, ticket_no=ticket_no)

    def _read(self, identity, *, status=None, limit=100, ticket_no=None):
        if (not isinstance(identity, str) or not identity.strip()
                or identity != identity.strip() or len(identity) > 36):
            return _failure("UNAUTHENTICATED", "审核身份无效，请重新登录。")
        try:
            with self.session_factory() as session:
                user = OrderRepository(session).get_user(identity)
                if user is None or not user.is_active:
                    return _failure("UNAUTHENTICATED", "审核身份无效，请重新登录。")
                if user.role is not UserRole.STAFF:
                    return _failure("FORBIDDEN", "当前账号没有工单审核权限。")
                repository = TicketRepository(session)
                if ticket_no is not None:
                    row = repository.detail_for_review(ticket_no)
                    if row is None:
                        return _failure("TICKET_NOT_FOUND", "未找到该工单。")
                    data = _view(row)
                else:
                    data = [_view(row) for row in repository.list_for_review(
                        status=TicketStatus(status) if status is not None else None, limit=limit,
                    )]
                return {"ok": True, "data": data, "error": None}
        except SQLAlchemyError:
            logger.error("Ticket review query failed; details omitted")
            return _failure("TEMPORARY_UNAVAILABLE", "审核列表暂时不可用，请稍后重试。", True)
