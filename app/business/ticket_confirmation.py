"""Process-local two-stage workflow. Only backend button handlers may submit.

Session state contains an opaque handle, never authoritative draft fields or a
confirmed flag. Restart/expiry loses drafts safely; ticket idempotency is durable.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import secrets
from threading import RLock

from sqlalchemy.orm import Session

from business.models import TicketType
from business.schemas import PreparedTicketResponse, TicketResponse
from business.services import OrderService, TicketService


logger = logging.getLogger("agent")
DRAFT_TTL = timedelta(minutes=10)
SESSION_TTL = timedelta(hours=2)
NOTICE = "提交后进入人工审核，不代表退款成功。"


def _failure(code, message, retryable=False):
    return {"ok": False, "data": None, "error": {
        "code": code, "message": message, "retryable": retryable,
    }}


@dataclass(frozen=True)
class _Draft:
    draft_id: str
    user_id: str
    order_no: str
    ticket_type: str
    description: str
    expires_at: datetime
    idempotency_key: str = field(repr=False)

    def summary(self):
        return {
            "draft_id": self.draft_id, "order_no": self.order_no, "type": self.ticket_type,
            "description": self.description, "expires_at": self.expires_at.isoformat(),
            "requires_confirmation": True, "notice": NOTICE,
        }


class TicketPreparer:
    """Narrow runtime capability: prepare only, no create/confirm method."""

    def __init__(self, prepare: Callable):
        self.__prepare = prepare

    def prepare(self, *, current_user_id, order_no, ticket_type, description) -> PreparedTicketResponse:
        return self.__prepare(
            current_user_id=current_user_id, order_no=order_no,
            ticket_type=ticket_type, description=description,
        )


class TicketConfirmation:
    def __init__(
        self, user_id: str, session_factory: Callable[[], Session], *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.__user_id = user_id
        self.__orders = OrderService(session_factory)
        self.__tickets = TicketService(session_factory)
        self.__clock = clock
        self.__draft: _Draft | None = None
        self.__result: TicketResponse | None = None
        self.__completed_draft_id: str | None = None
        self.__revoked = False
        self.__lock = RLock()
        self.preparer = TicketPreparer(self.__prepare)

    def belongs_to(self, user_id: str) -> bool:
        return not self.__revoked and isinstance(user_id, str) and user_id == self.__user_id

    def invalidate(self) -> None:
        with self.__lock:
            self.__draft = None
            self.__result = None
            self.__completed_draft_id = None

    def revoke(self) -> None:
        """Permanently revoke capabilities from a discarded/switched session."""
        with self.__lock:
            self.__revoked = True
            self.invalidate()

    def __prepare(self, *, current_user_id, order_no, ticket_type, description):
        with self.__lock:
            # Even an invalid attempted edit invalidates the earlier proposal.
            self.invalidate()
            if not self.belongs_to(current_user_id):
                return _failure("UNAUTHENTICATED", "登录身份无效，请重新登录。")
            if not isinstance(ticket_type, str) or ticket_type not in {kind.value for kind in TicketType}:
                return _failure("INVALID_ARGUMENT", "售后申请类型不合法。")
            if not isinstance(description, str) or "\x00" in description or not 1 <= len(description.strip()) <= 2000:
                return _failure("INVALID_ARGUMENT", "问题描述须为1–2000个字符。")
            try:
                description.encode("utf-8")
                order = self.__orders.get_order(current_user_id=current_user_id, order_no=order_no)
                if not order["ok"]:
                    return order
                if ticket_type not in order["data"]["available_ticket_types"]:
                    return _failure("TICKET_TYPE_NOT_ALLOWED", "当前订单状态不允许该售后申请类型。")
                self.__draft = _Draft(
                    draft_id=secrets.token_hex(16), user_id=current_user_id,
                    order_no=order_no, ticket_type=ticket_type, description=description.strip(),
                    expires_at=self.__clock() + DRAFT_TTL, idempotency_key=secrets.token_hex(32),
                )
                return {"ok": True, "data": self.__draft.summary(), "error": None}
            except UnicodeError:
                return _failure("INVALID_ARGUMENT", "问题描述包含无效字符。")
            except Exception:
                logger.error("ticket_prepare_failed details_omitted")
                return _failure("TEMPORARY_UNAVAILABLE", "申请准备暂时不可用，请稍后重试。", True)

    def pending(self, current_user_id: str):
        with self.__lock:
            if not self.belongs_to(current_user_id):
                self.invalidate()
                return None
            if self.__draft is not None and self.__clock() >= self.__draft.expires_at:
                self.__draft = None
                self.__result = _failure("PREPARATION_EXPIRED", "申请已过期，请重新准备并确认。")
            # Return a detached dictionary, never the authoritative immutable record.
            return self.__draft.summary() if self.__draft is not None else None

    def result(self, current_user_id: str):
        with self.__lock:
            if not self.belongs_to(current_user_id) or self.__result is None:
                return None
            return {
                "ok": self.__result["ok"],
                "data": dict(self.__result["data"]) if self.__result["data"] is not None else None,
                "error": dict(self.__result["error"]) if self.__result["error"] is not None else None,
            }

    def confirm_from_button(self, *, current_user_id: str, draft_id: str) -> TicketResponse:
        """BACKEND ONLY: call solely in the branch of a real confirmation button event."""
        with self.__lock:
            if self.belongs_to(current_user_id) and draft_id == self.__completed_draft_id and self.__result is not None:
                return self.result(current_user_id)
            summary = self.pending(current_user_id)
            if summary is None or summary["draft_id"] != draft_id:
                result = self.result(current_user_id)
                if result is not None and result["error"] is not None and result["error"]["code"] == "PREPARATION_EXPIRED":
                    return result
                return _failure("CONFIRMATION_REQUIRED", "申请已失效，请重新准备并点击确认提交。")
            draft = self.__draft
            assert draft is not None
            try:
                # All fields come from the backend record, never session/widget data.
                logger.info("ticket_confirmation_submit_requested")
                self.__result = self.__tickets.create_ticket(
                    current_user_id=current_user_id, order_no=draft.order_no,
                    ticket_type=draft.ticket_type, description=draft.description,
                    idempotency_key=draft.idempotency_key, confirmed=True,
                )
            except Exception:
                logger.error("ticket_confirmation_failed details_omitted")
                self.__result = _failure("TEMPORARY_UNAVAILABLE", "售后申请暂时不可用，请稍后重试。", True)
            if self.__result["ok"]:
                logger.info("ticket_confirmation_submit_succeeded status=pending_review")
                self.__completed_draft_id = draft.draft_id
                self.__draft = None
            return self.result(current_user_id)

    def cancel_from_button(self, *, current_user_id: str, draft_id: str) -> None:
        with self.__lock:
            if self.belongs_to(current_user_id) and self.__draft is not None and self.__draft.draft_id == draft_id:
                self.invalidate()


# Private process-local registry. Nothing here is a model tool or widget payload.
_sessions: dict[str, tuple[TicketConfirmation, datetime]] = {}
_sessions_lock = RLock()


def discard_ticket_session(handle) -> None:
    with _sessions_lock:
        entry = _sessions.pop(handle, None) if isinstance(handle, str) else None
        if entry is not None:
            entry[0].revoke()


def ticket_session(handle, user_id: str, session_factory) -> tuple[str, TicketConfirmation]:
    now = datetime.now(timezone.utc)
    with _sessions_lock:
        for key, entry in list(_sessions.items()):
            if now - entry[1] >= SESSION_TTL:
                discard_ticket_session(key)
        entry = _sessions.get(handle) if isinstance(handle, str) else None
        if entry is not None and entry[0].belongs_to(user_id):
            _sessions[handle] = (entry[0], now)
            return handle, entry[0]
        discard_ticket_session(handle)
        if len(_sessions) >= 1024:
            discard_ticket_session(min(_sessions, key=lambda key: _sessions[key][1]))
        handle = secrets.token_urlsafe(32)
        workflow = TicketConfirmation(user_id, session_factory)
        _sessions[handle] = (workflow, now)
        return handle, workflow
