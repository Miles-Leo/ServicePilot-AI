"""Parameterized persistence queries; no authentication or response formatting."""

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from business.models import Order, Ticket, TicketStatus, TicketType, User


class OrderRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_user(self, current_user_id: str) -> User | None:
        return self.session.scalar(select(User).where(User.id == current_user_id))

    def get_by_order_no_for_user(
        self, *, order_no: str, current_user_id: str
    ) -> Order | None:
        # Always scope in SQL, rather than loading another user's order first.
        return self.session.scalar(
            select(Order).where(
                Order.order_no == order_no,
                Order.user_id == current_user_id,
            )
        )


class TicketRepository:
    def __init__(self, session: Session):
        self.session = session

    def begin_sqlite_write(self) -> None:
        """Acquire SQLite's writer reservation before read/validate/insert."""
        self.session.execute(text("BEGIN IMMEDIATE"))

    def get_by_idempotency_key(self, *, user_id: str, key: str) -> Ticket | None:
        return self.session.scalar(select(Ticket).where(
            Ticket.user_id == user_id, Ticket.idempotency_key == key,
        ))

    def get_active(self, *, user_id: str, order_id: str, ticket_type: TicketType) -> Ticket | None:
        return self.session.scalar(select(Ticket).where(
            Ticket.user_id == user_id, Ticket.order_id == order_id, Ticket.type == ticket_type,
            Ticket.status.in_([
                TicketStatus.PENDING_REVIEW, TicketStatus.APPROVED, TicketStatus.IN_PROGRESS,
            ]),
        ))

    def add(self, ticket: Ticket) -> None:
        self.session.add(ticket)
        self.session.flush()  # Enforce constraints before constructing the response.

    def get_by_ticket_no(self, ticket_no: str) -> Ticket | None:
        return self.session.scalar(select(Ticket).where(Ticket.ticket_no == ticket_no))

    def list_for_review(self, *, status: TicketStatus | None, limit: int):
        query = select(Ticket, Order.order_no).join(Order, Ticket.order_id == Order.id)
        if status is not None:
            query = query.where(Ticket.status == status)
        return self.session.execute(query.order_by(Ticket.created_at.asc(), Ticket.ticket_no.asc()).limit(limit)).all()

    def detail_for_review(self, ticket_no: str):
        return self.session.execute(select(Ticket, Order.order_no).join(
            Order, Ticket.order_id == Order.id,
        ).where(Ticket.ticket_no == ticket_no)).first()

    def review_if_pending(self, *, ticket_no, status, reviewer_id, note, reviewed_at) -> bool:
        """Compare-and-set: an established or inconsistent review is never overwritten."""
        result = self.session.execute(
            update(Ticket).where(
                Ticket.ticket_no == ticket_no,
                Ticket.status == TicketStatus.PENDING_REVIEW,
                Ticket.reviewed_by.is_(None), Ticket.reviewed_at.is_(None),
                Ticket.review_note.is_(None),
            ).values(
                status=status, reviewed_by=reviewer_id, review_note=note,
                reviewed_at=reviewed_at, updated_at=reviewed_at,
            ).execution_options(synchronize_session=False)
        )
        return result.rowcount == 1
