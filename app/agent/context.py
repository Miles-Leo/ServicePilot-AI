"""Backend-only invocation context; never derive identity from model messages."""

from dataclasses import dataclass, field

from business.services import OrderService
from business.ticket_confirmation import TicketPreparer


@dataclass(frozen=True)
class AgentContext:
    current_user_id: str | None = None
    order_service: OrderService | None = field(default=None, repr=False, compare=False)
    ticket_preparer: TicketPreparer | None = field(default=None, repr=False, compare=False)
