"""Append-only synthetic DEMO seeding, including legacy files and collisions."""

from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest
from sqlalchemy import func, select

from business.database import Base, create_session_factory, create_sqlite_engine
from business.demo import (
    DEMO_AFTERSALES_ORDERS, DEMO_USERS, DemoDataConflict, initialize_demo_database,
    initialize_or_supplement_demo_database, supplement_demo_database,
)
from business.models import Order, OrderStatus, Product, Ticket, TicketType, User
from business.services import OrderService


def snapshot(path):
    engine = create_sqlite_engine(path)
    try:
        with create_session_factory(engine)() as session:
            return {
                order.order_no: {column.name: getattr(order, column.name) for column in Order.__table__.columns}
                for order in session.scalars(select(Order))
            }
    finally:
        engine.dispose()


def test_new_initialization_and_repeat_are_idempotent(tmp_path):
    path = tmp_path / "new.sqlite3"
    assert initialize_or_supplement_demo_database(path) == 2
    before = snapshot(path)
    assert initialize_or_supplement_demo_database(path) == 0
    assert snapshot(path) == before
    engine = create_sqlite_engine(path)
    try:
        factory = create_session_factory(engine)
        with factory() as session:
            for model, number in [(User, 2), (Product, 1), (Order, 4), (Ticket, 0)]:
                assert session.scalar(select(func.count()).select_from(model)) == number
        for user in DEMO_USERS.values():
            service = OrderService(factory)
            eligible = service.get_order(current_user_id=user.user_id, order_no=DEMO_AFTERSALES_ORDERS[user.user_id].order_no)
            assert eligible["data"]["available_ticket_types"] == ["return_request", "exchange_request", "repair_request"]
            assert eligible["data"]["status"] == DEMO_AFTERSALES_ORDERS[user.user_id].status.value
            assert eligible["data"]["total_amount_cents"] == 19900
            assert service.get_order(current_user_id=user.user_id, order_no=user.order_no)["data"]["status"] == "pending_payment"
    finally:
        engine.dispose()


def test_legacy_database_is_supplemented_without_changing_originals(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    initialize_demo_database(path)
    originals = snapshot(path)
    assert supplement_demo_database(path) == 2
    after = snapshot(path)
    assert {key: after[key] for key in originals} == originals


def test_repeat_preserves_existing_order_changes(tmp_path):
    path = tmp_path / "edited.sqlite3"
    initialize_or_supplement_demo_database(path)
    engine = create_sqlite_engine(path)
    try:
        with create_session_factory(engine).begin() as session:
            order = session.scalar(select(Order).where(Order.order_no == "DEMO-AFTERSALE-A"))
            order.status = OrderStatus.CANCELLED
            order.product_name_snapshot = "Synthetic edited snapshot"
        before = snapshot(path)
        assert supplement_demo_database(path) == 0
        assert snapshot(path) == before
    finally:
        engine.dispose()


def test_legacy_ticket_and_original_status_are_preserved(tmp_path):
    path = tmp_path / "legacy-ticket.sqlite3"
    initialize_demo_database(path)
    engine = create_sqlite_engine(path)
    try:
        factory = create_session_factory(engine)
        with factory.begin() as session:
            order = session.scalar(select(Order).where(Order.order_no == "DEMO-ORDER-A"))
            order.status = OrderStatus.COMPLETED
            session.add(Ticket(
                ticket_no="SYNTHETIC-EXISTING-TICKET", user_id=order.user_id, order_id=order.id,
                type=TicketType.REPAIR_REQUEST, description="Synthetic preexisting request",
                idempotency_key="synthetic-existing-key", request_fingerprint="a" * 64,
            ))
        before = snapshot(path)
        assert supplement_demo_database(path) == 2
        after = snapshot(path)
        assert {key: after[key] for key in before} == before
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(Ticket)) == 1
            ticket = session.scalar(select(Ticket))
            assert ticket.ticket_no == "SYNTHETIC-EXISTING-TICKET"
            assert ticket.description == "Synthetic preexisting request"
    finally:
        engine.dispose()


def test_missing_supplement_does_not_create_file(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError): supplement_demo_database(path)
    assert not path.exists()


def test_arbitrary_existing_file_not_overwritten(tmp_path):
    path = tmp_path / "existing.sqlite3"
    path.write_bytes(b"Unrelated synthetic file - preserve")
    before = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(DemoDataConflict): initialize_or_supplement_demo_database(path)
    assert hashlib.sha256(path.read_bytes()).digest() == before


def test_unknown_business_database_not_seeded(tmp_path):
    path = tmp_path / "unknown.sqlite3"
    engine = create_sqlite_engine(path)
    try:
        Base.metadata.create_all(engine)
        with pytest.raises(DemoDataConflict): supplement_demo_database(path)
        with create_session_factory(engine)() as session:
            assert session.scalar(select(func.count()).select_from(Order)) == 0
            assert session.scalar(select(func.count()).select_from(User)) == 0
    finally:
        engine.dispose()


@pytest.mark.parametrize("conflict", ["number", "id", "product", "user"])
def test_collisions_abort_all_supplement_writes(tmp_path, conflict):
    path = tmp_path / "conflict.sqlite3"
    initialize_demo_database(path)
    engine = create_sqlite_engine(path)
    try:
        with create_session_factory(engine).begin() as session:
            user_b = list(DEMO_USERS.values())[1]
            if conflict == "product":
                session.scalar(select(Product)).sku = "SYNTHETIC-OTHER-SKU"
            elif conflict == "user":
                session.get(User, user_b.user_id).auth_subject = "synthetic-other-auth"
            else:
                fixture = DEMO_AFTERSALES_ORDERS[user_b.user_id]
                session.add(Order(
                    id=fixture.id if conflict == "id" else "synthetic-existing-id",
                    order_no=fixture.order_no if conflict == "number" else "SYNTHETIC-EXISTING",
                    user_id=user_b.user_id, product_id=session.scalar(select(Product.id)),
                    product_name_snapshot="Synthetic existing", sku_snapshot="SYNTHETIC",
                    unit_price_cents=1, quantity=1, total_amount_cents=1,
                ))
        before = snapshot(path)
        with pytest.raises(DemoDataConflict): supplement_demo_database(path)
        assert snapshot(path) == before
    finally:
        engine.dispose()


def test_concurrent_supplement_does_not_duplicate(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    initialize_demo_database(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        added = list(pool.map(supplement_demo_database, [path, path]))
    assert sorted(added) == [0, 2]
    assert len(snapshot(path)) == 4
