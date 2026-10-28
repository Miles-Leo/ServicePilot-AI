"""Explicit local DEMO only. Fixed synthetic identities are not authentication."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from business.database import Base, PROJECT_ROOT, create_session_factory, create_sqlite_engine, get_database_path
from business.models import Order, OrderStatus, Product, User, UserRole
from business.services import OrderService


@dataclass(frozen=True)
class DemoUser:
    user_id: str
    order_no: str


DEMO_USERS = MappingProxyType({
    "DEMO 用户 A（虚构）": DemoUser("00000000-0000-4000-8000-000000000001", "DEMO-ORDER-A"),
    "DEMO 用户 B（虚构）": DemoUser("00000000-0000-4000-8000-000000000002", "DEMO-ORDER-B"),
})
DEMO_DATABASE_PATH = PROJECT_ROOT / "servicepilot_demo.sqlite3"
DEMO_STAFF = MappingProxyType({"DEMO 员工（虚构）": "00000000-0000-4000-8000-000000000003"})


@dataclass(frozen=True)
class DemoAfterSalesOrder:
    id: str
    order_no: str
    status: OrderStatus


DEMO_AFTERSALES_ORDERS = MappingProxyType({
    DEMO_USERS["DEMO 用户 A（虚构）"].user_id: DemoAfterSalesOrder(
        "00000000-0000-4000-8000-000000000021", "DEMO-AFTERSALE-A", OrderStatus.COMPLETED,
    ),
    DEMO_USERS["DEMO 用户 B（虚构）"].user_id: DemoAfterSalesOrder(
        "00000000-0000-4000-8000-000000000022", "DEMO-AFTERSALE-B", OrderStatus.SHIPPED,
    ),
})


class DemoDataConflict(ValueError):
    """Existing files/records could not be verified as compatible synthetic data."""


def initialize_demo_database(database_path: Path = DEMO_DATABASE_PATH) -> None:
    """Explicitly seed a NEW dedicated file; never overwrite or update an existing DB."""
    database_path = get_database_path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with database_path.open("xb"):
        pass
    engine = create_sqlite_engine(database_path)
    try:
        Base.metadata.create_all(engine)
        with create_session_factory(engine).begin() as session:
            product = Product(
                id="00000000-0000-4000-8000-000000000010", sku="DEMO-SKU-001",
                name="DEMO 虚构商品", description="仅用于本地演示，不是真实商品。",
                price_cents=19900,
            )
            session.add(product)
            session.flush()
            for label, demo_user in DEMO_USERS.items():
                user = User(id=demo_user.user_id, auth_subject=f"demo:{demo_user.user_id}", display_name=label)
                session.add(user)
                session.flush()
                session.add(Order(
                    order_no=demo_user.order_no, user_id=user.id, product_id=product.id,
                    product_name_snapshot=product.name, sku_snapshot=product.sku,
                    unit_price_cents=19900, quantity=1, total_amount_cents=19900,
                    status=OrderStatus.PENDING_PAYMENT,
                ))
    finally:
        engine.dispose()


def create_demo_order_service(database_path: Path = DEMO_DATABASE_PATH) -> OrderService:
    if not Path(database_path).is_file():
        raise FileNotFoundError("Initialize the dedicated DEMO database first")
    return OrderService(create_session_factory(create_sqlite_engine(database_path)))


def supplement_demo_database(database_path: Path = DEMO_DATABASE_PATH) -> int:
    """Add missing fixed after-sales fixtures only; never update existing rows/schema."""
    path = get_database_path(database_path)
    if not path.is_file():
        raise FileNotFoundError("请先初始化独立 DEMO 数据库。")
    engine = create_sqlite_engine(path)
    try:
        with create_session_factory(engine)() as session, session.begin():
            session.execute(text("BEGIN IMMEDIATE"))
            product = session.get(Product, "00000000-0000-4000-8000-000000000010")
            if product is None or (
                product.sku, product.name, product.description, product.price_cents, product.currency
            ) != ("DEMO-SKU-001", "DEMO 虚构商品", "仅用于本地演示，不是真实商品。", 19900, "CNY"):
                raise DemoDataConflict("DEMO 标识或保留记录不匹配，未补充任何数据。")
            missing = []
            for user in DEMO_USERS.values():
                account = session.get(User, user.user_id)
                original = session.scalar(select(Order).where(Order.order_no == user.order_no))
                if (
                    account is None or account.auth_subject != f"demo:{user.user_id}"
                    or original is None or original.user_id != user.user_id or original.product_id != product.id
                ):
                    raise DemoDataConflict("DEMO 标识或保留记录不匹配，未补充任何数据。")
                fixture = DEMO_AFTERSALES_ORDERS[user.user_id]
                existing = session.scalar(select(Order).where(Order.order_no == fixture.order_no))
                by_id = session.get(Order, fixture.id)
                if existing is not None:
                    if (existing.id, existing.user_id, existing.product_id) != (fixture.id, user.user_id, product.id):
                        raise DemoDataConflict("预留 DEMO 订单标识存在冲突，未补充任何数据。")
                    # Preserve status, timestamps and any existing user edits.
                    continue
                if by_id is not None:
                    raise DemoDataConflict("预留 DEMO 订单标识存在冲突，未补充任何数据。")
                missing.append((user, fixture))
            created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for user, fixture in missing:
                session.add(Order(
                    id=fixture.id, order_no=fixture.order_no, user_id=user.user_id, product_id=product.id,
                    product_name_snapshot="DEMO 虚构商品", sku_snapshot="DEMO-SKU-001",
                    unit_price_cents=19900, quantity=1, total_amount_cents=19900,
                    status=fixture.status, created_at=created_at,
                    paid_at=created_at + timedelta(hours=1), shipped_at=created_at + timedelta(days=1),
                    completed_at=created_at + timedelta(days=2) if fixture.status == OrderStatus.COMPLETED else None,
                ))
        return len(missing)
    except SQLAlchemyError:
        raise DemoDataConflict("现有文件不是兼容的 DEMO 数据库，未补充任何数据。") from None
    finally:
        engine.dispose()


def initialize_or_supplement_demo_database(database_path: Path = DEMO_DATABASE_PATH) -> int:
    """Explicit UI entry point: exclusive new-file creation or safe append-only seeding."""
    path = get_database_path(database_path)
    if not path.exists():
        try:
            initialize_demo_database(path)
        except FileExistsError:
            pass  # Another initializer won; supplement still validates every marker.
    return supplement_demo_database(path)


def initialize_demo_staff(database_path: Path = DEMO_DATABASE_PATH) -> bool:
    """Explicit append-only staff provisioning; never promote/re-enable an account."""
    path = get_database_path(database_path)
    if not path.is_file():
        raise FileNotFoundError("请先初始化独立 DEMO 数据库。")
    engine = create_sqlite_engine(path)
    try:
        with create_session_factory(engine)() as session, session.begin():
            session.execute(text("BEGIN IMMEDIATE"))
            for demo_user in DEMO_USERS.values():
                user = session.get(User, demo_user.user_id)
                if user is None or user.auth_subject != f"demo:{demo_user.user_id}":
                    raise DemoDataConflict("未确认是虚构 DEMO 数据库，未创建员工。")
            label, identity = next(iter(DEMO_STAFF.items()))
            subject = f"demo:staff:{identity}"
            existing = session.scalar(select(User).where((User.id == identity) | (User.auth_subject == subject)))
            if existing is not None:
                if (existing.id != identity or existing.auth_subject != subject
                        or existing.role is not UserRole.STAFF or not existing.is_active):
                    raise DemoDataConflict("DEMO 员工标识冲突或账号停用，未修改已有账号。")
                return False
            session.add(User(id=identity, auth_subject=subject, display_name=label, role=UserRole.STAFF))
        return True
    except SQLAlchemyError:
        raise DemoDataConflict("DEMO 员工初始化失败，未修改已有账号。") from None
    finally:
        engine.dispose()
