"""Local DEMO review UI; model text and customer session identity are never used."""

import logging

from business.database import create_session_factory, create_sqlite_engine
from business.demo import (
    DEMO_DATABASE_PATH, DEMO_STAFF, initialize_demo_staff,
    initialize_or_supplement_demo_database,
)
from business.review_queries import TicketReviewQueryService
from business.services import TicketReviewService

logger = logging.getLogger(__name__)


def render_review_page(st):
    st.title("ServicePilot AI｜员工审核台")
    st.warning("仅限本地 DEMO：使用预设虚构员工，不具备生产环境身份认证能力。请勿公开部署。")
    st.info("批准仅表示售后申请获得批准，不代表已执行退款、支付、换货或维修。")
    if not st.checkbox("启用本地 DEMO 员工审核", value=False):
        return
    if st.button("初始化 / 补充 DEMO 员工"):
        try:
            initialize_or_supplement_demo_database()
            initialize_demo_staff()
            st.success("DEMO 员工已就绪，已有账号及工单未覆盖。")
        except Exception:
            logger.error("Demo staff initialization failed; details omitted")
            st.error("DEMO 初始化失败，请检查本地配置；已有账号未被覆盖或重新启用。")
    selection = st.selectbox("选择预设员工", ["未选择", *DEMO_STAFF])
    reviewer_id = DEMO_STAFF.get(selection)
    if reviewer_id is None:
        return
    if not DEMO_DATABASE_PATH.is_file():
        st.info("请先初始化独立 DEMO 数据库及员工。")
        return
    engine = None
    try:
        engine = create_sqlite_engine(DEMO_DATABASE_PATH)
        factory = create_session_factory(engine)
        render_review_content(st, reviewer_id, TicketReviewQueryService(factory), TicketReviewService(factory))
    except Exception:
        logger.error("Review page failed; details omitted")
        st.error("员工审核台暂时不可用，请稍后重试。")
    finally:
        if engine is not None:
            engine.dispose()


def render_review_content(st, reviewer_id, queries, reviews):
    # Both reads and writes independently recheck the persisted role/active flag.
    view = st.selectbox("工单范围", ["待审核", "全部（只读查看已审核工单）"])
    st.button("刷新工单列表")  # Streamlit reruns automatically on a button event.
    result = queries.list_tickets(current_reviewer_id=reviewer_id,
                                  status="pending_review" if view == "待审核" else None)
    if not result["ok"]:
        st.error(result["error"]["message"])
        return
    feedback = st.session_state.pop("staff_review_feedback", None)
    if feedback:
        st.success(feedback)
    st.caption("最多显示100条，按创建时间、工单号升序排列；所有时间为 UTC。")
    if not result["data"]:
        st.info("当前没有工单。")
        return
    st.dataframe(result["data"], hide_index=True)
    selected = st.selectbox("选择工单查看详情", [row["ticket_no"] for row in result["data"]])
    detail = queries.get_ticket(current_reviewer_id=reviewer_id, ticket_no=selected)
    if not detail["ok"]:
        st.error(detail["error"]["message"])
        return
    item = detail["data"]
    st.subheader("工单详情")
    for field in ("ticket_no", "order_no", "type", "description", "created_at", "status", "reviewed_at", "review_note"):
        st.text(f"{field}：{item[field] if item[field] is not None else '—'}")
    if item["status"] != "pending_review":
        st.info("此工单已不处于待审核状态，仅可查看，不能再次审核。")
        return
    note = st.text_area("审核意见（必填，1–2000字）", max_chars=2000, key=f"staff_note_{selected}")
    approve = st.button("批准", key=f"staff_approve_{selected}")
    reject = st.button("拒绝", key=f"staff_reject_{selected}")
    if approve or reject:
        response = reviews.review_ticket(current_reviewer_id=reviewer_id, ticket_no=selected,
                                         decision="approve" if approve else "reject", review_note=note)
        if response["ok"]:
            st.session_state["staff_review_feedback"] = f"工单 {selected} 已审核，状态：{response['data']['status']}。未执行退款。"
            st.rerun()
        else:
            st.error(response["error"]["message"])
