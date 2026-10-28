import time

import streamlit as st
from agent.react_agent import ReactAgent
from agent.errors import AIServiceUnavailable, AI_UNAVAILABLE_MESSAGE, log_ai_failure
from business.demo import DEMO_USERS, DEMO_AFTERSALES_ORDERS, DEMO_DATABASE_PATH, initialize_or_supplement_demo_database, create_demo_order_service
from business.ticket_confirmation import discard_ticket_session, ticket_session

# 标题
st.set_page_config(page_title="ServicePilot AI")
st.title("ServicePilot AI｜智能客服")
st.caption("欢迎使用 ServicePilot AI，可查询订单、准备售后申请并在页面确认提交。")
st.divider()

current_user_id = None
demo_service = None
with st.sidebar:
    st.markdown("### ServicePilot AI")
    st.warning("DEMO：仅本地虚构账号演示，不是真实身份认证。")
    demo_enabled = st.checkbox("启用 DEMO 订单查询", value=False)
    if demo_enabled:
        selected = st.selectbox("选择预设虚构用户", ["未选择", *DEMO_USERS.keys()])
        if st.button("初始化独立 DEMO 数据库"):
            try:
                added = initialize_or_supplement_demo_database()
                st.success(f"DEMO 初始化/补充完成，新增售后订单 {added} 笔；已有记录未覆盖。")
            except FileExistsError:
                st.info("DEMO 数据库已存在，未覆盖任何内容。")
            except Exception:
                st.error("DEMO 初始化失败，请检查本地配置。")
        demo_user = DEMO_USERS.get(selected)
        if demo_user is not None and DEMO_DATABASE_PATH.is_file():
            if "demo_order_service" not in st.session_state:
                st.session_state["demo_order_service"] = create_demo_order_service()
            demo_service = st.session_state["demo_order_service"]
            current_user_id = demo_user.user_id
            st.caption(f"可查询的虚构订单：{demo_user.order_no}")
            st.caption(f"售后验收订单：{DEMO_AFTERSALES_ORDERS[demo_user.user_id].order_no}（请先初始化/补充）")
        elif demo_user is not None:
            st.info("请先初始化 DEMO 数据库。")

if st.session_state.get("demo_identity") != current_user_id:
    discard_ticket_session(st.session_state.pop("ticket_session_handle", None))
    st.session_state["message"] = []
    st.session_state.pop("agent", None)
    st.session_state["demo_identity"] = current_user_id

ticket_workflow = None
if current_user_id is not None and demo_service is not None:
    handle, ticket_workflow = ticket_session(
        st.session_state.get("ticket_session_handle"), current_user_id, demo_service.session_factory
    )
    st.session_state["ticket_session_handle"] = handle

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

if st.session_state.pop("ai_service_error", False):
    st.error(AI_UNAVAILABLE_MESSAGE)

# 用户输入提示词
prompt = st.chat_input()

if ticket_workflow is not None:
    # Any new chat supersedes the old proposal, even text saying "confirm".
    if prompt:
        ticket_workflow.invalidate()
    pending = ticket_workflow.pending(current_user_id)
    result = ticket_workflow.result(current_user_id)
    if result is not None:
        if result["ok"]:
            st.success(f"工单 {result['data']['ticket_no']}，状态：{result['data']['status']}。等待人工审核。")
        else:
            st.error(result["error"]["message"])
    if pending is not None and not (result and result["ok"]):
        st.subheader("待确认售后申请")
        st.write(f"订单号：{pending['order_no']}")
        st.write(f"售后类型：{pending['type']}")
        st.text(pending["description"])
        st.caption(pending["notice"])
        st.caption(f"有效期至：{pending['expires_at']}")
        if st.button("确认提交", key=f"ticket_confirm_{pending['draft_id']}"):
            ticket_workflow.confirm_from_button(current_user_id=current_user_id, draft_id=pending["draft_id"])
            st.rerun()
        elif st.button("取消申请", key=f"ticket_cancel_{pending['draft_id']}"):
            ticket_workflow.cancel_from_button(current_user_id=current_user_id, draft_id=pending["draft_id"])
            st.rerun()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        stage = "initialization"
        try:
            if "agent" not in st.session_state:
                st.session_state["agent"] = ReactAgent()
            stage = "response_render"
            res_stream = st.session_state["agent"].execute_stream(
                prompt, current_user_id=current_user_id, order_service=demo_service,
                ticket_preparer=ticket_workflow.preparer if ticket_workflow is not None else None,
            )
            st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        except AIServiceUnavailable:
            # The Agent already recorded safe metadata. Do not log its cause again.
            st.session_state["ai_service_error"] = True
        except Exception:
            log_ai_failure(stage, partial_output=bool(response_messages))
            st.session_state["ai_service_error"] = True
        else:
            st.session_state["message"].append({"role": "assistant", "content": "".join(response_messages)})
    # Rerun outside the catch: clears any partially rendered answer, preserves the
    # input box, and displays only the fixed error banner on the next render.
    st.rerun()
