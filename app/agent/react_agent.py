from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent.context import AgentContext
from agent.business_response import commerce_failure_messages
from agent.errors import AIServiceUnavailable, log_ai_failure
from agent.tools.commerce_tools import get_order, prepare_ticket
from business.database import create_session_factory, create_sqlite_engine
from business.services import OrderService


class ReactAgent():
    def __init__(self, *, model=None, order_service=None, rag_tools=None):
        # Lazy defaults support offline model and RAG dependencies.
        from utils.prompt_loader import load_system_prompt
        from agent.tools.middleware import monitor_tool, log_before_model

        if model is None:
            from model.factory import chat_model
            model = chat_model
        if rag_tools is None:
            from agent.tools.agent_tools import rag_summarize
            rag_tools = [rag_summarize]
        self.order_service = order_service if order_service is not None else OrderService(
            create_session_factory(create_sqlite_engine())
        )
        self.agent = create_agent(
            model=model,
            system_prompt=load_system_prompt(),
            tools=[get_order, prepare_ticket, *rag_tools],
            middleware=[monitor_tool, log_before_model],
            context_schema=AgentContext,
        )

    def execute_stream(self, query: str, *, current_user_id: str | None = None, order_service=None, ticket_preparer=None):
        """Identity/service arguments are supplied by the backend, never model tools."""
        input_dict = {
            "messages":[
                {"role":"user","content":query},
            ]
        }
        context = AgentContext(
            current_user_id=current_user_id,
            order_service=order_service if order_service is not None else self.order_service,
            ticket_preparer=ticket_preparer,
        )
        emitted = False
        business_failures = []
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context=context):
                for failure in commerce_failure_messages(chunk["messages"]):
                    if failure not in business_failures:
                        business_failures.append(failure)
                latest_message = chunk["messages"][-1]
                if isinstance(latest_message, AIMessage) and not latest_message.tool_calls and latest_message.text.strip():
                    emitted = True
                    yield ("\n".join(business_failures) if business_failures else latest_message.text.strip()) + "\n"
            if not emitted:
                raise AIServiceUnavailable()
        except Exception:
            log_ai_failure("stream", partial_output=emitted)
            # No retries: an earlier tool in this run may already have executed.
            raise AIServiceUnavailable() from None


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("你好，请介绍可以提供的客服帮助"):
        print(chunk, end="", flush=True)
