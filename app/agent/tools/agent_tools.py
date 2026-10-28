"""RAG tool adapter; commerce tools live in commerce_tools.py."""

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService


rag = RagSummarizeService()


@tool(description="检索知识库并依据资料回答；无资料或检索异常时返回明确说明，不得自行补充知识库事实")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)
