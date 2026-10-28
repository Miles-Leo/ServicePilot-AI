"""Knowledge-grounded generation with safe empty/error boundaries."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agent.errors import install_provider_log_filters
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompt


install_provider_log_filters()
NO_KNOWLEDGE_MESSAGE = "当前知识库没有找到相关资料，无法依据知识库回答。"
RETRIEVAL_UNAVAILABLE_MESSAGE = "知识库检索暂时不可用，请稍后重试。"
GENERATION_UNAVAILABLE_MESSAGE = "知识库回答暂时不可用，请稍后重试。"


class RAGRetrievalUnavailable(RuntimeError):
    def __init__(self):
        super().__init__(RETRIEVAL_UNAVAILABLE_MESSAGE)


class RagSummarizeService:
    def __init__(self, *, retriever=None, model=None):
        # Defaults preserve Chroma/DashScope; injection enables API-free tests.
        if retriever is None:
            from rag.vector_store import SectorStoreService
            self.vector_store = SectorStoreService()
            retriever = self.vector_store.get_retriever()
        self.retriever = retriever
        if model is None:
            from model.factory import chat_model
            model = chat_model
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(self, query: str):
        try:
            return self.retriever.invoke(query)
        except Exception:
            logger.error("rag_failed stage=retrieval details_omitted")
            raise RAGRetrievalUnavailable() from None

    def rag_summarize(self, query: str):
        try:
            context_docs = self.retriever_docs(query)
        except RAGRetrievalUnavailable:
            return RETRIEVAL_UNAVAILABLE_MESSAGE
        if not context_docs:
            logger.info("rag_empty stage=retrieval documents=0")
            return NO_KNOWLEDGE_MESSAGE
        try:
            context = ""
            for count, doc in enumerate(context_docs, 1):
                context += f"[参考资料]:{count}, [资料内容]:{doc.page_content}, [参考资料来源]:{doc.metadata}\n"
            logger.info("rag_retrieved stage=generation documents=%d", len(context_docs))
            return self.chain.invoke({"context": context, "input": query})
        except Exception:
            logger.error("rag_failed stage=generation details_omitted")
            return GENERATION_UNAVAILABLE_MESSAGE


if __name__ == '__main__':
    try:
        RagSummarizeService()
        logger.info("rag_ready stage=initialization")
    except Exception:
        logger.error("rag_failed stage=initialization details_omitted")
