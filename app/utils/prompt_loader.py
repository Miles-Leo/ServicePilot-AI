from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompt() -> str:
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt]在yaml配置文件中未找到配置项main_prompt_path")
        raise

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception:
        logger.error("prompt_loading_failed stage=system details_omitted")
        raise RuntimeError("提示配置暂时不可用。") from None


def load_rag_prompt() -> str:
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt]在yaml配置文件中未找到配置项rag_summarize_prompt_path")
        raise

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception:
        logger.error("rag_failed stage=prompt_loading details_omitted")
        raise RuntimeError("知识库提示配置暂时不可用。") from None
