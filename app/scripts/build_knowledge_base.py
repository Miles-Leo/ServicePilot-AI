"""Explicit additive knowledge import. No imports of model SDKs during --help."""

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "参数不合法，请使用 --help 查看用法。\n")


def build_parser():
    parser = SafeParser(description="向现有 RAG collection 增量导入知识资料；DashScope Embedding API 可能收费。")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "sample_knowledge",
                        help="TXT/PDF 所在目录，默认主项目 sample_knowledge；不递归扫描")
    parser.add_argument("--yes", action="store_true", help="确认允许调用可能收费的 DashScope Embedding API")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("提示：导入可能调用 DashScope Embedding API 并产生费用；不会删除已有 Chroma 数据。", flush=True)
    if not args.yes:
        print("未确认费用提示，未执行导入。确认后使用 --yes。")
        return 2
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key or key == "REPLACE_WITH_YOUR_DASHSCOPE_API_KEY":
        print("未配置可用的 DASHSCOPE_API_KEY 环境变量，请先按 ENV_SETUP.md 配置。")
        return 2
    try:
        from utils.config_handler import chroma_conf
        from utils.file_handler import listdir_with_allowed_type
        source = args.source.resolve()
        files = listdir_with_allowed_type(str(source), tuple(chroma_conf["allow_knowledge_type"]))
        if not files:
            print("知识目录不存在、不可读取或没有支持的 TXT/PDF 文件，未执行导入。")
            return 2
        from rag.vector_store import SectorStoreService
        result = SectorStoreService().load_document(source_dir=source)
        print(f"导入结果：文件={result.files}，新增文件={result.imported}，跳过={result.skipped}，失败={result.failed}，新增分片={result.chunks_added}。")
        if result.failed:
            print("部分资料导入失败，已成功写入的分片保留，可安全重试；请检查阶段日志。")
            return 1
        return 0
    except Exception:
        print("知识库导入失败，内部详情已隐藏。请检查依赖、配置、网络和模型权限。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
