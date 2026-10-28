# ServicePilot AI 虚构客服知识库

`sample_knowledge/` 包含五份原创 TXT：订单状态、退货、换货、维修和审核流程。
这些是可随项目公开分发的虚构 DEMO 资料，不包含第三方教程文本或真实客户数据，
不代表任何真实平台政策。知识只解释规则；实时订单、归属、资格与审核结果以业务工具和数据库为准。

## 导入

使用根 README 中已安装 `requirements.txt` 的 `.venv`，从 `app/` 执行：

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_knowledge_base.py --help
```

先按 [ENV_SETUP.md](ENV_SETUP.md) 在当前 PowerShell 安全设置 `DASHSCOPE_API_KEY`。
项目不会自动读取 `.env`；同一个变量供模型和 Embedding 使用。不要把真实密钥写进文件或命令历史。

**以下命令会允许调用 DashScope Embedding API，可能产生费用：**

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_knowledge_base.py --yes
```

没有 `--yes` 会显示费用说明并退出；没有密钥、使用模板占位值或没有支持的知识文件时，
会在创建向量库服务前拒绝导入。退出码 0 表示完成，2 表示参数/前置条件未满足，1 表示导入失败或部分失败。
非空变量不等于密钥有效，真实请求仍需要网络、权限与额度。

默认读取 `sample_knowledge/` 的直接子文件，不递归。可选 `--source` 指向其他已获授权的
TXT/PDF 目录；相对路径以执行命令时的当前目录为准。不要用此命令导入真实敏感客户资料。

## 结果与已有数据保护

- 复用 `config/chroma.yml` 的 collection `agent`、持久化路径 `rag/chroma_db`、
  文本切分配置（200/20）及现有 TXT/PDF 加载器；Embedding 沿用 `config/rag.yml`。
- 输出文件数、新增文件数、跳过数、失败数及新增分片数，不输出内容、密钥、完整路径或异常原文。
  全新导入通常处理五个文件；准确分片数量以结果为准。重复导入应跳过已有分片。
- 继续维护 `md5.txt`，并使用基于文件 MD5、分片序号和内容的稳定 ID 核对当前 collection。
  Chroma 已写入但 MD5 写入失败时，可重试而不重复插入；旧 MD5 存在但 collection 为空也不会误跳过。
- 对旧版随机 ID，只有 MD5 已知且原 `source` 下的实际分片内容能够核对时才兼容跳过。
  不删除、重建 collection，不清空 MD5，也不改写不同内容的已有保留 ID。
- 本命令是增量追加，不是完整同步：修改文件会追加新版本，不会删除旧版本。旧资料移动目录、
  切分配置或 Embedding 模型变化时，应先核对已有索引；不要直接删除用户向量库来排障。
- 部分失败时已写入内容会保留，退出码为 1；修复配置、权限或网络后使用同一命令重试。
  CLI 不在 Streamlit 启动时自动运行，也不接触业务 SQLite 数据库。

Chroma、MD5 和运行日志被 Git 忽略；示例 TXT、CLI 及此说明可被跟踪。

## 在客服页面演示

导入后启动或重启 Streamlit，询问：

> 请使用知识库说明 ServicePilot DEMO 的换货申请条件，以及用户确认后如何审核。

应调用 `rag_summarize` 并依据检索片段回答。若询问某笔订单的当前状态，必须使用 `get_order`；
知识库中的 DEMO 初始状态不能代替实时查询。知识库和模型都不能代替页面确认或员工审核。

空检索时 RAG 不调用生成模型，返回无资料说明；检索异常与生成异常返回各自的安全错误提示。
重复导入前可先看上次结果，避免无必要的真实请求。真实模型效果仍需另行验收。

## 离线测试与临时数据

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests -q
```

自动化测试使用离线 Embedding、Mock 和 pytest 专用临时 Chroma，不调用真实 DashScope API。
临时数据由 pytest 的临时目录保留/清理机制管理；不需要清空项目向量库。
如需手动清理，只处理已经确认由该测试运行创建的专用临时目录，并先退出相关测试进程；
不要删除项目 `rag/chroma_db`、`md5.txt` 或任何已有业务数据库。
