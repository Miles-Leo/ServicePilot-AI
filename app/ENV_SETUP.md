# ServicePilot AI 本地模型 API 环境配置

聊天模型通过 LangChain `init_chat_model(model_provider="openai")` 连接 DashScope
OpenAI-compatible Chat Completions API，默认仍为 `qwen-max`；向量模型仍使用
`DashScopeEmbeddings`。两者均使用 `DASHSCOPE_API_KEY`，不需要安装 python-dotenv。

聊天接口可通过 `DASHSCOPE_CHAT_BASE_URL` 配置，默认北京地址为
`https://dashscope.aliyuncs.com/compatible-mode/v1`。该设置只影响聊天，不改变 Embedding。
只接受无账号、查询参数或片段的 DashScope HTTPS `/compatible-mode/v1` 地址。
若使用其他地域或 workspace 地址，必须与密钥所属地域、工作空间及模型权限一致。
参见 [DashScope 官方地址说明](https://help.aliyun.com/en/model-studio/base-url)。
不要使用 Coding Plan 地址作为此业务服务的兼容接口，也不要把密钥放入 URL。
Embedding 仍保持原来的原生 SDK 配置；仅切换聊天地址不代表 Embedding 已切换地域。

例如在当前 PowerShell 中显式选择默认聊天地址：

```powershell
$env:DASHSCOPE_CHAT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
```

**`.env.example` 只是配置示例，不是配置加载器。当前项目不会自动加载 `.env`。**
复制或编辑 `.env` 不会使变量自动生效。模板中的占位值不是可用密钥，不能用于实际调用。
真实密钥只能在本机运行环境设置，不能写入模板、源码、截图、聊天或 Git 提交。

## 1. 在 PowerShell 设置当前进程环境变量

使用根 README 中已安装 `requirements.txt` 的 `.venv`（推荐 CPython 3.13.9）。
以下命令均从 `app/` 执行；如果终端位于仓库根目录，先执行一次：

```powershell
Set-Location '.\app'
```

然后在准备启动 Streamlit 的同一个 PowerShell 窗口中执行：

```powershell
$dashscopeSecret = Read-Host '请输入你自己的 DashScope API Key（输入不回显）' -AsSecureString
$env:DASHSCOPE_API_KEY = [System.Net.NetworkCredential]::new('', $dashscopeSecret).Password
Remove-Variable dashscopeSecret
```

密钥由交互输入提供，不作为命令字面量保存在 PowerShell 历史中。
环境变量在运行进程中仍是明文；不要打印进程环境或将其作为调试日志输出。
这不会写入系统/用户永久环境变量，也不会改变已经运行的 Streamlit 进程。

## 2. 只检查变量是否存在

```powershell
& '.\.venv\Scripts\python.exe' -c "import os; print('DASHSCOPE_API_KEY configured:', bool(os.getenv('DASHSCOPE_API_KEY', '').strip()))"
```

只输出 `True` / `False`，不输出密钥。`True` 仅表示变量非空，不代表密钥有效、
有模型权限或有可用额度。不要使用 `echo $env:DASHSCOPE_API_KEY` 检查。

## 3. 启动 Streamlit

仍在 `app/` 和同一个 PowerShell 窗口中执行：

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run app.py --server.address 127.0.0.1
```

Python 子进程继承当前 PowerShell 的环境变量；客服模型和 RAG Embedding 使用同一个变量。
员工审核台从应用侧边栏的 Staff Review 入口进入，不需要单独配置模型密钥。
目录迁移后，本地虚拟环境仍原样保留，但旧激活脚本或 `pip` / `streamlit` 启动器可能含旧绝对路径。
请使用选定环境中的 `Scripts/python.exe -m ...`；新克隆按根 README 新建环境，不要覆盖已有环境。
如果原服务已运行，请先在其终端用 Ctrl+C 停止，再从已设置变量的终端重新启动。

## 4. 新终端和清除配置

新的独立终端不会继承另一个终端中临时设置的变量，需要重新执行第 1、2 步。
若操作系统中已有永久配置，终端可能继承该配置；本说明不创建永久配置。
退出终端会结束本次临时配置。也可以在停止 Streamlit 后执行：

```powershell
Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:DASHSCOPE_CHAT_BASE_URL -ErrorAction SilentlyContinue
```

## 5. 未配置或配置错误

未设置变量时，工厂在构造模型对象前会报缺少 `DASHSCOPE_API_KEY` 的安全配置错误。
`chat_model` 和 `embed_model` 仍在首次导入 `model.factory` 时构造，没有改为懒加载；
无密钥的离线测试应注入假模型或 Mock 工厂，不能靠占位值调用真实服务。
客服首次提问会初始化 Agent，页面可能只显示统一提示“AI 服务暂时不可用，请稍后重试。”；
首页能够打开不代表模型配置有效。使用占位值或无效密钥也可能在实际请求时遭到认证拒绝。
存在变量但调用失败，还需检查模型权限、额度与网络；不要在排查中粘贴密钥或请求头。

`.gitignore` 忽略真实 `.env`、`.env.*` 和 `*.env`，仅为本目录 `.env.example` 设置例外。
忽略规则不是加密措施，也无法自动清除已被 Git 跟踪的秘密。提交前始终检查暂存内容。
离线自动化测试不需要真实模型密钥；不要为了运行测试填写或提交真实密钥。
