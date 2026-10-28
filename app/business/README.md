# ServicePilot AI business layer

Requires Python 3.12+ (SQLite connection `autocommit` support) and the dependencies
in `../requirements-business.txt`. This package is independent of Agent/RAG imports.

## Configuration and lifecycle

- `SERVICEPILOT_DB_PATH` configures a SQLite file path; callers may also pass a
  path directly to `create_sqlite_engine()` for tests or explicit configuration.
- Default: `app/servicepilot.sqlite3`. Relative paths resolve against
  the Agent project directory. Allowed suffixes are `.db`, `.sqlite`, `.sqlite3`.
- `create_session_factory(engine)` returns a SQLAlchemy session factory. Use
  `factory.begin()` for commit-on-success/rollback-on-error transactions.
- Imports never connect or create tables. Schema creation must be explicit after
  importing `business.models`; no production schema initialization is run here.
- SQLite foreign keys are enabled on every new connection. Referenced records
  are not cascade-deleted. Busy timeout is five seconds.
- All local databases and `.venv-business/` are ignored by Git.

## Data contract

- One order references one product/SKU, with quantity and historical snapshots.
- Money is integer CNY cents. SQLite CHECK constraints reject negative amounts,
  noninteger stored values, nonpositive quantities and inconsistent order totals.
- Enum values are lowercase strings, protected by database CHECK constraints.
- Ticket ownership must match order ownership through a composite foreign key.
- `(user_id, idempotency_key)` is unique across all ticket states. A separate
  partial unique index on `(user_id, order_id, type)` covers `pending_review`,
  `approved`, and `in_progress`. Terminal tickets do not occupy that index.
- Duplicate inserts raise `IntegrityError`; `TicketService` handles idempotency
  replay/conflicts after rollback, while direct ORM callers own their handling.
- UTC timestamps accept timezone-aware inputs only. Before persistence they are
  converted to timezone-naive UTC; typed reads restore `timezone.utc`. Naive input
  is rejected. Reload/refresh after flush to observe normalization of assigned
  values. Raw SQL must honor this storage convention itself.
- IDs, initial statuses and timestamps use ORM defaults; raw SQL inserts must
  supply those values. `updated_at` changes through SQLAlchemy updates, not a DB
  trigger.
- Legal enum values are enforced here. `TicketService` normalizes and fingerprints
  creation requests. `TicketReviewService` checks that the reviewer is an active
  STAFF account and atomically transitions `pending_review` to `approved` or
  `rejected`, recording reviewer, UTC review time and required review note.
  This is role authorization, not production login authentication.

## Tests

From the repository root on Windows:

```powershell
& './app/.venv/Scripts/python.exe' -m pytest './app/tests/test_models.py' -q
```

Database tests use temporary SQLite databases and synthetic records. RAG tests
use temporary Chroma collections with offline embeddings or Mock/Stub objects.
Tests do not create the default business database or access user-owned databases
or existing vector stores; they do not call real DashScope APIs.

For the entire suite, install the complete `app/requirements.txt`, not only the
business or Agent test subset, and replace the test-file argument with `./app/tests`.
Use the same virtual-environment interpreter as the root README; create it there
first without overwriting an existing environment.

## Order queries

- Construct `OrderService(session_factory)` and call
  `get_order(current_user_id=..., order_no=...)` using keyword arguments.
- The authenticated backend must supply `current_user_id`. This service checks
  account existence and availability; it does not authenticate login credentials
  and must not be exposed with a model-controlled identity argument.
- Order numbers are case-sensitive, 1–64 ASCII letters/digits/underscores/hyphens,
  beginning with a letter or digit. Whitespace is rejected rather than normalized.
- `OrderRepository` performs parameterized SQL queries with both order number and
  current user ID as predicates. Staff have no ownership bypass on this path.
- `schemas.py` defines standard-library TypedDict responses. Runtime argument
  validation is explicit in the service; no Pydantic dependency is required.
- Every response has `ok`, `data`, and `error`. Success returns whitelisted order
  fields and historical product snapshots, integer CNY cents, and ISO 8601 UTC
  timestamps (`+00:00`). Optional timestamps are null. User identifiers and ORM
  relationships are never serialized.
- Errors are `INVALID_ARGUMENT`, `UNAUTHENTICATED`, `ORDER_NOT_FOUND`, or
  `TEMPORARY_UNAVAILABLE`. Missing and foreign-owned orders return identical
  responses. Only database-unavailable responses have `retryable=true`; logs
  include the exception class but not exception text, SQL, parameters or traces.
- Paid orders allow return requests; shipped/completed orders allow returns,
  exchanges and repairs; pending-payment/cancelled orders allow none. This is
  status-based application eligibility, not an approval guarantee or refund.
- The service creates/closes its own read-only session. It does not create a
  schema, modify orders, create tickets, or integrate with Agent tools.

## Agent order tool and local DEMO

- API compatibility was checked against the existing Python 3.13 installation:
  LangChain 1.4.2, langchain-core 1.6.12 and LangGraph 1.2.12. These support
  `create_agent(context_schema=...)`, `ToolRuntime` and `stream(context=...)`.
  `../requirements-agent-test.txt` retains the smaller offline Agent dependency
  subset; the complete app and current full suite require `../requirements.txt`.
- `agent.context.AgentContext` is a frozen dataclass. Only backend code supplies
  `current_user_id` and the `OrderService` dependency. Identity is immutable;
  the former tutorial report state has been removed.
- The `get_order` tool exposes only `order_no`. Runtime is an injected argument,
  excluded from the model schema and overwritten by LangGraph ToolNode. Unknown
  model arguments are rejected with a sanitized `INVALID_ARGUMENT` response.
- The tool delegates to the existing service and its short-lived sessions. No
  identity is inferred from chat text or model-provided identity fields.
- `ReactAgent` registers only `get_order`, preparation-only `prepare_ticket`, and
  `rag_summarize`, and accepts optional backend model/service/RAG-tool dependencies
  (`rag_tools`) for offline tests. Weather, random identities/months and usage-report
  tools have been removed; there is no report prompt switch or CSV report loader.
  Streaming emits final assistant messages, excluding raw tool/user messages.
- The main prompt requires order tool evidence. Prompt instructions
  guide model behavior; authorization is enforced by the service and database.
- In Streamlit, enable **DEMO 订单查询**, select one of the two fixed synthetic
  users, and explicitly click **初始化独立 DEMO 数据库**. The dedicated file is
  `app/servicepilot_demo.sqlite3`, excluded by the existing `.gitignore`.
  The UI calls `initialize_or_supplement_demo_database()`: exclusively create a
  new file if absent, then append missing verified fixtures transactionally.
  Existing files/rows are never overwritten. No initialization occurs on import
  or page load. The original `initialize_demo_database()` remains new-file-only.
- User A owns `DEMO-ORDER-A`; user B owns `DEMO-ORDER-B`. Users, product and orders
  are synthetic. Only the backend allowlist maps selector labels to identity;
  changing/disabling the selected account clears chat and recreates the Agent.
- **DEMO is local testing, not authentication.** Do not expose this account
  selector publicly. Real authentication must replace this injection point before
  accepting real users. The Agent has no ticket-creation tool; the standalone
  `TicketService` records requests only. Human review is a separate backend-only
  service described below; the separate local DEMO staff page is not an Agent tool.
  Refund execution is not implemented.
- Offline integration tests execute real LangChain/LangGraph graphs with a fake
  chat model. Streamlit wiring tests use a lightweight UI substitute; RAG tests
  also use temporary real Chroma with offline embeddings. No real model API or
  existing user vector store is required.

Full integration tests from the repository root:

```powershell
& './app/.venv/Scripts/python.exe' -m pip install -r './app/requirements.txt'
& './app/.venv/Scripts/python.exe' -m pytest './app/tests' -q
```

## AI failure boundary

- `ReactAgent.execute_stream()` converts ordinary generation/stream failures to
  `AIServiceUnavailable`, without automatic retries. Empty responses also fail.
- The page catches both that signal and initialization/rendering failures. A
  rerun removes transient partial output and shows only
  `AI 服务暂时不可用，请稍后重试。`, while keeping the chat input available.
- Assistant history is appended only after the stream/render completes. Failed
  user questions remain visible; incomplete assistant answers are not saved as
  successful messages. A new user question is a new invocation, not an automatic
  replay of a possibly side-effecting tool call.
- Failure diagnostics include only a fixed event, stage and partial-output flag.
  Middleware logs tool names/message counts, never message bodies, arguments,
  exception payloads, headers or tracebacks. No provider credentials or complete
  order data are included by these logging paths.
- Tongyi/DashScope SDK loggers are filtered to fixed metadata too: their retry
  warnings and debug records can otherwise include exception or request bodies.

## Confirmed ticket requests (service only)

- Construct `TicketService(session_factory)`, then call `create_ticket()` with
  keyword-only `current_user_id`, `order_no`, `ticket_type`, `description`,
  `idempotency_key`, and `confirmed`. Identity, stable request key and confirmation
  must be supplied by a trusted backend, not inferred from model text.
- Only literal boolean `True` confirms submission. Otherwise the method returns
  `CONFIRMATION_REQUIRED` without accessing the database. Rendering/retrying a
  request does not establish human confirmation by itself.
- Order numbers follow the order-query format. Ticket types use the existing
  enum values. Descriptions are stripped, 1–2000 characters, valid UTF-8 without
  NUL. Keys are 1–255 ASCII letters/digits/`_.:-`, starting with a letter or digit;
  keys are not trimmed. The backend must reuse a key for retries of one submission.
- User existence/availability and order ownership are rechecked. Eligibility uses
  the same `_available_ticket_types()` rule as order queries: paid permits return;
  shipped/completed permit all three types; pending-payment/cancelled permit none.
- New tickets are `pending_review`. The JSON-ready `ok/data/error` response returns
  `ticket_no`, `order_no`, `type`, `status`, ISO 8601 UTC `created_at`, and `created`.
  It excludes user IDs, description, key, fingerprint and internal database fields.
- SHA-256 fingerprints cover a versioned canonical JSON of order number, type and
  stripped description. A per-user key with matching fingerprint returns the
  existing ticket (`created=false`), including terminal tickets. A mismatch returns
  `IDEMPOTENCY_CONFLICT`. Replay follows authorization but precedes current-state
  eligibility, so a later order-state change does not invalidate a successful retry.
- A different key with an existing active ticket of the same user/order/type
  returns `ACTIVE_TICKET_EXISTS`; it does not silently return or overwrite that
  ticket. New keys are allowed after terminal states, subject to order eligibility.
- A short SQLite `BEGIN IMMEDIATE` transaction reserves the writer before reads,
  preventing competing state changes between validation and insertion. Existing
  unique/partial indexes and foreign keys remain the final integrity guard. The
  success response is released only after commit completes.
- An `IntegrityError` rolls back the entire transaction. A fresh read session
  rechecks authorization and durable key/active-ticket records to classify the
  result. Unclassified constraint failures, busy locks and database errors return
  `TEMPORARY_UNAVAILABLE`; no SQL or exception payload is exposed or logged.
- Error codes: `INVALID_ARGUMENT`, `UNAUTHENTICATED`, `ORDER_NOT_FOUND`,
  `TICKET_TYPE_NOT_ALLOWED`, `CONFIRMATION_REQUIRED`, `IDEMPOTENCY_CONFLICT`,
  `ACTIVE_TICKET_EXISTS`, `TEMPORARY_UNAVAILABLE`. Only temporary failures are
  marked retryable; the service itself never automatically retries writes.
- The service itself performs no model/UI operations, review, refunds, exchanges
  or repair execution. The two-stage UI integration below calls it only after a
  confirmation button event. Tests use temporary synthetic data.

## Two-stage after-sales confirmation

- `prepare_ticket(order_no, ticket_type, description)` is the only model-facing
  after-sales tool. Runtime identity is injected from frozen `AgentContext`;
  `current_user_id`, `confirmed` and `idempotency_key` are forbidden model inputs.
  The context supplies a narrow `TicketPreparer` capability with no create or
  confirm method. There is no `create_ticket` LangChain tool.
- Preparation calls the existing `OrderService` to validate the active account,
  order ownership and `available_ticket_types`, then records an immutable draft
  in backend memory. It never inserts a Ticket. The structured response is
  `ok/data/error` with `draft_id`, `order_no`, `type`, `description`, `expires_at`,
  `requires_confirmation=true` and the human-review notice. Natural-language model
  output is never parsed as proof that a draft or confirmation exists.
- Each Streamlit session stores only a random opaque `ticket_session_handle`.
  Authoritative drafts and random stable idempotency keys stay in a private
  process registry, bound to the selected backend user. Arbitrary session-state
  draft dictionaries or confirmation flags are ignored. Registry handles are
  backend capabilities and must not be disclosed as model input or public data.
- Drafts expire after 10 minutes. Registry sessions expire after two idle hours,
  are cleaned lazily and capped at 1024 entries. Restarting the process safely
  loses unsubmitted drafts; this local single-process design is not a distributed
  persistent workflow or a replacement for real authentication.
- The page renders the backend summary and states: **提交后进入人工审核，不代表退款成功。**
  Only the actual `确认提交` button branch calls `confirm_from_button()`. It checks
  user, current draft version and expiry, then calls the existing `TicketService`
  with the stored fields/key and `confirmed=True`. The service rechecks account,
  ownership and current eligibility inside its transaction.
- `取消申请` clears the draft without writing a Ticket. Changing/disabling users
  discards and permanently revokes the entire old backend session, including
  capabilities held by a late-running old model call. Any new chat conservatively
  invalidates a pending draft, even text saying “confirm”. A new preparation gets
  a new version/key and requires its own button confirmation. A stale button event
  cannot confirm the new version. New text is processed before button branches.
- Reruns/page reloads never infer confirmation. Double clicks are serialized and
  reuse a cached successful result; explicit retries after a temporary failure use
  the original key. The existing database constraints still prevent duplicate
  tickets. Successful UI feedback shows the ticket number and `pending_review`.
- Expiry returns `PREPARATION_EXPIRED`; stale/cancelled proposals require new
  confirmation. Submission errors use safe business messages, with no exception
  details in UI/logs and no automatic retries.
- `DEMO-ORDER-A/B` remain pending-payment fixtures and reject after-sales requests.
  Explicit initialization/supplementation adds `DEMO-AFTERSALE-A` (user A,
  completed) and `DEMO-AFTERSALE-B` (user B, shipped), both 19900 CNY cents with a
  clearly synthetic product. These allow all three existing after-sales types.
- `supplement_demo_database()` validates the fixed DEMO accounts, product marker
  and original order ownership before adding missing fixtures in `BEGIN IMMEDIATE`.
  Reserved ID/number conflicts or unrecognized databases are rejected and rolled
  back. Repeated calls add zero rows, and preserve existing order edits and Tickets.
  It never calls `create_all()` on an existing file or updates/deletes records.
- Confirmation auditing logs fixed submit-requested/succeeded events without
  identity, order, description, key, exception payload or other sensitive fields.

## Human ticket review (backend service only)

- `TicketReviewService(session_factory).review_ticket()` accepts keyword-only
  `current_reviewer_id`, `ticket_no`, `decision`, and `review_note`. Reviewer ID
  must be supplied by an authenticated trusted backend, never by model/chat input.
  This service checks authorization; it does not implement login or staff provisioning.
- Existing `UserRole.STAFF` is the MVP review permission. Missing/inactive accounts
  return `UNAUTHENTICATED`; active customers return `FORBIDDEN` before ticket lookup.
  All active staff have review permission; finer-grained roles are not modeled.
- Decisions are exactly `approve` or `reject`, transitioning only `pending_review`
  to `approved` or `rejected`. Both require a stripped 1–2000-character UTF-8 note,
  without NUL. Ticket numbers use the existing 1–64-character ASCII identifier format.
- Existing fields store reviewer FK, note and aware UTC review time; `updated_at`
  uses that same instant. A pending record with existing review metadata is rejected
  rather than overwritten. Repeated reviews return `INVALID_TICKET_STATUS`, even
  if the requested decision matches the original. No reopen/edit-review operation exists.
- SQLite `BEGIN IMMEDIATE` reserves the writer before account/ticket reads. The
  SQL UPDATE also compares status to `pending_review` and requires all review fields
  to be NULL. Its affected-row count must be one. Thus concurrent reviewers cannot
  both succeed; the second receives `INVALID_TICKET_STATUS` after the first commits.
  An exhausted database lock timeout instead returns `TEMPORARY_UNAVAILABLE`.
- State and audit fields commit atomically, and success is returned only after
  commit. Database exceptions roll back; fixed logs and safe errors omit SQL,
  parameters, file paths, exception text and review content. No automatic retries.
- Response: `ok/data/error`; successful data contains only `ticket_no`, `status`,
  `reviewed_by`, `reviewed_at` (ISO UTC), `review_note`. This staff-facing response
  is not exposed through Agent tools. Errors: `INVALID_ARGUMENT`, `UNAUTHENTICATED`,
  `FORBIDDEN`, `TICKET_NOT_FOUND`, `INVALID_TICKET_STATUS`, `TEMPORARY_UNAVAILABLE`.
- Existing ORM supports this single immutable service-level decision, not an
  append-only audit history of multiple reviews or protection against direct SQL
  writes by a database administrator. No schema change or migration is needed.
- Approval does not execute refunds, payment, exchange, repair or fulfillment.
  No Agent, Streamlit, RAG, customer ownership rules or creation transactions change.

## Local DEMO staff console

- Run the existing Streamlit app with its runtime environment, binding locally:
  `& '.\.venv\Scripts\python.exe' -m streamlit run app.py --server.address 127.0.0.1` from `app/`.
  Streamlit exposes `pages/1_Staff_Review.py` as **Staff Review** in sidebar navigation (page title: ServicePilot AI｜员工审核台).
  The customer chat entry point is unchanged; no review tool is registered to Agent.
- On that page, enable **启用本地 DEMO 员工审核**, explicitly click
  **初始化 / 补充 DEMO 员工**, then choose **DEMO 员工（虚构）**. This adds the fixed
  staff identity `00000000-0000-4000-8000-000000000003` to the same dedicated
  `servicepilot_demo.sqlite3` used by customer confirmation. It does not create tickets.
  Initialization is append-only and repeatable; ID/subject collisions, incompatible
  databases, customer-role collisions and disabled staff are rejected, never promoted,
  overwritten or re-enabled. Existing customer initializer behavior stays unchanged.
- This account selector is local demonstration, not authentication. Anyone with
  access to this local page can select the demo employee; do not expose it publicly.
  The page does not accept arbitrary IDs, use customer session identity, model outputs,
  or chat text to grant review permissions. Backend mapping supplies identity.
- `TicketReviewQueryService` checks persisted account existence, active flag and
  STAFF role independently on every list/detail call, before accessing tickets.
  Reads use short-lived sessions and parameterized queries, return allowlisted DTOs,
  and never commit writes. Database errors are safe fixed responses with metadata logs.
- Default list is pending-review only; choose **全部** to view previous decisions.
  Results are limited to 100, ordered by creation time ascending then unique ticket
  number. No pagination is implemented yet. Detail includes order number, ticket
  number/type/description/status, created UTC time, review note and reviewed UTC time;
  it excludes owner ID, identity subject, request key, fingerprint and ORM internals.
- Select a ticket, enter a 1–2000-character review note and click **批准** or **拒绝**.
  Only the button branch calls existing `TicketReviewService`; it rechecks permissions
  and performs the atomic pending-state update. Success reruns and reloads database
  state. Reviewed tickets have no review buttons. Stale/concurrent clicks still hit
  the backend state guard and cannot overwrite an established decision.
- Approval is application approval only, with an explicit no-refund notice. Page
  errors never render exception payloads. No query results are cached across reruns.
