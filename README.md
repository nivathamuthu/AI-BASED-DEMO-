# DocIntel Lab: Document AI Demos for Aviation (Django + LangChain/LangGraph)

## Purpose (read this first, Cursor)
I am a Django developer interviewing at a product company that builds a document-AI platform for airlines and aircraft manufacturers. They already have an "AI Query" interface. The hiring manager asked for **proof (demo videos) that I can add document-AI capabilities, not just Django CRUD**.

Build ONE small repo with **8 small, working modules** on a shared **aviation document corpus**. They can be simple (minimal UI, Django templates or Streamlit-style pages), but each must **run end to end and be demo-able in 2-3 minutes**.

Pitch line: *"I'm a Django developer who also turns documents into knowledge, rules, code and agents. Here are 8 things I can add to your platform."*

---

## How to run

The app, the Celery worker, Postgres, Redis, Neo4j and the tests all run in Docker. There is no host virtualenv. `.\make.ps1` is the task runner. It creates `.env` from `.env.example` on the first `up`.

```powershell
.\make.ps1 up        # build, start web + worker + postgres + redis + neo4j, migrate
.\make.ps1 test      # pytest inside the web container
.\make.ps1 demo      # synthetic PDF, DOCX and PPTX, then ingest them
```

Then open http://localhost:8010/ingest/

`LLM_PROVIDER=mock` and `EMBEDDINGS_PROVIDER=mock` are the defaults, so this works with no API key. The worker uses Celery's prefork pool. Code is bind-mounted: edit a file and the dev server reloads; restart the worker (`docker compose restart worker`) after changing task code. Rebuild the image only when `requirements/` changes.

Equivalent commands without the script:

```powershell
docker compose up -d --build
docker compose exec web python manage.py migrate
docker compose exec web pytest
docker compose exec web python scripts/seed_demo.py
```

---

## Tech stack
- Python 3.11, Django 5 (shell app, admin, simple templates + HTMX)
- LangChain + LangGraph (agents / workflows)
- LLM: Claude or OpenAI via env var `LLM_PROVIDER` (keep a thin wrapper in `core/llm.py`)
- Embeddings + vector store: Chroma or pgvector
- Keyword search: `rank_bm25` (or Postgres full-text)
- Reranker (optional): `sentence-transformers` cross-encoder
- Knowledge graph: Neo4j (Docker) or NetworkX + pyvis for the zero-setup fallback
- Parsing: `pypdf`, `python-docx`, `python-pptx`, `unstructured` (optional)
- Pipelines: Celery + Redis (or Prefect) for async ingestion
- Pydantic v2 for ALL structured LLM outputs
- Tests: pytest. Eval: simple scripts with a results table

## Sample corpus (public, safe to use)
Put in `data/corpus/`. Use free public aviation docs, for example:
- FAA Advisory Circulars (maintenance, MEL, airworthiness)
- EASA Part-145 / Part-M excerpts
- A public Airbus/Boeing maintenance or airworthiness directive PDF
- 1 self-made PPTX and 1 DOCX (a fake "Maintenance Planning Procedure") to prove multi-format support

---

## Repo layout
```
docintel-lab/
  manage.py
  config/                 # Django settings
  core/                   # llm.py, embeddings.py, schemas.py, prompts loader
  apps/
    ingest/               # Module 1 + 2: pipeline
    rag/                  # Module 3: hybrid retrieval + AI query UI
    kg/                   # Module 4: knowledge graph
    rules/                # Module 5: rules / business logic extraction
    qa_gen/               # Module 6: Q&A generation + eval
    codegen/              # Module 7: rules -> code
    agents/               # Module 8: LangGraph compliance agent
  prompts/                # versioned prompt templates (Jinja2 .md files)
  data/corpus/
  evals/
  docs/demo_scripts/
  README.md
```

---

## Modules (build in this order)

### Module 1: AI Data Pipeline (ingestion)
**What it proves:** I understand what an "AI data pipeline" really is.
Flow: `upload -> detect type -> parse -> clean -> chunk (structure-aware) -> metadata enrich -> embed -> index (vector + BM25 + graph)`.
- Celery tasks per stage, status shown in a Django page (Queued / Parsing / Embedding / Done / Failed)
- Idempotent (file hash dedupe), retry on failure, per-stage timing logs
- Chunking: split by headings/sections, keep page number + section title as metadata
- Show a pipeline diagram (Mermaid) in `docs/`
**Demo:** upload a PDF, DOCX and PPTX; watch the stages complete; show chunks with metadata.

### Module 2: Prompt Template Library
**What it proves:** "Prompt engineering" means engineered, versioned, tested assets, not ad hoc chat.
- `prompts/<task>/v1.md`, `v2.md` in Jinja2 with variables (`{{ domain }}`, `{{ chunk }}`, `{{ schema }}`)
- Loader that picks version, injects few-shot examples, enforces Pydantic output (retry on validation failure)
- Mini eval: run v1 vs v2 on 10 samples and print a score table
**Demo:** show a bad prompt vs an engineered prompt on the same document, with the metric difference.

### Module 3: Hybrid Retrieval RAG ("better AI Query")
**What it proves:** I can upgrade their existing AI Query interface.
- Retrieval = BM25 + vector, merged with Reciprocal Rank Fusion, optional cross-encoder rerank
- Answers with **citations** (doc, page, section) and "I don't know" when context is weak
- Toggle in UI: Vector-only / BM25-only / Hybrid / Hybrid+Rerank
- Eval: 15 questions, report hit@5 for each mode
**Demo:** ask an exact-term query (e.g. a part number or "AD 2023-xx") where vector-only fails and hybrid wins.

### Module 4: Automatic Knowledge Graph from Documents
**What it proves:** I can build a graph with no manual modeling.
- LLM extracts entities + relations per chunk via Pydantic schema. Aviation ontology: `Aircraft, Component, Task, Regulation, Procedure, Part, Interval`; relations like `APPLIES_TO, REQUIRES, GOVERNED_BY, REPLACES, HAS_INTERVAL`
- Entity resolution (normalize names, merge duplicates), keep source chunk for each edge
- Store in Neo4j (or NetworkX), render interactive graph (pyvis) in Django
- **GraphRAG query:** question -> find entities -> traverse 1-2 hops -> combine with retrieved chunks -> answer
**Demo:** upload docs, graph appears automatically, ask a multi-hop question ("Which tasks are governed by regulation X for component Y?").

### Module 5: Rules / Business Logic Extraction from Documents
**What it proves:** Documents (PDF, PPTX, DOCX, transcripts) -> structured rules, automatically.
- Prompt extracts rules as JSON: `{id, condition, action, exception, source_doc, page, confidence}`
- Example output: *IF component is life-limited AND flight hours >= limit THEN replace; exception: ...*
- Dedupe/merge conflicting rules, flag contradictions between documents
- Review screen in Django: approve / edit / reject rules; export as JSON/CSV
**Demo:** feed a maintenance procedure PPTX and get a reviewable rule table with source citations.

### Module 6: Q&A Generation + Evaluation Set
**What it proves:** Same documents -> training/eval/FAQ content.
- Generate question-answer pairs per section (factual, multi-hop, "unanswerable")
- Quality filter with an LLM judge (groundedness score) and drop weak ones
- Auto-use this set to evaluate Module 3 (closes the loop: generate -> evaluate -> improve)
**Demo:** generate 50 Q&As, filter, then run them through the RAG and show accuracy.

### Module 7: Rules -> Working Code (AI-assisted software generation)
**What it proves:** Prompt-driven code generation with verification.
- Input: approved rules from Module 5
- Output: Python functions or Django validators + pytest tests, generated by LLM
- **Self-repair loop (LangGraph):** generate -> run tests in sandbox -> feed errors back -> regenerate (max 3 tries)
- Show generated code + test results in the UI
**Demo:** pick 3 rules, watch code and tests generated and passing.

### Module 8: Agentic Compliance Assistant (LangGraph)
**What it proves:** Real LLM-powered agent workflow with tools, state and human-in-the-loop.
- Graph nodes: `classify_request -> plan -> retrieve (hybrid) -> graph_lookup -> rule_check -> draft_answer -> verify_citations -> (human approval) -> final`
- Tools: hybrid search, KG query, rules lookup
- Persistent state (checkpointer), conditional edges (re-retrieve if verification fails), full trace of steps shown in the UI
- Example task: "Is task T allowed to be deferred on aircraft type A under the MEL? Cite sources."
**Demo:** show the step-by-step trace, a failed verification looping back, then the final cited answer.

---

## Cursor instructions
1. Work **module by module**, in the order above. After each module: runnable, with `README` section, 1 sample command and 1 screenshot-ready page.
2. All LLM outputs go through **Pydantic schemas** with validation + retry. No free-text parsing with regex.
3. Keep Django thin: models for Document, Chunk, Rule, QAPair, PipelineRun; business logic in service modules.
4. Add `Makefile` targets: `make up` (docker: postgres, redis, neo4j), `make ingest`, `make demo`.
5. Include `.env.example`, seed script `scripts/seed_demo.py` that loads the corpus so a demo starts in one command.
6. Add minimal tests for each module (schema validity, retrieval fusion math, graph merge, codegen loop).
7. Do not over-build UI. Clean, simple tables and one graph view is enough.

## Suggested schedule (5 days)
| Day | Deliverable |
|-----|-------------|
| 1 | Repo skeleton, Module 1 (pipeline) + Module 2 (prompts) |
| 2 | Module 3 (hybrid RAG + eval) |
| 3 | Module 4 (knowledge graph + GraphRAG) |
| 4 | Module 5 (rules) + Module 6 (Q&A) + Module 7 (codegen) |
| 5 | Module 8 (agent), polish, record videos |

## Demo video plan (one 8-10 min video, or 8 short clips)
Open with 20 seconds: *"I'm a Django developer, but here is what else I can add to your product."* Then one clip per module (60-90 seconds each): problem, live demo, result/metric. Close with: *"These are aviation examples because your product serves aviation; the same pipeline works for insurance and finance, where I've already applied it."*

Always show **numbers** (hit@5, rules extracted, tests passed) and **citations**. That is the proof.

## Mapping to the CV claims
| CV claim | Proof in this repo |
|----------|--------------------|
| Generative AI for documents | Modules 5, 6, 7 |
| Agentic AI / LangGraph | Modules 7, 8 |
| LangChain | `core/` wrappers, chains in Modules 3-6 |
| Hybrid retrieval | Module 3 (with eval) |
| Knowledge graphs | Module 4 |
| AI data pipelines | Module 1 |
| Prompt engineering / templates | Module 2 + prompts/ folder |

---

## Module 1: Ingest

Upload a file and the worker runs seven tasks: detect, parse, clean, chunk, enrich, embed, index. Status, attempts and per-stage milliseconds are on the document page. The same sha256 is not stored twice. Chunks keep page, section path and the identifiers found in the text (an AD number, a part number). Vectors are 384-dimensional in pgvector. With `EMBEDDINGS_PROVIDER=mock` the vectors are a deterministic hash, so the page works offline.

```powershell
.\make.ps1 up
.\make.ps1 test
.\make.ps1 demo
```

Optional public FAA circulars, then a second ingest that picks them up:

```powershell
docker compose exec web python scripts/fetch_corpus.py
.\make.ps1 ingest
```

To show the retry on screen, set `INGEST_CHAOS_STAGE=parse` in `.env`, run `docker compose up -d web worker`, and click Reprocess. The parse row ends on attempt 2. Set the variable back to empty afterwards.

The talk track is `docs/demo_scripts/m1.md`. The stage diagram is `docs/pipeline.md`.

---

## Module 2: Prompts

`prompts/<task>/v1.md` is a one-line prompt. `v2.md` sets the role, the rules, the JSON schema and the few-shot examples in `examples.yaml`, and it must answer "unknown" when the chunk does not contain the answer. Outputs are Pydantic objects. A validation failure is sent back to the model and retried, and every call is a row in `LLMCall`.

```powershell
docker compose exec web python manage.py eval_prompts
```

The score table is printed, saved under `evals/results/`, and stored on `EvalRun` with the LLM provider and model. Open http://localhost:8010/prompts/ for the side-by-side comparison. The talk track is `docs/demo_scripts/m2.md`.

## Module 3: Hybrid retrieval

Ask a question at http://localhost:8010/rag/. The mode toggle is vector, BM25, hybrid (reciprocal rank fusion), or hybrid+rerank. Answers cite the document, page and section. Weak context returns "I don't know."

`RERANK_ENABLED` defaults to false, so hybrid+rerank uses a lexical rerank. The cross-encoder and local MiniLM embeddings are not in the default image. Build them with:

```powershell
docker compose build --build-arg INSTALL_ML=1
```

OpenAI embeddings stay 384-dimensional (`text-embedding-3-small`), so the pgvector column does not change:

```powershell
# in .env
EMBEDDINGS_PROVIDER=openai
OPENAI_API_KEY=sk-...
docker compose up -d web worker
docker compose exec web python manage.py reembed
```

Chunks already in the database were embedded with whichever provider was active at ingest. Switching provider requires `reembed`, because the vectors are not in the same space.

```powershell
.\make.ps1 eval
```

That writes hit@5 for all four modes. The report names the LLM and the embedding provider and model. The talk track is `docs/demo_scripts/m3.md`.

## Module 4: Knowledge graph and GraphRAG

Each chunk goes through `prompts/extract_kg/v2.md` (v1 is the naive version). The output is validated against `KGOutput` (entity types Aircraft, Component, Task, Regulation, Procedure, Part, Interval; relations APPLIES_TO, REQUIRES, GOVERNED_BY, REPLACES, HAS_INTERVAL, PART_OF) and retried on failure. A relation is dropped, and counted, when an endpoint is missing from the chunk's entity list or its evidence quote is not in the chunk. Results are cached in `KGExtraction`, one row per chunk with its prompt version.

Entity resolution unifies case, punctuation, hyphens, part-number prefixes (`PN-77-4418` = `P/N 77-4418`) and flight-hour spellings, then merges names that share a key or an alias. An alias is accepted only when its normalized identifier tokens (the tokens with digits) equal the name's, so `77-4418` and `P/N 77-4418` merge, while a serial range (`S1000 through S1899`) or another part number (`PN-77-4410`) never merges into a part. Rejected aliases are listed in the merge log (stored on `GraphBuild`) and printed by `build_graph`. Every node and edge carries its source chunk id, document and page.

The store is Neo4j (`GRAPH_BACKEND=neo4j`, the compose service, browsable at http://localhost:7474) or a NetworkX JSON file (`GRAPH_BACKEND=networkx`). When an ingest run finishes, the worker extracts the new chunks and rebuilds the graph (`KG_AUTO_BUILD=true`).

```powershell
docker compose exec web python manage.py extract_kg            # cached chunks are skipped; --force re-extracts
docker compose exec web python manage.py build_graph           # rebuild from KGExtraction, no LLM calls
docker compose exec web python manage.py build_graph --extract --force --prompt-version v1
docker compose exec web python manage.py eval_kg
```

http://localhost:8010/kg/ shows the interactive graph (pyvis), a type filter, the node, edge, merged and dropped counts, and a node panel with source chunks and citations. The GraphRAG box links question entities by name, alias and embedding similarity, walks one or two hops, adds the linked chunks to the Module 3 hybrid results, and answers with citations. The traversed path is shown under the answer, next to the hybrid-only answer.

`eval_kg` runs the 8 multi-hop questions in `evals/datasets/kg_questions.yaml` through plain hybrid RAG and through GraphRAG, saves the table to `EvalRun` and `evals/results/`, and names the LLM, the embeddings and the graph build. The talk track is `docs/demo_scripts/m4.md`.

## Module 5: Business rules

Every chunk of every finished document (PDF, DOCX, PPTX, text) goes through `prompts/extract_rules/v2.md`. v1 is the naive one-line version. The output is a `RuleSet` of `{condition, action, exception, source_quote, confidence}`, validated by Pydantic and retried on failure. A rule is kept only when its `source_quote` appears verbatim in the chunk (whitespace and typographic quotes are folded, wording and case are not). Dropped rules are logged and counted. The validated output is cached in `RuleExtraction` per chunk and prompt version, so an unchanged chunk is not sent again unless you pass `--force`.

Rules get stable keys (`R-0001`, ...). A re-extraction that finds the same normalized condition and action, or the same quote, keeps the key and the review state. A pending rule that disappears is retired, not deleted. Each rule keeps its document, chunk, page, section, prompt version, model, and its raw model output.

After extraction, near-duplicates are merged into the highest-confidence rule. A merge needs the same normalized condition and action, or an embedding cosine of at least `RULES_MERGE_SIMILARITY` (0.9) with word overlap. Rules with different numbers are never merged. The merge log is stored on `RuleRun`. Conflict candidates are pairs that share a part or directive number, or whose conditions have a cosine of at least `RULES_CONFLICT_SIMILARITY` (0.6). Only candidates go to the judge (`prompts/judge_rule_conflict/v1.md`, schema `ConflictJudgment`). Each verdict is stored as a `RuleConflict` with kind, severity and reason, and an unchanged pair is not judged again.

The synthetic corpus includes `engineering_order_EO-2291.docx`, a one-paragraph order that gives 11500 flight hours for part 65-45210-3 where the procedure says 12000. It exists so the conflict check has a real contradiction to find.

```powershell
.\make.ps1 extract-rules        # = docker compose exec web python manage.py extract_rules [--force] [--prompt-version v1] [--document 11]
.\make.ps1 eval-rules           # = docker compose exec web python manage.py eval_rules [--force]
```

http://localhost:8010/rules/ shows the counts (extracted, dropped because the quote was not found, merged, conflicts, approved), the LLM and embedding provider and model, a banner when the LLM is the mock, the conflicts, the rules table with status, document and confidence filters, and the merge log. "Extract rules" queues the same pipeline on the worker. A rule's page highlights the quote inside the full chunk text and has Approve, Reject (with a reviewer note) and Edit. An edit keeps the original wording. `/rules/export.json` and `/rules/export.csv` export approved rules with their source and model.

`eval_rules` scores v1 and v2 against the 12 hand-written rules in `evals/datasets/rules_gold.yaml` (precision = matched / kept, recall = matched / 12). The report names the LLM and lists every unmatched prediction, since the gold set is not exhaustive. The talk track is `docs/demo_scripts/m5.md`.

## Module 8: Compliance agent (LangGraph)

`apps/agents/graph.py` is a LangGraph `StateGraph`:

```
classify -> plan -> hybrid_retrieve -> graph_lookup -> rule_check -> draft -> verify_citations -> human_approval -> final
classify --out_of_scope--> refuse
verify_citations --failed, loops < AGENT_MAX_LOOPS (2)--> hybrid_retrieve (with a refined query)
```

Every LLM node goes through `core.llm.structured_call` (same provider, backoff and concurrency limit as the other modules) with a Pydantic schema from `apps/agents/schemas.py` and a prompt under `prompts/agent_*/v1.md`:

- `classify` returns `{intent: lookup | compliance_check | conflict_check | out_of_scope, entities}`. Out of scope goes to a fixed, polite refusal with no further calls.
- `plan` returns the tools to run. The schema rejects a plan without `hybrid_search` or with a tool twice. Compliance and conflict questions always get `rule_lookup` added (the step output says so when that happens).
- The tools in `apps/agents/tools.py` are plain functions with typed keyword arguments: `hybrid_search(query, k, mode)` (Module 3, `AGENT_RETRIEVAL_MODE`, default `hybrid_rerank`, which tied BM25 for the best hit@1 and MRR on the real eval_rag run), `kg_query(question, hops, limit)` (Module 4 entity linking and 2-hop traversal), and `rule_lookup(question, entities, limit)` (Module 5 approved and pending rules with their conflict flags; a conflicting partner rule is always included).
- `rule_check` asks the model, per candidate rule, whether its condition is met (`applies`) and whether its outcome bears on the answer (`relevant`). Conflicts of applicable or relevant rules are carried to the draft and the reviewer.
- `draft` answers only from the numbered sources (search, graph and rule chunks). Each claim has a source number and a verbatim quote. `cannot_determine` is a valid status.
- `verify_citations` checks that every cited source is in the retrieved set and that its quote is in that chunk (no LLM), then asks a judge (`Verification`) whether each quote supports its claim. If an applicable rule has a conflict, the answer must say so. A failure loops back to retrieval with the judge's refined query, at most twice, then the draft goes to approval with a warning.
- `human_approval` calls `interrupt()`. The state is saved by `PostgresSaver` (`langgraph-checkpoint-postgres`) in the app database; the thread id is `AgentRun.thread_id`. Approve or Reject (a note is required) resumes the thread on the worker.

`AgentRun` and `AgentStep` keep each node's input, output, tool calls (args, result summary, ms), LLM calls (schema, attempts, latency, tokens), duration and loop number.

http://localhost:8010/agents/ has the question box, the provider and model, and recent runs. A run page shows the trace while the worker runs (HTMX, every 1.5 s), the loop count, the pending-approval panel with the conflict warning and the citation check, and the final answer with a citation table (claim, quote, document, page, section, verified).

```powershell
.\make.ps1 ask-agent "Part 65-45210-3 is at 11,800 flight hours on a narrow-body aircraft. Is it due for replacement now?"
.\make.ps1 eval-agent     # = docker compose exec web python manage.py eval_agent [--case conflict-limit]
```

`eval_agent` runs the 6 cases in `evals/datasets/agent_cases.yaml` (2 lookups, 2 compliance checks, the 11500 vs 12000 conflict, 1 out of scope), approves every paused run itself, and reports per case: intent correct, answer correct (required strings), citations verified, conflict surfaced, loops used. The report names the LLM and is saved to `EvalRun` and `evals/results/`. The talk track is `docs/demo_scripts/m8.md`.

#   A I - B A S E D - D E M O -  
 