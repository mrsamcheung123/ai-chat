# PDF Desk

A small, CPU-only RAG chatbot for an AI course project. Users open a web address, ask questions about a collection of English PDFs, and see answers with document/page references. The host prepares a shared backend PDF library before visitors arrive. Browser uploads are an optional host tool.

All extraction, embedding, retrieval and answer generation run on the host machine. There is no hosted LLM API or API key. Internet access is needed to install dependencies and initially download models. Ordinary question answering uses local files only.

## Open it on this Mac

Double-click **Start.command** in this folder. The launcher opens the browser at **http://localhost:8501**. Keep the launcher window open while using the app. Stop it with Control-C or close that terminal window.

The dependencies and both models have been prepared in this project's `.venv/` and `models/` directories.

**Main workflow: host prepares PDFs; visitors simply chat.**

1. As the host, copy your prepared English PDFs into `/Users/cyril/Downloads/ProjectFile/data/pdfs/` (the `data/pdfs/` folder inside this project).
2. Double-click **Start.command**. It prepares the search index before opening the WebUI. Unchanged PDFs reuse the existing index.
3. Visitors open the website and ask questions. They do not upload or download PDFs and do not need accounts for chat.
4. To update the library, add, replace or remove PDFs in the same backend folder. Restart the server or refresh the app to update the index automatically.

The existing uploaded PDF collection is already in this backend folder and remains available to all visitors. No migration is needed. Source excerpts and page references appear with answers.

**Optional browser management:** expand **Host tools → Manage documents**. The host can download models if needed, inspect the collection, rebuild the index, or expand **Optional: upload PDFs through the browser**. These uploads add documents to the same shared library. Set `ADMIN_PASSWORD` before sharing the app to restrict document changes to the host.

The administrator handles installation and server startup. End users only need a browser and the deployed URL.

## Assignment deadline chat

Choose **Assignment deadlines** in the sidebar, then click its only action button, **Tell me when to submit**. This separate chat reads Markdown schedules only; it does not retrieve PDF passages or load the language model. Dates are parsed and filtered deterministically so the answer cannot invent deadlines.

Edit `/Users/cyril/Downloads/ProjectFile/assignments/schedule.md` directly. It contains clearly marked fictional examples, including an expired row. Replace them with your actual assignments before sharing the app. Add other `.md` files to the same folder if needed; each file must include a table with `Assignment` and `Due (GMT+8)` columns. Optional columns include `Course Code`, `Reminder`, and `Submission instructions`. Columns can be reordered and trailing reminders can be left blank. For example:

```markdown
| Assignment | Due (GMT+8) | Submission instructions |
| --- | --- | --- |
| AI project report | 2026-12-01 23:59 | Upload the report to the course portal. |
```

Use `YYYY-MM-DD HH:MM` in 24-hour GMT+8 time. Deadlines at or before the current instant are excluded; the remaining deadlines are sorted earliest first. The listed minute is the exact cutoff (23:59 means 23:59:00). Invalid dates produce a visible error instead of silently dropping an assignment. The button rereads the files every time, so Markdown edits do not require a server restart. The PDF and assignment chats have separate state.

The optional `ASSIGNMENTS_DIR` environment variable changes the schedule folder for native installations. Docker Compose mounts the project's `assignments/` folder read-only into the container, so the host can edit schedules on the VM without rebuilding the image. Restart/rebuild the application once when installing this new feature; subsequent schedule edits are read immediately on the next button click.

## Fresh installation

Use Python 3.11. The macOS launcher automates virtual-environment creation and dependency installation. The first installation needs internet access and a C/C++ compiler for `llama-cpp-python`; on macOS install Xcode Command Line Tools if needed. A Windows launcher, `Start.bat`, expects Python 3.11 and the C++ build tools. Docker is the more predictable installation option for a Linux/Azure host.

Manual administrator setup:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m rag.server --server.address=127.0.0.1
```

On Windows, the virtual-environment Python is `.venv\Scripts\python.exe`.

Copy `.env.example` to `.env` to customize settings. Set `ADMIN_PASSWORD` before sharing the application. The password protects model downloads and document changes; the chat itself is accessible to anyone who can reach the web address. With no password, document management is open for a local course demonstration.

## Azure deployment

These instructions target a **Linux Azure VM** with Docker Engine and the Compose plugin. They do not provision an Azure subscription, VM, domain or firewall rules. If the allocated environment is App Service or Container Apps instead, storage and startup configuration will need adapting.

Recommended starting size: **8 GB RAM and at least 2 vCPUs**, with sufficient persistent disk space. Reserve at least 5 GB free for model downloads, temporary files, PDFs and indexes, plus separate room for the Docker image/build. Four GB is a constrained target: try a 2,048-token context, keep one active generation, and measure memory and latency on the actual VM.

Administrator steps:

1. Copy the project source to the VM. You can omit `.venv/`, `models/`, `data/`, `work/` and caches; the Docker image excludes these anyway.
2. Copy `.env.example` to `.env` and set a strong `ADMIN_PASSWORD`.
3. Run `docker compose up -d --build`. The first build compiles the CPU model runtime and can take several minutes.
4. Access the app with an SSH tunnel while preparing it:

   ```sh
   ssh -L 8501:127.0.0.1:8501 your-user@your-vm
   ```

   Open http://localhost:8501 in your browser, expand **Host tools**, sign in to **Manage documents**, and download models if needed. Prepare your backend PDFs before sharing the URL:

   ```sh
   docker compose cp data/pdfs/. app:/app/data/pdfs/
   docker compose exec --user root app chown -R app:app /app/data/pdfs
   docker compose restart app
   ```

   The startup process indexes the prepared library. Visitors can then go straight to chat. The optional browser upload tool is another way for the host to maintain the same library.

The default Compose configuration binds the app to VM loopback. For an HTTPS website that classmates can open directly:

1. Point a domain/subdomain at the VM's public IP.
2. Set `APP_DOMAIN=chat.example.edu` in `.env` using your actual hostname.
3. Allow inbound TCP ports 80 and 443 in the Azure network security group and VM firewall.
4. Start the included reverse proxy:

   ```sh
   docker compose -f compose.yaml -f compose.public.yaml up -d --build
   ```

5. Open `https://your-hostname`. Caddy obtains an HTTPS certificate and proxies Streamlit, including its WebSocket connection.

Both containers restart automatically after a VM reboot when Docker is enabled at boot. Named volumes persist PDFs/indexes and models across container restarts. **Do not run `docker compose down -v` unless you intend to delete those volumes.** Back them up before deleting or replacing the VM. Source-folder `models/` and `data/` directories are separate from Docker's named volumes; downloading models on the Mac does not populate the VM's volumes.

Useful administration commands:

```sh
docker compose logs -f app
docker compose restart app
docker compose down
```

For a 4 GB VM, set these in `.env` before benchmarking:

```dotenv
APP_MEMORY_LIMIT=3g
RAG_CONTEXT_TOKENS=2048
RAG_THREADS=2
```

No response-time guarantee is made for an unspecified Azure CPU. A burstable VM can slow down once CPU credits are exhausted. The health check verifies that the WebUI responds; it does not mean models are downloaded or documents are indexed.

## Architecture and defaults

```text
PDFs -> pypdf -> page-preserving token chunks -> local BGE embeddings -> index.npz
Question -> local BGE embedding -> cosine search -> relevant passages -> local Qwen3 -> cited answer
```

| Part | Implementation |
|---|---|
| WebUI | Streamlit, one shared application process |
| PDF parsing | pypdf; physical PDF page numbers start at 1 |
| Embeddings | BAAI/bge-small-en-v1.5, quantized ONNX via FastEmbed, CPU provider |
| Answer model | Qwen3-1.7B, Q4_K_M GGUF from bartowski's quantization repository, about 1.28 GB |
| Inference | llama-cpp-python, CPU only; Qwen3 ChatML prompt with an empty thinking block |
| Chunking | 320 embedding-token chunks, 40-token overlap; chunks stay on one page |
| Retrieval | Normalized vectors and NumPy cosine similarity, up to 4 passages |
| Generation | 4,096-token context, 256-token maximum answer; prompt budget checked with the LLM tokenizer |
| Concurrency | One active generation, shared model lock; waiting requests time out after 60 seconds |
| Storage | Local PDFs and a single atomic NumPy/JSON index; no pickle loading |

Model repository revisions are recorded in manifests when downloaded. Subsequent starts reuse the local files. Downloading uses Hugging Face; normal model loading sets local-only flags and does not download missing files. Streamlit usage statistics and Hugging Face telemetry are disabled; ONNX Runtime telemetry events are disabled before session creation.

The index is rebuilt after additions, replacements or removals. Input validation and embeddings complete before the old collection is replaced. Failed updates preserve the previous working index. Changes made directly to stored PDFs are detected and indexed automatically at startup and on app refresh. If preparation fails, visitors see a temporary-unavailability message and the host can inspect the problem in host tools. Non-PDF files such as Finder metadata are ignored. Use one app process/replica with these local volumes: this version does not coordinate writes across multiple replicas.

## Scope and limits

- Up to 10 PDFs, 10 MB per file, 50 MB total; 20 pages per PDF and 100 pages total.
- English, selectable text. Encrypted PDFs and files with no extractable text are rejected. Empty/image-only pages in otherwise readable PDFs are skipped and reported.
- Source text is displayed as evidence. Tables with complex layouts, equations and image content may not extract accurately; OCR and visual document understanding are outside this version.
- Chat history is shown within the browser session and bounded to 20 exchanges. Each question is retrieved independently; topic-dependent follow-ups should repeat the relevant topic. History is not persisted across sessions or sent to the model.
- Retrieval below the configured similarity threshold abstains before loading the LLM. The default threshold (0.45) is a starting point, not a calibrated probability; tune it against your PDFs.
- The model is instructed to answer only from excerpts and cite them. Final answers with missing or nonexistent source IDs are replaced by an abstention. This checks reference existence, **not whether every claim is entailed by a passage**. Small local models can still make mistakes, so check source support in evaluation.
- No fine-tuning, external vector database, agent framework or GPU is required.

## Tests and course evaluation

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The default suite uses actual PDF extraction and persistent indexes, deterministic test embeddings, and Streamlit's UI test harness. It covers page references, long-page chunking, unsafe filenames, malformed/encrypted PDFs, failed updates, file replacement/removal, index changes, retrieval thresholds, citation checks, prompt boundaries, context budgeting and administrator access.

After models are downloaded, run the real-model check:

```sh
RUN_MODEL_TESTS=1 .venv/bin/python -m pytest tests/test_integration.py -q -s
```

This builds an isolated two-page test collection, answers a supported question, checks an out-of-scope question and blocks Python socket connections during normal use. It does not add test PDFs to your application collection. Timing and process memory measurements are written to `work/integration-report.json`. Results on the development Mac do not predict Azure performance or demonstrate quality on your actual PDFs.

For the course report, curate approximately 30 questions from your real PDFs, including unanswerable questions and questions requiring multiple passages. Use `evaluation/questions.example.json` as the schema, then run:

```sh
.venv/bin/python scripts/evaluate.py evaluation/your-questions.json --output evaluation/results.json
```

The script records retrieval hits, source pages, abstention behavior and response times. Manually grade answer correctness and citation support in the output. A retrieval hit means at least one expected page was retrieved; the separate `all_expected_pages_retrieved` field checks every expected page. The first question includes model cold-start time.

## Project layout

```text
app.py                    Browser interface and administrator screen
rag/config.py             Settings and resource limits
rag/documents.py          PDF validation, extraction and token chunking
rag/models.py             Explicit downloads and local model loading
rag/index.py              Index persistence, automatic backend refresh and retrieval
rag/server.py             Prepare the host library before starting the WebUI
rag/answer.py             Prompt budgeting, CPU generation and citation checks
tests/                    Automated checks
scripts/                  Optional administrator preparation/evaluation tools
evaluation/               Question-set example
Dockerfile, compose*.yaml  CPU deployment and optional HTTPS proxy
Start.command, Start.bat  Local launchers
models/, data/            Generated local storage (excluded from version control)
```

Model references: [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5), [FastEmbed ONNX model](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q), [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B), [GGUF quantization](https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF), [llama-cpp-python](https://llama-cpp-python.readthedocs.io/).
