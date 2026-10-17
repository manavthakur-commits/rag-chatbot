# Northwind Knowledge Assistant — RAG Chatbot for Documents (PDF / TXT / CSV)

Northwind Knowledge Assistant lets you chat with PDFs, text files, and CSVs. Every answer
cites the retrieved document chunks it came from, grounding responses in your
files instead of outside knowledge.

## Try it online

Open the live [Northwind Knowledge Assistant dashboard](https://rag-chatbot-manav.streamlit.app/).
Enter your OpenAI API key, upload PDF, TXT, or CSV documents, and build a
knowledge base to start asking questions.

## What it does

- Upload PDF, TXT, and CSV files
- Loads → chunks → embeds → indexes them in a local vector store
- Ask questions in the Northwind Knowledge Assistant chat UI; answers are generated **only**
  from retrieved content, with inline `[source#chunk_id]` citations
- If nothing relevant was retrieved, it says so instead of guessing
- Shows retrieved source chunks with a text preview below each answer

## How it works

```
 upload           load            chunk              embed + index         retrieve + answer
┌──────────┐    ┌─────────┐    ┌───────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ file.pdf │───▶│ loaders │───▶│   chunking    │───▶│    vectorstore    │───▶│         rag          │
│ file.txt │    │  .py    │    │     .py       │    │  (Chroma, cosine)  │    │ retrieve top-k above │
│ file.csv │    └─────────┘    │ prose: token- │    └──────────────────┘    │ relevance threshold, │
└──────────┘                   │ aware splitter│                            │ then ask the LLM using│
                                │ CSV row: kept │                            │ ONLY that context     │
                                │ atomic        │                            └─────────────────────┘
                                └───────────────┘
```

Two implementation details worth calling out (this is what most "chat with
your PDF" tutorials skip):

1. **Chunking is type-aware.** Prose (PDF/TXT) is split by *token* count with
   overlap (`src/chunking.py`, via tiktoken) — token count is what actually
   determines context-window usage, not character count. CSV rows are kept
   as one chunk each, since splitting a row by length would sever a record's
   fields from each other for no benefit.
2. **Retrieval has a relevance floor.** Answers only use chunks whose cosine
   similarity to the question clears `score_threshold` (default `0.25`,
   `src/rag.py`). Below that, the pipeline says it doesn't know rather than
   forcing an answer out of whatever ranked highest.

## Dashboard

The Streamlit dashboard is branded as **Northwind Knowledge Assistant**, with a responsive
hero header, API-key and document-upload panels on the main page, and a cited
chat interface. The two setup panels sit side by side on wide screens and
stack on narrow screens. Subtle entrance and hover animations add feedback;
animations are reduced for users whose operating system requests reduced
motion.

## Run the project

### Requirements

- Python 3.10 or later and pip
- An OpenAI API key with access to the API for document embeddings and generated answers
- An internet connection when installing packages and using the OpenAI API

Run all commands from the project root—the directory containing `app.py` and
`requirements.txt`. The virtual environment keeps this project's packages
separate from other Python projects.

### 1. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
```

If `py` is not available but `python` is, use `python -m venv .venv`.

**macOS / Linux:**

```bash
python3 -m venv .venv
```

### 2. Install the project dependencies

Use the virtual environment's Python to install the pinned dependencies from
the project root.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Using the environment's Python directly avoids installing packages into a
different Python installation by mistake. If you prefer to activate the
environment, use `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then run the commands with `python`.

### 3. Start the dashboard

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux:**

```bash
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually
`http://localhost:8501`. Keep the terminal running while you use the app.

### 4. Add your API key and documents

In the dashboard, enter your OpenAI API key in the **OpenAI API key** panel
and select **Apply API key**. The key is held for that browser session and is
not written to a file. Alternatively, set the `OPENAI_API_KEY` environment
variable before starting the app.

Keep your API key private. Do not commit it to source control or share
screenshots that show it.

Upload one or more PDF, TXT, or CSV files and select **Build knowledge base**.
Then ask a question in the chat. Answers cite retrieved document content; if
the content does not support an answer, the assistant says it does not have
enough information.

### Use the project in VS Code

Open the project root folder in VS Code. Open the Command Palette
(`Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS), run
**Python: Select Interpreter**, and select the interpreter inside this
project's `.venv`:

- Windows: `.venv\Scripts\python.exe`
- macOS / Linux: `.venv/bin/python`

If the `.venv` interpreter is not listed, create the environment and install
the dependencies using the commands above, then select it. In the VS Code
integrated terminal, confirm the selected environment is active or use its
Python executable explicitly when launching the app.

### Troubleshooting

- **`ModuleNotFoundError`, such as `No module named 'langchain_text_splitters'`:**
  Make sure the app is being launched with this project's `.venv` Python and
  install the dependencies again from the project root:
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` (Windows) or
  `.venv/bin/python -m pip install -r requirements.txt` (macOS/Linux).
  The install name `langchain-text-splitters` uses hyphens in
  `requirements.txt`; its Python import name uses underscores.
- **PowerShell says script execution is disabled:** You do not need to
  activate the environment. Use the `.venv\Scripts\python.exe` commands above
  directly.
- **The app says the API key is missing or an OpenAI request fails:** Apply a
  valid API key in the dashboard and check that the key and OpenAI account can
  access the API. Keep the terminal output available when diagnosing API
  errors.
- **Port 8501 is already in use:** Stop the other Streamlit app using that
  port, or start this one on another port with
  `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502`
  (Windows) or
  `.venv/bin/python -m streamlit run app.py --server.port 8502`
  (macOS/Linux).

## Project layout

```
app.py                      Streamlit UI
requirements.txt            Application dependencies
src/
  loaders.py                 PDF/TXT/CSV → LangChain Documents
  chunking.py                Type-aware chunking (token-based prose, atomic CSV rows)
  vectorstore.py             Chroma persistence + relevance-score filtering
  rag.py                     Pipeline: ingest() and answer()
  llm.py                     OpenAI provider functions
```

## Swapping the LLM provider

Every LLM and embedding call is created through the provider functions in
`src/llm.py`. To use a different provider, adapt those functions and the
provider-specific API-key handling as needed; the RAG pipeline and UI are
otherwise isolated from the provider SDK.

## Where this is intentionally scoped down

This is the "core scope, fully working" version, not the maximal one. Known
directions to extend it (good material for "what would you improve?" in an
interview):

- Re-ranking retrieved chunks with a cross-encoder before generation
- Table-aware PDF parsing (tables currently flatten to plain text)
- Multi-hop questions that need synthesizing facts across several chunks
- Conversation memory across turns — see the companion "Conversational RAG"
  project, which builds directly on this one

## Known limitation

`langchain-community` (used for the PDF/TXT/CSV loaders) prints a
deprecation warning on import — LangChain is migrating loaders to standalone
packages. It still works correctly; this is just upstream churn to be aware
of, not a bug in this project.
