# Climate Policy Explainer Bot

This project provides a simple retrieval‑augmented chatbot focused on climate policy.  
It ingests a small set of policy documents (PDFs) such as the IPCC AR6 summary, the Paris Agreement, UNEP Emissions Gap reports and related briefings, indexes them locally, and answers questions using only the information contained in those documents.  

The pipeline is intentionally lightweight so it can run on a laptop without a GPU.  It uses Python built‑ins and `scikit‑learn` for vector search.  For answer generation you can plug in any OpenAI‑compatible LLM; the supplied code includes a placeholder for calling OpenAI's chat completions API.  If you run locally with a provider such as [Ollama](https://ollama.ai/), set the appropriate environment variables as described below.

## Quick start

1. **Install dependencies** (Python ≥ 3.9 recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Add your climate PDFs** into the `docs/` folder.  Five example files (IPCC AR6 SPM, Paris Agreement, UNEP Emissions Gap Report 2024, EU Fit for 55 briefing and CMA 2023 NDC synthesis) are included.  You can drop additional PDF or plain‑text files into this directory and they will be indexed.

3. **Ingest documents** to build the search index.  Run:

```bash
python ingest.py
```

This extracts text from each PDF, splits it into overlapping chunks, computes TF‑IDF vectors and stores them in `index.pkl`.  Re‑run the script if you add new documents.

4. **Ask questions** about your documents with:

```bash
python app.py --question "What temperature goal is stated in the Paris Agreement?"
```

The script retrieves the most relevant chunks, optionally summarizes them via OpenAI, and prints an answer with citations.

## Environment variables

To enable answer generation via an LLM, set the following variables in your shell or a `.env` file:

- `OPENAI_API_KEY` – your OpenAI key or the placeholder `ollama` when using an OpenAI‑compatible local server (e.g., via [Ollama](https://ollama.ai/)).
- `OPENAI_BASE_URL` – override to point to a local OpenAI‑compatible server (for Ollama: `http://localhost:11434/v1`).  Leave unset to use OpenAI's cloud API.
- `LLM_MODEL` – model name to request (e.g. `gpt-3.5-turbo` for OpenAI; `llama3:8b` for Ollama).  Ignored if no key is set.
- `MAX_TOKENS` – maximum tokens to generate; default 256.

If `OPENAI_API_KEY` is not set, the app simply returns the retrieved context paragraphs as the answer.

## Files

| File | Purpose |
| --- | --- |
| `ingest.py` | Parses PDFs in `docs/`, splits them into chunks and builds a TF‑IDF vector index. |
| `app.py` | CLI tool that loads the index, retrieves the top‑k chunks for a question, optionally calls an LLM to summarize, and prints the answer with citations. |
| `eval.py` | Mini evaluation harness for your own question/answer pairs; writes results to `eval_results.csv`. |
| `requirements.txt` | Python dependencies; includes `scikit‑learn`, `pymupdf` and optional `openai` for LLM calls. |

## Extending the project

* **Reranker:** The retrieval in this starter uses TF‑IDF cosine similarity.  To improve quality, you can add a cross‑encoder reranker (e.g. `sentence-transformers`' MiniLM) to reorder the top‑k passages.  Simply import the model in `app.py` and call it on the candidate snippets.

* **Persona control:** Modify the `build_prompt` function in `app.py` to prepend a persona‐specific system message (e.g. Plain English vs Policy‑maker), and expose a command‑line flag to switch personas.

* **Streamlit UI:** Build a simple Streamlit interface that calls the same retrieval code so non‑technical users can chat with the bot.  The skeleton here is CLI‑only for simplicity.

* **Evaluation:** Use `eval.py` to automatically test your bot on a set of questions.  Provide a CSV file with columns `question` and `gold` (gold standard answer).  The script will compute a simple F1‑like token overlap score and record latency and retrieved sources.

## License

This project is released under the MIT License.  The included documents are public, but be sure to respect their original copyrights when distributing.