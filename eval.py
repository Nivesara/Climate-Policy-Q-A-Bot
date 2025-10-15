"""Evaluate the Climate Policy chatbot on a small Q&A set.

Provide a CSV file with two columns: `question` and `gold` (the expected answer).
The script retrieves answers from the chatbot (using retrieval + optional LLM) and
writes a report to `eval_results.csv` containing the question, gold answer,
predicted answer, F1 score and retrieval latency.

Usage:
    python eval.py --csv questions.csv [--top-k 6] [--persona plain] [--limit 10]

F1 scoring is a simple token overlap measure; for more rigorous evaluation
consider using RAGAS or other metrics.
"""

import argparse
import csv
import os
import pickle
import string
import time

from typing import List

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

try:
    import openai
except ImportError:
    openai = None

INDEX_FILE = os.path.join(os.path.dirname(__file__), "index.pkl")
PERSONAS = {
    "plain": "You are a helpful assistant who explains climate policy in plain English. Avoid jargon and be concise.",
    "policy": "You are a policy analyst. Provide concise, formal responses and cite sources.",
    "journalist": "You are a climate journalist. Lead with the key finding, then support it with evidence from the context. Keep it readable.",
}


def load_index():
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    return data


def retrieve(question: str, data: dict, top_k: int) -> List[int]:
    vectorizer = data["vectorizer"]
    matrix = data["matrix"]
    q_vec = vectorizer.transform([question])
    sim = cosine_similarity(q_vec, matrix).flatten()
    return np.argsort(-sim)[:top_k].tolist()


def build_prompt(question: str, snippets: List[str], persona: str) -> str:
    persona_prompt = PERSONAS.get(persona, PERSONAS["plain"])
    context = "\n\n".join(snippets)
    return (
        f"{persona_prompt}\n\n"
        "Answer the question using only the information provided in the context."
        " If the context does not contain enough information, say 'I don't know based on the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def call_llm(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""
    model = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
    base_url = os.environ.get("OPENAI_BASE_URL")
    max_tokens = int(os.environ.get("MAX_TOKENS", 256))
    if openai is None:
        return ""
    openai.api_key = api_key
    if base_url:
        openai.base_url = base_url
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def f1_score(pred: str, gold: str) -> float:
    """Compute a simple F1‑like overlap between two strings."""
    trans = str.maketrans('', '', string.punctuation)
    pred_tokens = set(pred.lower().translate(trans).split())
    gold_tokens = set(gold.lower().translate(trans).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(csv_path: str, top_k: int, persona: str, limit: int) -> None:
    data = load_index()
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            q = row['question']
            gold = row['gold']
            t0 = time.time()
            indices = retrieve(q, data, top_k)
            snippets = [data['chunks'][idx] for idx in indices]
            prompt = build_prompt(q, snippets, persona)
            answer = call_llm(prompt)
            if not answer:
                answer = "\n\n".join(snippets)
            latency = time.time() - t0
            score = f1_score(answer, gold)
            rows.append({
                'question': q,
                'gold': gold,
                'answer': answer,
                'f1': round(score, 3),
                'latency_s': round(latency, 2),
                'sources': "; ".join([f"{data['metadata'][idx]['source']} p{data['metadata'][idx]['page']}" for idx in indices]),
            })
    out_path = 'eval_results.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Climate Policy Chatbot")
    parser.add_argument('--csv', required=True, help='CSV file with question and gold columns')
    parser.add_argument('--top-k', type=int, default=6, help='Number of passages to retrieve')
    parser.add_argument('--persona', choices=list(PERSONAS.keys()), default='plain', help='Persona style')
    parser.add_argument('--limit', type=int, default=0, help='Maximum number of questions to evaluate (0 = all)')
    args = parser.parse_args()
    evaluate(args.csv, args.top_k, args.persona, args.limit)


if __name__ == '__main__':
    main()