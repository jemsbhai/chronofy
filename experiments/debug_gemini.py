"""Opt-in diagnostic for four TIME-Lite prompts against the configured Gemini model."""

from __future__ import annotations

import json
import os

from experiments.reproducibility import GEMINI_MODEL_ID, TIME_LITE_REVISION

SYSTEM_PROMPT = (
    "You are answering a multiple-choice temporal reasoning question.\n\n"
    "INSTRUCTIONS:\n"
    "1. Read the provided context carefully, paying attention to dates "
    "and temporal relationships.\n"
    "2. Select the best answer from the options given.\n"
    "3. Respond in EXACTLY this JSON format (no other text):\n\n"
    '{"answer": "A", "confidence": 0.85, "reasoning": "Short reason."}\n\n'
    '- "answer": single letter A, B, C, D, or E\n'
    '- "confidence": number 0.0 to 1.0\n'
    '- "reasoning": MAX 20 words. Be extremely brief.\n\n'
    "Output ONLY the JSON object. No markdown. No backticks. No extra text."
)


def main() -> None:
    """Run the network/API diagnostic only when invoked directly."""
    import google.generativeai as genai
    from datasets import load_dataset

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this diagnostic.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_ID,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    dataset = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    news_rows = [row for row in dataset["train"] if row["Dataset Name"] == "TIME-Lite-News"]

    for index in [5, 7, 0, 1]:
        row = news_rows[index]
        prompt = (
            f"Context:\n{row['Context']}\n\nQuestion:\n{row['Question']}\n\n"
            "Respond with JSON only."
        )
        print(f"\n=== Q{index} (ctx={len(row['Context'])}c, gold={row['Gold Answer']}) ===")
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            finish_reason = response.candidates[0].finish_reason
            print(f"finish_reason={finish_reason}, raw ({len(raw)}c): {raw[:250]!r}")
            parsed = json.loads(raw)
            print(f"PARSED: answer={parsed['answer']}, conf={parsed['confidence']}")
        except Exception as error:  # Diagnostic: preserve provider error type/message.
            print(f"ERROR: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
