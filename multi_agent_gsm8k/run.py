import argparse
import csv
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "gsm8k_20.jsonl"
PROMPT_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "outputs"
SETTINGS = ["single", "multi", "multi_verifier"]
SETTING_NAMES = {
    "single": "Single Agent",
    "multi": "Multi-Agent",
    "multi_verifier": "Multi-Agent + Verifier",
}
SETTING_ALIASES = {
    "single": "single",
    "single_agent": "single",
    "multi": "multi",
    "multi_agent": "multi",
    "multi_verifier": "multi_verifier",
    "multi_agent_verifier": "multi_verifier",
    "all": "all",
}
API_KEY_NAMES = ["API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def masked(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_examples() -> list[dict]:
    examples = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item["gold_answer"] = extract_gold(item["answer"])
            examples.append(item)
    return examples


def extract_gold(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return extract_answer(answer)


def extract_answer(text: str | None) -> str:
    if not text:
        return ""
    final_match = re.search(r"Final Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else text.strip()


def to_decimal(value: str) -> Decimal | None:
    cleaned = value.replace(",", "").strip()
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def is_correct(prediction: str, gold: str) -> bool:
    pred_num = to_decimal(extract_answer(prediction))
    gold_num = to_decimal(gold)
    if pred_num is None or gold_num is None:
        return extract_answer(prediction).strip() == gold.strip()
    return pred_num == gold_num


def usage_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def add_usage(total: dict, part: dict) -> None:
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        total[key] += part.get(key, 0)


def call_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip(), usage_dict(resp)


def parse_verifier(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {
        "verified_answer": extract_answer(text),
        "chosen_solver": "uncertain",
        "critique": "Verifier did not return valid JSON; extracted the last numeric answer.",
    }


def run_one(client: OpenAI, model: str, prompts: dict, example: dict, setting: str) -> dict:
    question = example["question"]
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    solver_a_output, usage = call_model(client, model, prompts["solver_a"], question)
    add_usage(token_usage, usage)

    solver_b_output = ""
    verifier_output = {}

    if setting in {"multi", "multi_verifier"}:
        solver_b_output, usage = call_model(client, model, prompts["solver_b"], question)
        add_usage(token_usage, usage)

    if setting == "single":
        final_input = f"Question:\n{question}\n\nSolver A output:\n{solver_a_output}"
    elif setting == "multi":
        final_input = (
            f"Question:\n{question}\n\n"
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}"
        )
    else:
        verifier_input = (
            f"Question:\n{question}\n\n"
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}"
        )
        verifier_text, usage = call_model(client, model, prompts["verifier"], verifier_input)
        add_usage(token_usage, usage)
        verifier_output = parse_verifier(verifier_text)
        final_input = (
            f"Question:\n{question}\n\n"
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}\n\n"
            f"Verifier JSON:\n{json.dumps(verifier_output, ensure_ascii=False)}"
        )

    final_output, usage = call_model(client, model, prompts["finalizer"], final_input)
    add_usage(token_usage, usage)
    final_prediction = extract_answer(final_output)
    correct = is_correct(final_prediction, example["gold_answer"])

    return {
        "question": question,
        "gold_answer": example["gold_answer"],
        "setting": setting,
        "setting_name": SETTING_NAMES[setting],
        "solver_a_output": solver_a_output,
        "solver_b_output": solver_b_output,
        "verifier_output": verifier_output,
        "final_prediction": final_prediction,
        "correct": correct,
        "token_usage": token_usage,
    }


def failure_type(prediction: str, gold: str) -> str:
    pred_num = to_decimal(prediction)
    gold_num = to_decimal(gold)
    if pred_num is None:
        return "non_numeric_prediction"
    if gold_num is not None and abs(pred_num - gold_num) <= Decimal("2"):
        return "off_by_small_amount"
    return "wrong_arithmetic_or_reasoning"


def build_failures(traces: list[dict]) -> list[dict]:
    failures = []
    for trace in traces:
        if trace["correct"]:
            continue
        failures.append(
            {
                "question": trace["question"],
                "prediction": trace["final_prediction"],
                "gold_answer": trace["gold_answer"],
                "failure_type": failure_type(trace["final_prediction"], trace["gold_answer"]),
                "analysis": (
                    f"{trace['setting']} predicted {trace['final_prediction']} but gold is "
                    f"{trace['gold_answer']}. Review solver arithmetic and whether the finalizer "
                    "copied the best-supported answer."
                ),
            }
        )

    if len(failures) < 5:
        for trace in traces[: 5 - len(failures)]:
            failures.append(
                {
                    "question": trace["question"],
                    "prediction": trace["final_prediction"],
                    "gold_answer": trace["gold_answer"],
                    "failure_type": "no_failure_observed",
                    "analysis": (
                        "This run produced fewer than 5 real failures. This supplemental entry "
                        "keeps the analysis file shape stable without marking the trace incorrect."
                    ),
                }
            )
    return failures[: max(5, len([t for t in traces if not t["correct"]]))]


def write_outputs(traces: list[dict]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "traces_all.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    grouped = {}
    for setting in SETTINGS:
        stats = grouped.setdefault(
            setting,
            {"n": 0, "correct": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    for trace in traces:
        stats = grouped[trace["setting"]]
        stats["n"] += 1
        stats["correct"] += int(trace["correct"])
        add_usage(stats, trace["token_usage"])

    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "setting_name",
                "num_examples",
                "correct",
                "accuracy",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "avg_total_tokens",
            ],
        )
        writer.writeheader()
        for setting in SETTINGS:
            stats = grouped[setting]
            accuracy = stats["correct"] / stats["n"] if stats["n"] else 0
            avg_total_tokens = stats["total_tokens"] / stats["n"] if stats["n"] else 0
            writer.writerow(
                {
                    "setting": setting,
                    "setting_name": SETTING_NAMES[setting],
                    "num_examples": stats["n"],
                    "correct": stats["correct"],
                    "accuracy": round(accuracy, 4),
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "total_tokens": stats["total_tokens"],
                    "avg_total_tokens": round(avg_total_tokens, 2),
                }
            )

    (OUTPUT_DIR / "failures.json").write_text(
        json.dumps(build_failures(traces), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_metrics_table(grouped)


def print_metrics_table(grouped: dict) -> None:
    print("\nAccuracy and token summary")
    print("-" * 103)
    print(
        f"{'Setting':<24} {'N':>3} {'Correct':>7} {'Accuracy':>9} "
        f"{'Prompt':>10} {'Completion':>12} {'Total':>10} {'Avg Total':>10}"
    )
    print("-" * 103)
    for setting in SETTINGS:
        stats = grouped[setting]
        accuracy = stats["correct"] / stats["n"] if stats["n"] else 0
        avg_total_tokens = stats["total_tokens"] / stats["n"] if stats["n"] else 0
        print(
            f"{SETTING_NAMES[setting]:<24} {stats['n']:>3} {stats['correct']:>7} "
            f"{accuracy:>9.4f} {stats['prompt_tokens']:>10} "
            f"{stats['completion_tokens']:>12} {stats['total_tokens']:>10} "
            f"{avg_total_tokens:>10.2f}"
        )
    print("-" * 103)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal multi-agent GSM8K benchmark demo.")
    parser.add_argument(
        "--setting",
        choices=list(SETTING_ALIASES),
        default="all",
        help=(
            "Experiment mode: single/single_agent, multi/multi_agent, "
            "multi_verifier/multi_agent_verifier, or all."
        ),
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate API configuration and data files without calling the model.",
    )
    args = parser.parse_args()
    selected_setting = SETTING_ALIASES[args.setting]

    env_path = ROOT / ".env"
    load_dotenv(env_path, override=True)
    api_key = env_value(*API_KEY_NAMES)
    base_url = env_value("BASE_URL", "OPENAI_BASE_URL", default="https://api.deepseek.com")
    model = env_value("MODEL_NAME", "OPENAI_MODEL", default="deepseek-v4-flash")
    if not api_key:
        if env_path.exists() and env_path.stat().st_size == 0:
            raise SystemExit(
                f"Missing API_KEY. {env_path} exists but is empty. "
                "Fill it with API_KEY, BASE_URL, and MODEL_NAME, then save the file."
            )
        raise SystemExit(
            "Missing API key. Add API_KEY to .env, or set DEEPSEEK_API_KEY/OPENAI_API_KEY "
            "in the environment."
        )

    prompts = {
        "solver_a": read_text(PROMPT_DIR / "solver_a.txt"),
        "solver_b": read_text(PROMPT_DIR / "solver_b.txt"),
        "verifier": read_text(PROMPT_DIR / "verifier.txt"),
        "finalizer": read_text(PROMPT_DIR / "finalizer.txt"),
    }
    settings = SETTINGS if selected_setting == "all" else [selected_setting]
    examples = load_examples()

    if args.check_config:
        print("Configuration OK")
        print(f"env_file: {env_path}")
        print(f"api_key: {masked(api_key)}")
        print(f"base_url: {base_url}")
        print(f"model: {model}")
        print(f"examples: {len(examples)}")
        print(f"settings: {', '.join(SETTING_NAMES[s] for s in settings)}")
        print(f"outputs: {OUTPUT_DIR}")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    traces = []
    for setting in settings:
        for idx, example in enumerate(examples, start=1):
            print(f"[{SETTING_NAMES[setting]}] {idx}/{len(examples)}")
            traces.append(run_one(client, model, prompts, example, setting))

    write_outputs(traces)
    print(f"Done. Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
