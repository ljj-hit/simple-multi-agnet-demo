import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import run as bench


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_hard20_multi_verifier"
SETTING = "multi_verifier"


HARD_EXAMPLES = [
    {
        "question": "A store sells notebooks for $3 each and pens for $2 each. On Monday it sold 4 times as many pens as notebooks and made $154. How many pens did it sell?",
        "answer": "Let notebooks be n. Pens are 4n. Revenue is 3n + 2(4n) = 11n = 154, so n = 14. Pens sold are 4 * 14 = 56. #### 56",
    },
    {
        "question": "Maya had some beads. She used 2/5 of them for a necklace, then gave 18 beads to her sister. She had 42 beads left. How many beads did Maya have at first?",
        "answer": "After using 2/5, Maya had 3/5 of her beads left. Then she gave away 18 and had 42, so before giving she had 60. Thus 3/5 of the original is 60, so the original is 60 * 5 / 3 = 100. #### 100",
    },
    {
        "question": "A school bus route has 6 stops. At the first stop 12 students get on. At each later stop, 5 more students get on than at the previous stop. If 7 students get off before school, how many students arrive at school?",
        "answer": "Students getting on are 12, 17, 22, 27, 32, and 37. Their sum is 147. After 7 get off, 147 - 7 = 140 arrive. #### 140",
    },
    {
        "question": "A baker made 3 trays of cupcakes with 24 cupcakes per tray. She sold 5/8 of all cupcakes in the morning and 19 more in the afternoon. How many cupcakes were left?",
        "answer": "There were 3 * 24 = 72 cupcakes. She sold 5/8 * 72 = 45 in the morning. After selling 19 more, she sold 64 total. Left are 72 - 64 = 8. #### 8",
    },
    {
        "question": "Daniel has twice as many red marbles as blue marbles. He has 15 fewer green marbles than red marbles. If he has 105 marbles total, how many green marbles does he have?",
        "answer": "Let blue be b, red be 2b, and green be 2b - 15. Total is b + 2b + 2b - 15 = 105, so 5b = 120 and b = 24. Green is 48 - 15 = 33. #### 33",
    },
    {
        "question": "A tank is 1/4 full. After 180 liters are added, it is 5/8 full. What is the tank's capacity in liters?",
        "answer": "The added amount is 5/8 - 1/4 = 5/8 - 2/8 = 3/8 of the tank. If 3/8 is 180 liters, capacity is 180 * 8 / 3 = 480. #### 480",
    },
    {
        "question": "Sofia buys 3 packs of markers and 2 sketchbooks for $45. Each sketchbook costs $5 more than each marker pack. How much does one sketchbook cost?",
        "answer": "Let a marker pack cost m. A sketchbook costs m + 5. Then 3m + 2(m + 5) = 45, so 5m + 10 = 45 and 5m = 35. Thus m = 7, and a sketchbook costs 12. #### 12",
    },
    {
        "question": "A charity packs 336 meals into small boxes of 8 and large boxes of 12. It uses twice as many small boxes as large boxes. How many large boxes does it use?",
        "answer": "Let large boxes be L and small boxes be 2L. Meals are 12L + 8(2L) = 28L. Since 28L = 336, L = 12. #### 12",
    },
    {
        "question": "A runner completes 3 laps in 7.5 minutes. At the same pace, how many seconds will it take her to complete 11 laps?",
        "answer": "One lap takes 7.5 / 3 = 2.5 minutes. Eleven laps take 11 * 2.5 = 27.5 minutes. In seconds, that is 27.5 * 60 = 1650. #### 1650",
    },
    {
        "question": "A movie theater sold adult tickets for $12 and child tickets for $7. It sold 85 tickets for $790 total. How many child tickets were sold?",
        "answer": "Let child tickets be c, so adult tickets are 85 - c. Revenue is 7c + 12(85 - c) = 790. That gives 7c + 1020 - 12c = 790, so -5c = -230 and c = 46. #### 46",
    },
    {
        "question": "Rina spends 30% of her money on a book and then spends 25% of what remains on lunch. She has $42 left. How much money did she start with?",
        "answer": "After the book she has 70% left. After lunch she keeps 75% of that, so she has 0.75 * 0.70 = 0.525 of the original. If 0.525x = 42, then x = 80. #### 80",
    },
    {
        "question": "A jar contains nickels and dimes. There are 48 coins worth $3.65. How many nickels are in the jar?",
        "answer": "Let nickels be n and dimes be 48 - n. In cents, 5n + 10(48 - n) = 365. So 5n + 480 - 10n = 365, -5n = -115, and n = 23. #### 23",
    },
    {
        "question": "A rectangular garden is 3 meters longer than twice its width. Its perimeter is 78 meters. What is the garden's area?",
        "answer": "Let width be w and length be 2w + 3. Perimeter is 2(w + 2w + 3) = 78, so 6w + 6 = 78 and w = 12. Length is 27. Area is 12 * 27 = 324. #### 324",
    },
    {
        "question": "Eli reads 20 pages on the first day. Each day after that, he reads 6 more pages than the day before. How many pages does he read in 9 days?",
        "answer": "The pages form an arithmetic sequence from 20 to 68 over 9 days. The sum is 9 * (20 + 68) / 2 = 396. #### 396",
    },
    {
        "question": "A recipe uses 3 cups of flour for every 2 cups of sugar. A baker used 35 cups of flour and sugar combined in this ratio. How many cups of flour did the baker use?",
        "answer": "The ratio has 3 + 2 = 5 parts. Each part is 35 / 5 = 7 cups. Flour is 3 parts, so 3 * 7 = 21 cups. #### 21",
    },
    {
        "question": "A shop marks up a jacket by 40% and then gives a 25% discount on the marked price. The final price is $63. What was the original price?",
        "answer": "After markup and discount, the final price is original * 1.40 * 0.75 = original * 1.05. If 1.05x = 63, then x = 60. #### 60",
    },
    {
        "question": "Nina has 4 fewer than three times as many stamps as Omar. Together they have 88 stamps. How many stamps does Nina have?",
        "answer": "Let Omar have o stamps. Nina has 3o - 4. Total is o + 3o - 4 = 88, so 4o = 92 and o = 23. Nina has 3 * 23 - 4 = 65. #### 65",
    },
    {
        "question": "A factory makes 125 toys per hour for 6 hours. Then 8% of the toys fail inspection. The remaining toys are packed equally into 23 boxes. How many toys go in each box?",
        "answer": "The factory makes 125 * 6 = 750 toys. 8% fail, so 92% pass. Passing toys are 0.92 * 750 = 690. Packed into 23 boxes gives 690 / 23 = 30 toys per box. #### 30",
    },
    {
        "question": "A club has 60 members. 2/3 of them are students. 3/5 of the students attend a meeting, and 1/4 of the non-students attend. How many members attend the meeting?",
        "answer": "Students are 2/3 * 60 = 40, and non-students are 20. Student attendees are 3/5 * 40 = 24. Non-student attendees are 1/4 * 20 = 5. Total attendees are 29. #### 29",
    },
    {
        "question": "A car travels 45 miles per hour for 2 hours, then 60 miles per hour for 1.5 hours. If it uses 1 gallon of gas per 30 miles, how many gallons of gas does it use?",
        "answer": "The car travels 45 * 2 = 90 miles, then 60 * 1.5 = 90 miles, for 180 miles total. At 30 miles per gallon, it uses 180 / 30 = 6 gallons. #### 6",
    },
]


def load_hard_examples(limit: int | None = None) -> list[dict]:
    selected = HARD_EXAMPLES[:limit] if limit else HARD_EXAMPLES
    examples = []
    for item in selected:
        example = dict(item)
        example["gold_answer"] = bench.extract_gold(example["answer"])
        examples.append(example)
    return examples


def write_outputs(traces: list[dict]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "traces_hard20_multi_verifier.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {"n": 0, "correct": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for trace in traces:
        stats["n"] += 1
        stats["correct"] += int(trace["correct"])
        bench.add_usage(stats, trace["token_usage"])

    accuracy = stats["correct"] / stats["n"] if stats["n"] else 0
    avg_total_tokens = stats["total_tokens"] / stats["n"] if stats["n"] else 0
    with (OUTPUT_DIR / "metrics_hard20_multi_verifier.csv").open("w", newline="", encoding="utf-8") as f:
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
        writer.writerow(
            {
                "setting": SETTING,
                "setting_name": bench.SETTING_NAMES[SETTING],
                "num_examples": stats["n"],
                "correct": stats["correct"],
                "accuracy": round(accuracy, 4),
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["total_tokens"],
                "avg_total_tokens": round(avg_total_tokens, 2),
            }
        )

    (OUTPUT_DIR / "failures_hard20_multi_verifier.json").write_text(
        json.dumps(bench.build_failures(traces), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nHard-20 Multi-Agent + Verifier summary")
    print("-" * 88)
    print(f"examples: {stats['n']}")
    print(f"correct: {stats['correct']}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"prompt_tokens: {stats['prompt_tokens']}")
    print(f"completion_tokens: {stats['completion_tokens']}")
    print(f"total_tokens: {stats['total_tokens']}")
    print(f"avg_total_tokens: {avg_total_tokens:.2f}")
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard-20 supplement with multi_verifier only.")
    parser.add_argument("--limit", type=int, default=None, help="Optional small dry-run limit.")
    parser.add_argument("--check-config", action="store_true", help="Check config without model calls.")
    args = parser.parse_args()

    env_path = ROOT / ".env"
    load_dotenv(env_path, override=True)
    api_key = bench.env_value(*bench.API_KEY_NAMES)
    base_url = bench.env_value("BASE_URL", "OPENAI_BASE_URL", default="https://api.deepseek.com")
    model = bench.env_value("MODEL_NAME", "OPENAI_MODEL", default="deepseek-v4-flash")
    if not api_key:
        raise SystemExit("Missing API key. Fill .env before running.")

    examples = load_hard_examples(args.limit)
    prompts = {
        "solver_a": bench.read_text(bench.PROMPT_DIR / "solver_a.txt"),
        "solver_b": bench.read_text(bench.PROMPT_DIR / "solver_b.txt"),
        "verifier": bench.read_text(bench.PROMPT_DIR / "verifier.txt"),
        "finalizer": bench.read_text(bench.PROMPT_DIR / "finalizer.txt"),
    }

    if args.check_config:
        print("Configuration OK")
        print(f"api_key: {bench.masked(api_key)}")
        print(f"base_url: {base_url}")
        print(f"model: {model}")
        print(f"examples: {len(examples)}")
        print(f"setting: {bench.SETTING_NAMES[SETTING]}")
        print(f"outputs: {OUTPUT_DIR}")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    traces = []
    for idx, example in enumerate(examples, start=1):
        print(f"[Hard-20 | {bench.SETTING_NAMES[SETTING]}] {idx}/{len(examples)}")
        traces.append(bench.run_one(client, model, prompts, example, SETTING))

    write_outputs(traces)
    print(f"Done. Wrote supplement outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
