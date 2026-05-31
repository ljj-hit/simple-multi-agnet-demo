# Minimal Multi-Agent GSM8K Benchmark

这是一个极简、可运行、可复现的 Multi-Agent GSM8K benchmark demo。它比较三个设置：

- `single`: Single Agent，只调用 Solver A，再由 Finalizer 输出最终答案
- `multi`: 普通 Multi-Agent，Solver A 和 Solver B 独立解题，再由 Finalizer 汇总
- `multi_verifier`: Multi-Agent + Verifier，两个 Solver 独立解题，Verifier 检查后再交给 Finalizer

项目不使用 AutoGen、LangGraph、CrewAI、数据库、Web UI、Docker 或虚拟环境脚本。核心逻辑都在 `run.py`。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置 API

复制 `.env.example` 为 `.env`，填入自己的 API key：

```env
API_KEY=your_api_key_here
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
```

代码使用 OpenAI-compatible chat completions API。

## 运行命令

```bash
python run.py --setting single
python run.py --setting multi
python run.py --setting multi_verifier
python run.py --setting all
```

也支持更直观的别名：

```bash
python run.py --setting single_agent
python run.py --setting multi_agent
python run.py --setting multi_agent_verifier
```

## 输出文件说明

运行后生成在 `outputs/`：

- `traces_all.json`: 每道题、每个 setting 的完整运行轨迹
- `metrics.csv`: 三个设置的准确率和 token 统计表
- `failures.json`: 失败样例分析

`traces_all.json` 包含：

- `question`
- `gold_answer`
- `setting`
- `setting_name`
- `solver_a_output`
- `solver_b_output`
- `verifier_output`
- `final_prediction`
- `correct`
- `token_usage`

`metrics.csv` 字段：

- `setting`
- `setting_name`
- `num_examples`
- `correct`
- `accuracy`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `avg_total_tokens`

## 实验设置说明

`data/gsm8k_20.jsonl` 固定保存 20 道 GSM8K-style 数学题，作为小样本评测集。评测逻辑会抽取 `####` 后的数字作为标准答案。

三种设置的含义：

- Single Agent: 最少调用，成本最低
- Multi-Agent: 两个 solver 独立作答，Finalizer 汇总
- Multi-Agent + Verifier: 额外加入 Verifier，检查两个 solver 的推理并给出建议答案

## 结果解释

看 `outputs/metrics.csv` 即可比较三种设置：

- `accuracy`: 正确题数 / 总题数
- `prompt_tokens`: 输入 token 总数
- `completion_tokens`: 输出 token 总数
- `total_tokens`: 总 token 数
- `avg_total_tokens`: 每道题平均总 token 数

如果 `Multi-Agent + Verifier` accuracy 更高，说明 verifier 在该样本上帮助纠错；如果 token 明显更高，则说明改进有额外成本。

## 如何复现

1. 安装依赖
2. 配置 `.env`
3. 确认 `data/gsm8k_20.jsonl` 不变
4. 运行：

```bash
python run.py --setting all
```

同一模型、同一 prompt、同一数据文件下即可复现实验流程和输出格式。
