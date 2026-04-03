# CoreInsight

**AI-powered performance profiler for Python, C++, and CUDA.**

CoreInsight finds hardware bottlenecks in your code, generates optimized replacements, and verifies the speedup mathematically inside an isolated Docker sandbox — all running locally on your machine.

---

## Install

```bash
pip install coreinsight-cli
```

**Requirements:** Python 3.9+ · Docker Desktop · [Ollama](https://ollama.com/download) (for local inference)

---

## Quick start

```bash
# Configure your AI provider (defaults to Ollama + llama3.2)
coreinsight configure

# Run the built-in demo
coreinsight demo

# Analyse your own file
coreinsight analyze path/to/your_file.py
```

---

## What it does

CoreInsight runs a full optimization pipeline on every function it extracts:

1. **Bottleneck analysis**
2. **Code generation**
3. **Sandbox verification**
4. **Hardware profiling**

Every result is stored in a local vector database. On repeat analyses, matching patterns are recalled instantly — no LLM call, no sandbox spin-up.

---

## Commands

| Command | Description |
|:--------|:------------|
| `coreinsight analyze <file>` | Analyse a `.py`, `.cpp`, or `.cu` file |
| `coreinsight demo [--lang cpp]` | Run on a built-in example |
| `coreinsight configure` | Set up AI provider and API keys |
| `coreinsight configure --pro-key <key>` | Activate Pro tier |
| `coreinsight memory` | Inspect stored optimizations |
| `coreinsight memory --clear` | Wipe the memory store |
| `coreinsight memory --export out.csv` | Export memory to CSV or Markdown |
| `coreinsight index [--dir <path>]` | Index a repo for cross-file RAG context |
| `coreinsight scan [--dir <path>]` | Rank hotspots by complexity without LLM |
| `coreinsight view` | Launch the interactive TUI |

All commands accept `--no-docker` to skip sandboxing when Docker is unavailable.

---

## Supported languages

| Language | Analysis | Benchmarking | Correctness |
|:---------|:--------:|:------------:|:-----------:|
| Python   | ✅ | ✅ | ✅ |
| C++      | ✅ | ✅ | ✅ |
| CUDA     | ✅ | ✅ | — |

---

## AI providers

| Provider | Tier | Notes |
|:---------|:----:|:------|
| Ollama | Free | `ollama pull llama3.2` |
| LM Studio / vLLM | Free | Any OpenAI-compatible server |
| OpenAI | Pro | GPT 5.3 recommended |
| Anthropic | Pro | Claude 4.6 Sonnet recommended |
| Google Gemini | Pro | Gemini 2.5 Pro recommended |

Local providers run entirely on-device. No code leaves your machine unless you configure a cloud provider.

---

## Pro — free during beta

Pro unlocks cloud providers and AI-free hardware profiling.  
Keys are being distributed manually during the beta.

**Request a key → [tally.so/r/xXZ9YE](https://tally.so/r/xXZ9YE)**

```bash
coreinsight configure --pro-key <your-key>
```

---

## Privacy

- **Local providers** — nothing leaves your machine
- **Cloud providers** — only the function code you analyse is sent to the provider API, under your own key
- The memory store lives at `~/.coreinsight/memory_db` on your filesystem

---

## Troubleshooting

**Docker not running**
```
open Docker Desktop, or: sudo systemctl start docker
```

**Ollama model not found**
```bash
ollama pull llama3.2
```

**ChromaDB / SQLite error**
```bash
pip install pysqlite3-binary
```

---

## Links

- PyPI: [pypi.org/project/coreinsight-cli](https://pypi.org/project/coreinsight-cli/)
- GitHub: [github.com/Prais3/coreinsight_cli](https://github.com/Prais3/coreinsight_cli)