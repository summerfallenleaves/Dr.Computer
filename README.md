# Dr.Computer

> A Python framework for building **Desktop AI Agents** — not just another
> Computer Use demo.

[![CI](https://github.com/summerfallenleaves/Dr.Computer/actions/workflows/ci.yml/badge.svg)](https://github.com/summerfallenleaves/Dr.Computer/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

Dr.Computer is a Framework, not an Application. Instead of locking you into
one model, one grounder, one platform, it gives you **six pluggable
protocols** (Provider / Perceiver / Grounder / Executor / Memory /
SafetyGuard) and a thin orchestration loop that wires them together. You mix
and match implementations to build the agent that fits your use case.

Think **LangGraph for LLM agents**, but for Desktop Computer Use.

---

## Why a Framework?

Most open-source Computer Use projects are **single-purpose applications**:
they pick one model, one grounding strategy, one platform, and ship it. They
work, but they're hard to extend. Dr.Computer takes a different bet: the
bottleneck isn't "we lack another agent", it's "we lack a common substrate
that lets agents be composed, swapped, and compared".

| Project | Type | Plug a different LLM? | Plug a different Grounder? | Persist trajectories? |
|---------|------|----------------------|---------------------------|----------------------|
| UI-TARS Desktop | App | No (UI-TARS only) | No | No |
| Open Computer Use | App | Limited | No | No |
| Browser Use | App | Yes | N/A (DOM) | No |
| **Dr.Computer** | **Framework** | **Yes (Protocol)** | **Yes (Protocol)** | **Yes (Protocol)** |

---

## Quickstart

### Prerequisites

- macOS (Windows/Linux support is on the roadmap)
- Python 3.11 or newer
- An API key for a vision LLM. Default Provider is **Qwen-VL** (Aliyun
  DashScope) — get a key at <https://bailian.console.aliyun.com/>

### 1. Install

```bash
git clone https://github.com/summerfallenleaves/Dr.Computer.git
cd Dr.Computer
uv sync --extra dev
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and put your DASHSCOPE_API_KEY in it
```

### 3. Grant macOS permissions

In **System Settings → Privacy & Security**, grant the terminal you'll run
Dr.Computer from:

- **Screen Recording** — needed for screenshots
- **Accessibility** — needed for mouse/keyboard control

Restart the terminal afterwards.

### 4. Run

```bash
# Either via the CLI
dr-computer run "Click on the Apple menu in the top-left corner"

# Or via the debug script (with pre-flight checks + log archiving)
./scripts/dev.sh "Click on the Apple menu in the top-left corner"

# Or via the example script directly
uv run python examples/01_open_notes.py "Click on the Apple menu"
```

You should see output like:

```
Goal: Click on the Apple menu in the top-left corner

  step  0 [executed] click @ (5, 5)         — Apple logo location
  step  1 [executed] click @ (5, 5)         — Apple logo location
  step  2 [executed] click @ (5, 5)         — Apple logo location

Outcome: loop_detected after 3 step(s).
```

The mouse will physically move to the Apple logo and click it.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AgentLoop                            │
│   Observe → Think → Ground → Safety → Act → Memory      │
└─────┬───────────────────────────────────────────────────┘
      │ 依赖注入
      ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Provider │ │Perceiver │ │ Grounder │ │ Executor │ │  Memory  │ │SafetyGuard│
│  (LLM)   │ │ (screenshot)│ │(locate) │ │ (mouse)  │ │ (history) │ │ (block)  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

Every component is a `typing.Protocol`. You implement the ones you need and
pass them to `AgentLoop`. Switching the LLM doesn't touch the Loop. Swapping
the Grounder doesn't touch the Provider. This is the Framework bet.

### Core abstractions

| Protocol | Responsibility | Default impl |
|----------|---------------|--------------|
| `Provider` | LLM call (returns `Intent`) | `QwenVLProvider` (fused — see ADR A3) |
| `Perceiver` | Capture environment state | `MacOSScreenshotPerceiver` |
| `Grounder` | Resolve description → bbox | (Phase 1.5: `OmniParserGrounder`) |
| `Executor` | Run the resolved `Action` | `PyAutoGUIExecutor` |
| `Memory` | Persist trajectory | `InMemoryMemory` (SQLite in Phase 2) |
| `SafetyGuard` | Block dangerous actions | `DefaultSafetyGuard` (blacklist) |
| `Verifier` | Check goal satisfaction | (Phase 2) |

---

## Programmatic usage

```python
from dr_computer import (
    AgentLoop,
    QwenVLProvider,
    MacOSScreenshotPerceiver,
    PyAutoGUIExecutor,
    InMemoryMemory,
    DefaultSafetyGuard,
)
import os

loop = AgentLoop(
    provider=QwenVLProvider(api_key=os.environ["DASHSCOPE_API_KEY"]),
    perceiver=MacOSScreenshotPerceiver(),
    executor=PyAutoGUIExecutor(),
    memory=InMemoryMemory(),
    safety_guard=DefaultSafetyGuard(),
    max_steps=20,
    loop_detection=3,
)

trajectory = loop.run("Open the Notes app")
print(trajectory.outcome)  # done | loop_detected | max_steps | failed | aborted
for step in trajectory.steps:
    print(f"  {step.num}: {step.intent.action_type if step.intent else '?'}")
```

See [`examples/01_open_notes.py`](examples/01_open_notes.py) for a complete
runnable example.

---

## CLI

```bash
dr-computer run "Open Safari"                       # basic
dr-computer run --verbose --max-steps 10 "Open..."  # debug
dr-computer run --provider qwen-vl --model qwen-vl-max "Open..."
dr-computer version
dr-computer --help
```

---

## Provider / Grounder matrix

| | Fused (returns bbox directly) | Non-fused (needs separate Grounder) |
|---|---|---|
| **Implemented** | `QwenVLProvider` (Qwen2.5-VL via DashScope) | — |
| **Phase 1.5** | — | `GLM4VProvider` + `OmniParserGrounder` |
| **Phase 2** | `OpenAIVisionProvider` (GPT-4o/5) | `OllamaProvider` + `PaddleOCRGrounder` |

---

## Safety

`DefaultSafetyGuard` ships with a blacklist that intercepts:

- **Dangerous hotkeys**: `Cmd+Q`, `Cmd+W`, `Cmd+Ctrl+Q`, `Ctrl+Alt+Del`, ...
- **Dangerous text** typed into terminals: `rm -rf`, `mkfs`, `dd if=`,
  `sudo`, `git push --force`, `git reset --hard`, fork bombs, ...
- **Optional human confirmation** for every spatial action (off by default;
  enable with `SafetyPolicy(ask_when_unmatched_spatial=True)`).

```python
from dr_computer import DefaultSafetyGuard, SafetyPolicy

guard = DefaultSafetyGuard(SafetyPolicy(
    block_hotkeys=[("cmd", "q"), ("cmd", "w")],
    block_text_patterns=[r"\brm\s+-rf?\b", r"\bsudo\b"],
    ask_when_unmatched_spatial=True,  # require confirmation per click
))
```

---

## Project status

**Phase 1 — Complete** ✅

- 6 Protocol abstractions, AgentLoop with async + sync wrappers
- Default Provider (QwenVLProvider, fused), Perceiver (macOS), Executor
  (PyAutoGUI), Memory (in-process), SafetyGuard (blacklist)
- 57 unit tests, all green
- End-to-end demo on macOS: screenshot → Qwen-VL → mouse click → goal done
- CLI (`dr-computer run`), debug script (`scripts/dev.sh`), CI

**Phase 1.5 — Pluggability validation** 🔲 (next)

- `OmniParserGrounder` (Microsoft OmniParser via local HTTP service)
- `GLM4VProvider` (non-fused, to prove the Provider/Grounder swap works)
- `examples/02_pluggability.py` running 4 combinations side-by-side

**Phase 2 — Production-grade** 🔲

- `Verifier` implementation (fixes the "model won't say done" issue)
- `SQLiteMemory` for persistence
- `OpenAIVisionProvider`, `OllamaProvider`
- macOS Accessibility API Perceiver

**Phase 3 — Differentiation** 🔲

- Skill Library (YAML)
- Workflow Recorder (pyobjc global event listener)
- Multi-model orchestration
- Electron GUI

See [`CLAUDE.md`](CLAUDE.md) for the full roadmap, ADRs, and detailed
status.

---

## Repository layout

```
Dr.Computer/
├── src/dr_computer/
│   ├── core/           # protocols, data models, AgentLoop (zero deps)
│   ├── providers/      # QwenVLProvider (+ base class)
│   ├── perception/     # MacOSScreenshotPerceiver
│   ├── grounding/      # (Phase 1.5: OmniParser)
│   ├── execution/      # PyAutoGUIExecutor
│   ├── memory/         # InMemoryMemory
│   ├── utils/          # bbox, image, events
│   └── cli/            # dr-computer CLI entry point
├── examples/           # 01_open_notes.py
├── scripts/            # dev.sh (debug runner), clean.sh
├── tests/unit/         # 57 tests
├── docs/               # (Phase 4)
├── .github/workflows/  # CI (macOS-latest, Python 3.11/3.12/3.13)
├── pyproject.toml      # uv + ruff + pytest config
├── CLAUDE.md           # project constitution (read this!)
└── README.md           # you are here
```

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run all checks locally (same as CI)
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit/

# One-shot debug run
./scripts/dev.sh "Click on the Apple menu"

# Clear caches and logs
./scripts/clean.sh
```

See [`CLAUDE.md` § 10](CLAUDE.md) for the full development standards
(type hints, docstrings, commit message format, security constraints).

---

## Why this name?

Dr.Computer — your computer's doctor. You describe what's wrong (the goal),
it examines (perceive), diagnoses (think), treats (act), and follows up
(verify). The "Dr." also nods to the framework being **prescriptive**: it
ships opinions on how an agent should be structured, not just a pile of
primitives.

---

## License

[MIT](LICENSE) © summerfallenleaves

## Acknowledgements

Dr.Computer stands on the shoulders of:

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — default LLM
- [OmniParser](https://github.com/microsoft/OmniParser) — future Grounder
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — mouse/keyboard
- [pyobjc](https://pyobjc.readthedocs.io/) — macOS native bindings
- [UI-TARS Desktop](https://github.com/bytedance/UI-TARS-desktop) and
  [Agent-S](https://github.com/simular-ai/Agent-S) — architectural
  inspiration
