# Copilot Onboarding Instructions - Learn2Slither

## Repository Overview

**Learn2Slither** is a hybrid Python-Rust project implementing a reinforcement learning (Q-learning) framework for training an AI agent to play Snake. The project demonstrates:

- **Snake game environment** written in Python with Pygame
- **Reinforcement learning agent** implemented in Rust (compiled as a Python extension module)
- **State encoding and Q-table management** using bit-packed states for efficient learning
- **Multi-mode gameplay**: AI-only training, human vs. AI (PVE), and visualization modes

### Project Size & Type
- **Size**: ~15K lines of code (mixed Python/Rust)
- **Languages**: Python 3.10+, Rust (2021 edition)
- **Key Frameworks**: Pygame, PyO3 (Rust-Python bindings), Maturin (build tool)
- **Primary Runtimes**: Python 3.10 (verified), Rust 1.86+

### Key Technical Details
- The Rust library (`snake_ai`) must be built and installed before the Python application can run
- The project uses state representation as 32-bit unsigned integers (packed boolean vectors)
- Q-learning uses substates: baseline, danger_death, danger_red_apple, green_apples (2), direction
- Build configuration uses Maturin for seamless Rust-Python compilation

---

## Build & Environment Setup

### Prerequisites
All steps assume execution in a bash shell with `cd /home/luis/proyects/Learn2Slither` as the working directory.

**System Requirements:**
- Python 3.10+ (verified: Python 3.10.12)
- Rust toolchain (verified: 1.86.0) - install via https://rustup.rs/ if not present
- Cargo (comes with Rust)
- GCC/build-essential (for compiling C extensions)

### Virtual Environment Setup (First Time)

```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Always run this before any Python work
pip install --upgrade pip

# Install core dependencies
pip install pygame matplotlib

# Install build tools
pip install maturin

# Install development tools (testing, linting, formatting)
pip install pytest black isort flake8

# Build Rust extension module
cd src/Rust
cargo build --release
cd ../..
```

**Important**: The Rust module must be built AFTER activating the venv and BEFORE running Python code that imports `snake_ai`.

### Activation (Every New Shell Session)

```bash
cd /home/luis/proyects/Learn2Slither
source .venv/bin/activate
```

After activation, verify with: `python -c "import snake_ai; print('OK')"`

### Rebuilding Rust Module

```bash
cd src/Rust
cargo build --release
cd ../..
```

This step is required if:
- Rust source files (`.rs`) have been modified
- `Cargo.toml` dependencies have changed
- When working with a fresh clone

**Build time**: ~10-15 seconds for incremental changes, ~60 seconds for clean builds.

---

## Code Quality & Validation

### Linting with Flake8

```bash
python -m flake8 src/
```

**Known Issues:**
- Current codebase has 20+ flake8 violations in `src/Python/learn_2_slither/Learn2Slither.py`
- Common issues: E251 (spaces around equals), E226 (missing whitespace around operators), F841 (unused variables)
- These should be fixed when touching those files

### Code Formatting with Black

```bash
# Check formatting without modifying
python -m black --check src/

# Auto-format files
python -m black src/
```

**Expected behavior:**
- Reformats 5 files: constants.py, menu.py, display.py, SnakeGame.py, Learn2Slither.py
- Uses line length of 88 characters (configured in pyproject.toml)
- Does not modify Rust files

### Import Sorting with isort

```bash
python -m isort --check-only src/

# Auto-fix imports
python -m isort src/
```

**Current status**: Passing (only 1 Rust file skipped)

### Testing

```bash
# Run pytest
pytest tests/ --maxfail=1 --disable-warnings --tb=short
```

**Important**: Test directories (`tests/Python/` and `tests/Rust/`) currently contain only `.gitkeep` files. No actual tests are defined. The pytest invocation will exit with code 5 (no tests collected) but this is expected and not a failure.

---

## Continuous Integration (GitHub Actions)

**Workflow file:** `.github/workflows/python-tests.yml`

**Triggers:** On push to `main` or pull request to `main`

**What it does:**
1. Runs on Ubuntu latest with Python 3.10
2. Installs dependencies: pip upgrade, pytest
3. Runs: `pytest tests/ --maxfail=1 --disable-warnings --tb=short || true`

**Important**: The `|| true` at the end ensures the workflow never fails, even if pytest finds no tests. This is intentional but means PR validation is currently minimal.

### To Replicate CI Locally

```bash
source .venv/bin/activate
pip install pytest
pytest tests/ --maxfail=1 --disable-warnings --tb=short
```

---

## Project Structure & Key Files

### Root Level
```
.flake8                    # Flake8 configuration (line-length: 88, ignores: E203, W503)
.pre-commit-config.yaml    # Pre-commit hooks (Black, isort, flake8)
pyproject.toml             # Black and isort configuration
.github/workflows/         # GitHub Actions CI/CD pipeline
src/                       # Main source code
tests/                     # Test directories (currently empty)
requirements.txt           # Empty (dependencies managed via pip commands above)
```

### Source Code (`src/`)
```
src/
├── main.py                           # Entry point: calls menu or Learn2Slither
├── Python/
│   ├── args/args.py                  # Argument parser for CLI
│   ├── constants/constants.py        # Game constants (colors, paths, cell size)
│   ├── display/display.py            # Pygame UI rendering
│   ├── menu/menu.py                  # Menu system
│   ├── learn_2_slither/Learn2Slither.py  # Main RL training loop
│   └── snake_game/SnakeGame.py       # Game engine (rules, state, physics)
├── Rust/
│   ├── Cargo.toml                    # Rust project manifest
│   ├── src/
│   │   ├── lib.rs                    # Rust library entry point
│   │   ├── action.rs                 # Action enum (Forward, Left, Right)
│   │   ├── heading.rs                # Direction/heading representation
│   │   ├── state.rs                  # State encoding, EnvironmentState, QTable
│   │   ├── qagent.rs                 # Q-learning agent logic
│   │   └── snake_ai.rs               # PyO3 bindings for Python
│   └── target/                       # Build output (auto-generated)
├── assets/
│   ├── images/background/            # Game background sprites
│   ├── images/food/                  # Food sprites (32px)
│   └── fonts/                        # Game fonts
└── models/
    ├── q_table.txt                   # Saved Q-table checkpoint
    └── pve/                          # Pre-trained agents for different difficulties
        ├── easy.txt, normal.txt, hard.txt
```

### Configuration Files

**`.flake8`**: Ignores E501 (long lines), E203/W503 (Black compatibility)

**`pyproject.toml`**:
```toml
[tool.black]
line-length = 79
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 79
```

**`Cargo.toml`** (Rust project):
- Package name: `snake_ai`
- Edition: 2021
- Crate type: `cdylib` (dynamic library for Python)
- Key dependencies: pyo3 (0.25.1), rand (0.9.2), once_cell

---

## Important Architecture Details

### State Representation

States are encoded as 32-bit unsigned integers via `State::from_bools()`:
- Each boolean array is converted to a bitfield
- Used for efficient Q-table lookups
- Core components in `src/Rust/src/state.rs`

### Q-Learning Substates

`EnvironmentState` contains multiple overlapping states:
1. **baseline** - Always active (for baseline reward learning)
2. **danger_death** - Active when wall/snake collision is one step ahead
3. **danger_red_apple** - Active when red apple is one step ahead
4. **green_apples[2]** - Tracks sightings of two green apples
5. **direction** - Current heading (Left, Right, Up, Down)

The learning mechanism updates multiple substates per transition to enable feature-level Q-learning.

### Main Entry Point Flow

1. `src/main.py` → parses args
2. If `--menu true` (default) → shows menu
3. Otherwise → instantiates `Learn2Slither(args)`
4. `Learn2Slither` manages game loops, calls into `snake_ai` (Rust) for Q-learning
5. Optionally visualizes with Pygame (`display/`)

### Rust-Python Integration

- **Rust module name**: `snake_ai`
- **Python interface**: Classes `TransitionInfo`, `EnvironmentState`, `State`; functions `act()`, `learn()`, `init()`
- **Build tool**: Maturin (handles compilation, packaging)
- **Binding library**: PyO3 (derives Python classes with `#[pyclass]`)

---

## Known Issues & Workarounds

### Python Linting Issues
- **Status**: 20+ flake8 warnings in `Learn2Slither.py` (E251, E226, F841, undefined names)
- **Workaround**: None needed for running; these are style violations. Fix by running `black` and editing by hand if using the code.

### Test Framework Status
- **Status**: `pytest` is installed but no tests are defined
- **Workaround**: Not critical; CI passes because workflow uses `|| true`. Add test files when needed.

### Rust Unused Import Warning
- **Status**: `use crate::heading::Heading;` in `src/action.rs:3` is unused
- **Workaround**: Can be removed; does not affect compilation or functionality

### Hardcoded File Paths
- **Status**: `src/Python/constants/constants.py` contains hardcoded path `/home/luis/proyects/...`
- **Workaround**: Works on developer machine but will fail on other systems. Consider using relative paths or environment variables.

---

## Verification Checklist

After making changes, verify the following:

### For Python Changes
1. **Black formatting**: `python -m black --check src/` (should pass or show files to format)
2. **Flake8 linting**: `python -m flake8 src/` (should not introduce new violations)
3. **Imports**: `python -m isort --check-only src/` (should pass)
4. **Module imports**: `python -c "from src.Python.snake_game import SnakeGame; from src.Python.learn_2_slither import Learn2Slither; print('OK')"`
5. **Rust module**: `python -c "import snake_ai; print(dir(snake_ai))"` (should list: Action, EnvironmentState, TransitionInfo, act, get_state, learn, etc.)

### For Rust Changes
1. **Compilation**: `cd src/Rust && cargo check` (should have 0 errors)
2. **Release build**: `cd src/Rust && cargo build --release` (may take 60s on clean build)
3. **Warnings**: `cargo build --release 2>&1 | grep -i warning` (document any new warnings)
4. **Python import after rebuild**: `python -c "import snake_ai; print('OK')"`

### End-to-End Test
```bash
cd src
timeout 5 python main.py --menu false --sessions 1 --visuals false || true
```

Expected output:
- "Session 1/1 completed"
- "Number of steps: [number]"
- No import errors or crashes

---

## Trust the Instructions

**Important**: Follow the instructions in this document exactly as written. The commands have been validated and tested. Only perform additional searches or explorations if:

1. You encounter an error NOT documented in the "Known Issues & Workarounds" section
2. The instructions reference a file or path that does not exist
3. You need to add a completely new feature requiring research into the codebase

For any other task, the instructions above provide the essential information to:
- Set up the environment correctly
- Build and compile the project
- Run tests and validation
- Understand the project structure
- Deploy changes successfully

If an issue persists after following the instructions, document the specific error and context for troubleshooting.
