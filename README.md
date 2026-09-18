# Agent-Orchestrated Engineering Simulation (AG2 / AutoGen)

An end-to-end framework demonstrating **agent-orchestrated numerical engineering workflows** using AutoGen (AG2) and Google Gemini LLMs, paired with deterministic supervisory control gates and quality assurance contracts.

This project simulates a **two-dimensional heat-diffusion process** across a discrete plate with fixed boundary conditions, orchestrating the entire lifecycle—from parameter ingestion and execution planning to safety validation, solver dispatch, output verification, and post-run engineering reporting.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Workflow](#architecture--workflow)
  - [The 4-Agent Orchestration Pattern](#the-4-agent-orchestration-pattern)
  - [Deterministic Control Gate Principle](#deterministic-control-gate-principle)
  - [Numerical Physics & Stability Constraint](#numerical-physics--stability-constraint)
- [Repository Structure](#repository-structure)
- [Data Artifacts & Pipeline](#data-artifacts--pipeline)
- [Prerequisites](#prerequisites)
- [Sequential Steps to Run the App](#sequential-steps-to-run-the-app)
  - [Step 1: Clone or Navigate to the Workspace](#step-1-clone-or-navigate-to-the-workspace)
  - [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
  - [Step 3: Install Required Dependencies](#step-3-install-required-dependencies)
  - [Step 4: Configure Environment Variables](#step-4-configure-environment-variables)
  - [Step 5: Run the Quality Control Smoke Test](#step-5-run-the-quality-control-smoke-test)
  - [Step 6: Run the Deterministic Orchestration Demo (No API Key Required)](#step-6-run-the-deterministic-orchestration-demo-no-api-key-required)
  - [Step 7: Run the Full AG2 / AutoGen Multi-Agent Orchestration (LLM-Powered)](#step-7-run-the-full-ag2--autogen-multi-agent-orchestration-llm-powered)
  - [Step 8: Review Generated Artifacts](#step-8-review-generated-artifacts)
- [Simulation Parameter Reference](#simulation-parameter-reference)
- [Troubleshooting & FAQs](#troubleshooting--faqs)
- [References](#references)

---

## Overview

Engineering simulations require strict precision, bounded computation, and physical validity. While Generative AI and Large Language Models (LLMs) excel at flexible reasoning and planning, unconstrained LLMs can hallucinate parameters or bypass domain safety rules.

This repository implements the architecture described in the accompanying textbook, *Agent-Orchestrated Engineering Simulation*:
1. **Separation of Concerns**: Specialized agents handle discrete phases—planning, domain review, execution, and deterministic control.
2. **Deterministic Control Gate**: Prompts do **not** replace hard limits. A supervisory software gate deterministically enforces physics and numerical limits before execution.
3. **Reproducible Pipeline**: All inputs, plans, decisions, solver metrics, and reports are persisted as structured JSON artifacts.

---

## Key Features

- **Dual-Mode Execution**:
  - **Deterministic Baseline (`orchestration_demo.py`)**: Runs completely offline without API calls, ideal for quick testing, CI/CD, and schema validation.
  - **Multi-Agent Orchestration (`ag2_orchestration.py`)**: Employs AG2/AutoGen conversational agents powered by Google Gemini to plan, review, and execute tool calls.
- **Finite Difference Numerical Heat Solver**: Explicit 2D heat-diffusion solver on a discrete grid with Dirichlet boundary conditions and von Neumann stability checks.
- **Strict Quality Contracts & Smoke Tests (`qc_smoke.py`)**: Pre-flight and post-run contract validators verifying schema compliance, bounded inputs, and output sanity.
- **Auditable Artifact Trail**: Every run produces or updates `plan.json`, `gate_pre.json`, `results.json`, and `report.json`.

---

## Architecture & Workflow

### The 4-Agent Orchestration Pattern

```
                       +-------------------------------+
                       |    simulation_request.json    |
                       +---------------+---------------+
                                       |
                                       v
                             [ Controller Agent ]
                               (Non-LLM Gate)
                                       |
          +----------------------------+----------------------------+
          |                                                         |
          v                                                         v
   [ Planner Agent ]                                        [ Reviewer Agent ]
 Converts request to                                       Inspects plan and checks
 structured plan.json                                      domain safety rules
          |                                                         |
          +----------------------------+----------------------------+
                                       |
                                       v
                             [ Controller Gate ]
                      Enforces hard physical limits
                           Creates gate_pre.json
                         [ Status: Passed / Failed ]
                                       |
                          (If status == "passed")
                                       v
                             [ Executor Agent ]
                        Calls heat_diffusion_simulation()
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
             results.json                             report.json
       (Numerical Solver Metrics)             (Threshold & Review Summary)
```

1. **Controller (`controller`)**:
   - Human-in-the-loop / non-LLM supervisory node (`human_input_mode="NEVER"`, `llm_config=False`).
   - Dispatches chats to the Planner, Reviewer, and Executor.
   - Enforces deterministic code guards that cannot be overridden by model outputs.
2. **Planner (`planner`)**:
   - Generates a structured multi-step execution plan specifying steps (`validate_request`, `run_heat_simulation`, `validate_results`, `generate_report`) and parameter constraints.
3. **Reviewer (`reviewer`)**:
   - Examines the input request and plan for numerical viability, returning `{"decision": "approve" | "revise", "issues": [...]}`.
4. **Executor (`executor`)**:
   - Registered with tool execution privileges for `heat_diffusion_simulation()`.
   - Executes the solver, validates result invariants, and formats the final engineering report.

### Deterministic Control Gate Principle

> **"Prompts do not replace hard limits."**

Even if an LLM Reviewer agent approves an invalid plan, the supervisory code gate (`validate_request`) intercepts the payload and cancels solver execution if any hard guardrails are breached:
- $3 \le \text{grid\_size} \le 100$
- $1 \le \text{time\_steps} \le 10,000$
- $0 < \text{diffusion\_rate} \le 0.25$ (stability limit)
- All temperatures numeric with $\text{hotspot\_temperature} \ge \text{initial\_temperature}$

### Numerical Physics & Stability Constraint

The numerical core discretizes the 2D heat equation $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$ using the explicit finite difference method:

$$T_{i,j}^{n+1} = T_{i,j}^n + r \left( T_{i+1,j}^n + T_{i-1,j}^n + T_{i,j+1}^n + T_{i,j-1}^n - 4T_{i,j}^n \right)$$

where $r = \alpha \frac{\Delta t}{\Delta x^2}$ is the dimensionless diffusion rate.
For numerical stability in two dimensions under explicit time stepping, the von Neumann criterion mandates:

$$r \le 0.25$$

If $r > 0.25$, numerical oscillations grow exponentially and diverge. The control gate strictly rejects any simulation request with $r > 0.25$.

---

## Repository Structure

```
_agent-orchestrated-engineering-simulation/
├── .env                                 # Environment variables (GEMINI_API_KEY, GROQ_API_KEY)
├── .gitignore                           # Git ignore rules
├── README.md                            # Comprehensive project documentation (this file)
├── requirements.txt                     # Python package dependencies (ag2[gemini], python-dotenv)
│
├── ag2_orchestration.py                 # Multi-agent AutoGen/AG2 orchestration script
├── orchestration_demo.py                # Deterministic baseline orchestration (no API key needed)
├── qc_smoke.py                          # Pre-flight and post-run contract smoke tests
│
├── simulation_request.json              # Input specification for the simulation case
├── plan.json                            # Generated execution plan
├── gate_pre.json                        # Pre-execution validation gate record
├── results.json                         # Numerical simulation solver metrics
├── report.json                          # Post-simulation engineering evaluation report
│
├── book/
│   └── agent-orchestrated-engineering-simulation-fast.pdf  # Reference textbook
│
├── code/
│   ├── chapter3_prepare_ground.zip     # Chapter 3 starter package
│   ├── orchestration_demo.py           # Reference demo implementation
│   └── simulation_request.json         # Reference request configuration
│
└── venv/                               # Local Python virtual environment
```

---

## Data Artifacts & Pipeline

The pipeline processes and emits the following structured JSON files:

| File | Producer | Description |
|---|---|---|
| [`simulation_request.json`](file:///d:/_agent-orchestrated-engineering-simulation/simulation_request.json) | User / Engineer | Ingestion specification defining case parameters and review thresholds. |
| [`plan.json`](file:///d:/_agent-orchestrated-engineering-simulation/plan.json) | Planner Agent | Execution sequence, tool specification, and parameter constraints. |
| [`gate_pre.json`](file:///d:/_agent-orchestrated-engineering-simulation/gate_pre.json) | Controller Gate | Decision gate outcome (`passed` / `failed`) with specific check statuses. |
| [`results.json`](file:///d:/_agent-orchestrated-engineering-simulation/results.json) | Simulation Solver | Minimum, maximum, average, and center temperatures, plus iteration count and runtime. |
| [`report.json`](file:///d:/_agent-orchestrated-engineering-simulation/report.json) | Executor / Reporter | Engineering summary comparing peak temperature against `review_threshold`. |

---

## Prerequisites

Before running the project, ensure you have:

1. **Python**: Python 3.10, 3.11, 3.12, or 3.13 installed.
2. **Operating System**: Windows, Linux, or macOS.
3. **Google Gemini API Key**:
   - Required for `ag2_orchestration.py`.
   - Obtain a free or paid API key at [Google AI Studio](https://aistudio.google.com/).
   - *Note: `orchestration_demo.py` and `qc_smoke.py` do NOT require an API key.*

---

## Sequential Steps to Run the App

Follow these steps sequentially to set up the environment and run both the deterministic baseline and the LLM multi-agent orchestration.

### Step 1: Clone or Navigate to the Workspace

Open PowerShell (Windows) or Terminal (macOS/Linux) and navigate to the project directory:

```powershell
cd D:\_agent-orchestrated-engineering-simulation
```

### Step 2: Create and Activate a Virtual Environment

#### On Windows (PowerShell):
If using the included virtual environment:
```powershell
# Using the pre-configured local environment:
.\venv\python.exe -V
```

Or to create a fresh virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Installed libraries include `ag2[gemini]` (AutoGen 2.0 with Gemini backend) and `python-dotenv`.*

### Step 4: Configure Environment Variables

Create or edit the `.env` file in the root directory:

```env
GEMINI_API_KEY="your_actual_google_gemini_api_key_here"
```

> [!IMPORTANT]
> Keep your `.env` file secure and never commit real API keys to public version control repositories.

### Step 5: Run the Quality Control Smoke Test

Before executing simulations, verify that all contracts, schemas, and invariant validators pass:

```bash
python qc_smoke.py
```

**Expected Output:**
```text
smoke: validating simulation request
smoke: validating execution plan
smoke: validating simulation results
smoke: PASS
```

### Step 6: Run the Deterministic Orchestration Demo (No API Key Required)

Run the baseline orchestration pipeline. This uses pure Python logic without calling an LLM, guaranteeing reproducibility and instant execution:

```bash
python orchestration_demo.py
```

**Expected Output:**
```text
Simulation completed.
Created: plan.json
Created: gate_pre.json
Created: results.json
Created: report.json
```

Inspect the generated output files:
- `plan.json`: Verify the planned steps.
- `gate_pre.json`: Verify `"status": "passed"`.
- `results.json`: Check temperatures and execution time.
- `report.json`: Check `"review_status": "within_threshold"`.

### Step 7: Run the Full AG2 / AutoGen Multi-Agent Orchestration (LLM-Powered)

Run the multi-agent system where Gemini-backed agents converse to plan, review, and execute:

```bash
python ag2_orchestration.py
```

*(Or use `.\venv\python.exe ag2_orchestration.py` if running from the root without activating the venv)*.

**What Happens During Execution:**
1. **Controller** initiates chat with **Planner** passing `simulation_request.json`.
2. **Planner** returns the execution plan JSON.
3. **Controller** prompts **Reviewer** with the request and proposed plan.
4. **Reviewer** checks numerical stability ($r \le 0.25$) and parameter ranges.
5. **Controller** evaluates the deterministic gate (`validate_request`) and writes `gate_pre.json`.
6. If passed, **Controller** directs **Executor** to run `heat_diffusion_simulation()`.
7. The simulation executes, validates output invariants, and saves `results.json` and `report.json`.

**Expected Console Output:**
```text
AG2 orchestration completed.
Created: plan.json
Created: gate_pre.json
Created: results.json
Created: report.json
```

### Step 8: Review Generated Artifacts

View the generated outputs:

```powershell
# In PowerShell:
Get-Content report.json
Get-Content results.json
```

Example `report.json`:
```json
{
  "case_id": "heat_case_001",
  "outcome": "Heat-diffusion simulation completed successfully.",
  "review_status": "within_threshold",
  "peak_temperature": 20.638,
  "review_threshold": 80.0,
  "recommendation": "Peak temperature is within the configured review threshold."
}
```

---

## Simulation Parameter Reference

Simulation cases are defined in `simulation_request.json`. Below is the schema and validation constraint reference:

| Field | Type | Description | Valid Range / Constraint |
|---|---|---|---|
| `case_id` | `string` | Unique identifier for the simulation run | Non-empty string (e.g., `"heat_case_001"`) |
| `grid_size` | `integer` | Square grid width and height ($N \times N$) | $3 \le N \le 100$ |
| `initial_temperature` | `float` | Uniform starting temperature of the plate | Numeric |
| `boundary_temperature`| `float` | Constant boundary temperature (Dirichlet) | Numeric |
| `hotspot_temperature` | `float` | Initial center hotspot temperature | Numeric, $\ge \text{initial\_temperature}$ |
| `time_steps` | `integer` | Number of finite-difference iterations | $1 \le \text{steps} \le 10,000$ |
| `diffusion_rate` | `float` | Dimensionless diffusion parameter ($r$) | $0.0 < r \le 0.25$ (Stability limit) |
| `review_threshold` | `float` | Peak temperature threshold for human review | Numeric |

### Testing Custom Simulation Scenarios

You can edit `simulation_request.json` to simulate different scenarios:

#### Scenario A: High Heat Dissipation
```json
{
  "case_id": "heat_case_fast_dissipation",
  "grid_size": 30,
  "initial_temperature": 20.0,
  "boundary_temperature": 15.0,
  "hotspot_temperature": 120.0,
  "time_steps": 500,
  "diffusion_rate": 0.2,
  "review_threshold": 50.0
}
```

#### Scenario B: Testing Safety Gate Interception (Invalid Stability Rate)
Set `"diffusion_rate": 0.35` (violates $r \le 0.25$):
```json
{
  "case_id": "unstable_case",
  "grid_size": 20,
  "initial_temperature": 20.0,
  "boundary_temperature": 20.0,
  "hotspot_temperature": 100.0,
  "time_steps": 100,
  "diffusion_rate": 0.35,
  "review_threshold": 80.0
}
```
When executed, the pre-execution gate triggers an immediate halt:
```text
Execution stopped: diffusion_rate must be greater than 0 and no more than 0.25
```
And `gate_pre.json` records `"status": "failed"`.

---

## Troubleshooting & FAQs

### 1. `ModuleNotFoundError: No module named 'autogen'`
- **Cause**: Script was run using global Python instead of the virtual environment where AG2 is installed.
- **Solution**: Activate the environment (`.\venv\Scripts\Activate.ps1` or `source venv/bin/activate`) or run directly via `.\venv\python.exe ag2_orchestration.py`.

### 2. `RuntimeError: GEMINI_API_KEY is not set. Add it to your .env file.`
- **Cause**: The `.env` file is missing or `GEMINI_API_KEY` is not defined.
- **Solution**: Add `GEMINI_API_KEY="your-api-key"` to `.env`.

### 3. `google.genai.errors.ServerError: 503 UNAVAILABLE`
- **Cause**: Google Gemini servers are experiencing high demand or temporary throttling for the specified model (`gemini-3.5-flash` or `gemini-2.0-flash`).
- **Solution**:
  - Wait a few moments and retry.
  - Or switch the model parameter in `ag2_orchestration.py` (line 31) to an active Gemini model, such as `"gemini-2.5-flash"`, `"gemini-2.0-flash"`, or `"gemini-1.5-flash"`.

### 4. `Execution stopped: hotspot_temperature must not be below initial_temperature.`
- **Cause**: Physical inconsistency where the center hotspot has a lower temperature than the ambient plate.
- **Solution**: Ensure `hotspot_temperature >= initial_temperature` in `simulation_request.json`.

---

## References

- **Accompanying Textbook**: `book/agent-orchestrated-engineering-simulation-fast.pdf`
- **AutoGen (AG2) Documentation**: [https://ag2.ai/](https://ag2.ai/)
- **Google GenAI Python SDK**: [https://github.com/googleapis/python-genai](https://github.com/googleapis/python-genai)
- **Numerical Stability for Parabolic PDEs**: Finite Difference Heat Diffusion and the Courant-Friedrichs-Lewy / von Neumann condition ($r \le 0.25$).