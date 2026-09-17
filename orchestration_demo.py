import json
import time
from pathlib import Path
from typing import Any

REQUEST_FILE = Path("simulation_request.json")
PLAN_FILE = Path("plan.json")
PRE_GATE_FILE = Path("gate_pre.json")
RESULTS_FILE = Path("results.json")
REPORT_FILE = Path("report.json")

REQUIRED_FIELDS = {
    "case_id",
    "grid_size",
    "initial_temperature",
    "boundary_temperature",
    "hotspot_temperature",
    "time_steps",
    "diffusion_rate",
    "review_threshold",
}

def read_json(path: Path)-> dict[str, Any]:
    """Read and return a JSON object from a file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def write_json(path: Path, data: dict[str, Any])-> None:
    """Write a dictionary to a formatted JSON file."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def create_plan(request: dict[str, Any])-> dict[str, Any]:
    """Planner: convert the request into an ordered execution plan."""
    return {
        "case_id": request["case_id"],
        "simulation": "two_dimensional_heat_diffusion",
        "tool": "heat_diffusion_simulation",
        "steps": [
            "validate_request",
            "run_heat_simulation",
            "validate_results",
            "generate_report",
        ],
        "parameters": {
            "grid_size": request["grid_size"],
            "initial_temperature": request["initial_temperature"],
            "boundary_temperature": request["boundary_temperature"],
            "hotspot_temperature": request["hotspot_temperature"],
            "time_steps": request["time_steps"],
            "diffusion_rate": request["diffusion_rate"],
            "review_threshold": request["review_threshold"],
        },
        "constraints": {
            "min_grid_size": 3,
            "max_grid_size": 100,
            "max_time_steps": 10000,
            "max_diffusion_rate": 0.25,
        },
    }

def validate_request(
    request: dict[str, Any],
    plan: dict[str, Any],
)-> dict[str, Any]:
    """Reviewer and control: validate inputs and enforce hard limits."""
    missing = sorted(REQUIRED_FIELDS - request.keys())

    if missing:
        return {
            "status": "failed",
            "checks": {"required_fields": "failed"},
            "message": f"Missing required fields: {missing}",
        }
    
    checks = {
            "required_fields": "passed",
            "grid_size": "passed",
            "time_steps": "passed",
            "diffusion_rate": "passed",
            "temperature_values": "passed",
    }

    grid_size = request["grid_size"]
    time_steps = request["time_steps"]
    diffusion_rate = request["diffusion_rate"]

    constraints = plan["constraints"]

    if not isinstance(grid_size, int) or not (
        3<=grid_size<= constraints["max_grid_size"]
    ):
        checks["grid_size"] = "failed"
        return {
            "status": "failed",
            "checks": checks,
            "message": (
                "grid_size must be an integer between 3 and "
                f"{constraints['max_grid_size']}"
            ),
        }

    if not isinstance(time_steps, int) or not (
        1<=time_steps<= constraints["max_time_steps"]
    ):
        checks["time_steps"] = "failed"
        return {
            "status": "failed",
            "checks": checks,
            "message": (
                "time_steps must be an integer between 1 and "
                f"{constraints['max_time_steps']}"
            ),
        }
    
    if not isinstance(diffusion_rate, (int, float)) or not (
        0< diffusion_rate<= constraints["max_diffusion_rate"]
    ):
        checks["diffusion_rate"] = "failed"
        return {
            "status": "failed",
            "checks": checks,
            "message": (
                "diffusion_rate must be greater than 0 and no more than "
                f"{constraints['max_diffusion_rate']}"
            ),
        }

    
    temperature_fields = [
        "initial_temperature",
        "boundary_temperature",
        "hotspot_temperature",
        "review_threshold",
    ]

    if not all(
        isinstance(request[field], (int, float))
        for field in temperature_fields
    ):
        checks["temperature_values"] = "failed"
        return {
            "status": "failed",
            "checks": checks,
            "message": "All temperature values must be numeric.",
        }
    if request["hotspot_temperature"]< request["initial_temperature"]:
        checks["temperature_values"] = "failed"
        return {
            "status": "failed",
            "checks": checks,
            "message": (
                "hotspot_temperature must not be below "
                "initial_temperature."
            ),
        }
    
    return {
        "status": "passed",
        "checks": checks,
        "message": "The request is approved for execution.",
    }


def build_initial_grid(request: dict[str, Any])-> list[list[float]]:
    """Create the plate and place a hotspot at its center."""
    size = request["grid_size"]
    initial = float(request["initial_temperature"])
    boundary = float(request["boundary_temperature"])
    hotspot = float(request["hotspot_temperature"])

    grid = [
        [initial for _ in range(size)]
        for _ in range(size)
    ]

    for index in range(size):
        grid[0][index] = boundary
        grid[size - 1][index] = boundary
        grid[index][0] = boundary
        grid[index][size - 1] = boundary

    center = size // 2
    grid[center][center] = hotspot

    return grid


def run_heat_simulation(
 request: dict[str, Any],
)-> dict[str, Any]:
    """Executor: run a simple two-dimensional heat-diffusion model."""
    grid = build_initial_grid(request)

    size = request["grid_size"]
    time_steps = request["time_steps"]
    diffusion_rate = float(request["diffusion_rate"])
    boundary = float(request["boundary_temperature"])

    start_time = time.perf_counter()

    for _ in range(time_steps):
        updated = [row[:] for row in grid]
        for row in range(1, size - 1):
            for column in range(1, size - 1):
                neighbor_difference = (
                    grid[row - 1][column]
                    + grid[row + 1][column]
                    + grid[row][column - 1]
                    + grid[row][column + 1]
                    - 4 * grid[row][column]
                )

                updated[row][column] = (
                    grid[row][column]
                    + diffusion_rate * neighbor_difference
                )

        for index in range(size):
            updated[0][index] = boundary
            updated[size - 1][index] = boundary
            updated[index][0] = boundary
            updated[index][size - 1] = boundary

        grid = updated

    runtime_seconds = time.perf_counter() - start_time
    temperatures = [
        value
        for row in grid
        for value in row
    ]

    center = size // 2

    return {
        "case_id": request["case_id"],
        "status": "completed",
        "grid_size": size,
        "iterations": time_steps,
        "minimum_temperature": round(min(temperatures), 3),
        "maximum_temperature": round(max(temperatures), 3),
        "average_temperature": round(
            sum(temperatures) / len(temperatures),
            3,
        ),
        "center_temperature": round(grid[center][center], 3),
        "runtime_seconds": round(runtime_seconds, 6),
    }


def validate_results(results: dict[str, Any])-> None:
    """Post-run gate: reject incomplete or invalid solver output."""
    required_result_fields = {
        "case_id",
        "status",
        "grid_size",
        "iterations",
        "minimum_temperature",
        "maximum_temperature",
        "average_temperature",
        "center_temperature",
        "runtime_seconds",
    }
    missing = sorted(required_result_fields - results.keys())

    if missing:
        raise ValueError(
            f"Results are missing required fields: {missing}"
        )

    if results["status"] != "completed":
        raise ValueError("The simulation did not complete successfully.")

    if (
        results["minimum_temperature"] > results["maximum_temperature"]
    ):
        raise ValueError(
            "The minimum temperature exceeds the maximum temperature."
        )

def create_report(
    request: dict[str, Any],
    results: dict[str, Any],
)-> dict[str, Any]:
    """Create a short decision-facing summary."""
    peak_temperature = results["maximum_temperature"]
    threshold = request["review_threshold"]

    if peak_temperature> threshold:
        recommendation = (
            "Peak temperature exceeds the review threshold. "
            "Inspect the case before approval."
        )
        review_status = "review_required"
    else:
        recommendation = (
            "Peak temperature is within the configured review threshold."
        )
        review_status = "within_threshold"

    return {
        "case_id": request["case_id"],
        "outcome": "Heat-diffusion simulation completed successfully.",
        "review_status": review_status,
        "peak_temperature": peak_temperature,
        "review_threshold": threshold,
        "recommendation": recommendation,
    }

def main()-> None:
    """Run the planner-reviewer-control-executor workflow."""
    if not REQUEST_FILE.exists():
        raise FileNotFoundError(
            f"Create {REQUEST_FILE.name} before running this program."
    )

    request = read_json(REQUEST_FILE)

    plan = create_plan(request)
    write_json(PLAN_FILE, plan)

    gate_report = validate_request(request, plan)
    write_json(PRE_GATE_FILE, gate_report)


    if gate_report["status"] != "passed":
        print(
            "Execution stopped: "
            f"{gate_report['message']}"
        )
        return

    
    results = run_heat_simulation(request)
    validate_results(results)
    write_json(RESULTS_FILE, results)


    report = create_report(request, results)
    write_json(REPORT_FILE, report)


    print("Simulation completed.")
    print(f"Created: {PLAN_FILE}")
    print(f"Created: {PRE_GATE_FILE}")
    print(f"Created: {RESULTS_FILE}")
    print(f"Created: {REPORT_FILE}")



if __name__ == "__main__":
    main()
