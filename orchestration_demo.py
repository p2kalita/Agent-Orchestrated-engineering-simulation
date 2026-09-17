import json
import time
from pathlib import Path
from typing import Any


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


def read_json(path: Path | str) -> dict[str, Any]:
    """Read and return a JSON object from a file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path | str, data: dict[str, Any]) -> None:
    """Write a dictionary to a formatted JSON file."""
    path = Path(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def create_plan(request: dict[str, Any]) -> dict[str, Any]:
    """Create the deterministic baseline execution plan."""

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
) -> dict[str, Any]:

    missing = sorted(REQUIRED_FIELDS - request.keys())

    if missing:
        return {
            "status": "failed",
            "checks": {
                "required_fields": "failed"
            },
            "message": f"Missing required fields: {missing}",
        }

    checks = {
        "required_fields": "passed",
        "grid_size": "passed",
        "time_steps": "passed",
        "diffusion_rate": "passed",
        "temperature_values": "passed",
    }

    # HARD CONTROLLER LIMITS
    MIN_GRID_SIZE = 3
    MAX_GRID_SIZE = 100
    MAX_TIME_STEPS = 10000
    MAX_DIFFUSION_RATE = 0.25

    grid_size = request["grid_size"]

    if not isinstance(grid_size, int) or not (
        MIN_GRID_SIZE <= grid_size <= MAX_GRID_SIZE
    ):
        checks["grid_size"] = "failed"

        return {
            "status": "failed",
            "checks": checks,
            "message": (
                f"grid_size must be an integer between "
                f"{MIN_GRID_SIZE} and {MAX_GRID_SIZE}"
            ),
        }

    time_steps = request["time_steps"]

    if not isinstance(time_steps, int) or not (
        1 <= time_steps <= MAX_TIME_STEPS
    ):
        checks["time_steps"] = "failed"

        return {
            "status": "failed",
            "checks": checks,
            "message": (
                f"time_steps must be an integer between 1 and "
                f"{MAX_TIME_STEPS}"
            ),
        }

    diffusion_rate = request["diffusion_rate"]

    if not isinstance(diffusion_rate, (int, float)) or not (
        0 < diffusion_rate <= MAX_DIFFUSION_RATE
    ):
        checks["diffusion_rate"] = "failed"

        return {
            "status": "failed",
            "checks": checks,
            "message": (
                "diffusion_rate must be greater than 0 and no more "
                f"than {MAX_DIFFUSION_RATE}"
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

    if request["hotspot_temperature"] < request["initial_temperature"]:
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


def build_initial_grid(request: dict[str, Any]) -> list[list[float]]:
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
) -> dict[str, Any]:
    """Run a simple two-dimensional heat-diffusion simulation."""

    size = request["grid_size"]
    time_steps = request["time_steps"]
    diffusion_rate = request["diffusion_rate"]

    grid = build_initial_grid(request)

    start_time = time.perf_counter()

    for _ in range(time_steps):
        new_grid = [row[:] for row in grid]

        for row in range(1, size - 1):
            for col in range(1, size - 1):

                neighbor_average = (
                    grid[row - 1][col]
                    + grid[row + 1][col]
                    + grid[row][col - 1]
                    + grid[row][col + 1]
                ) / 4.0

                new_grid[row][col] = (
                    grid[row][col]
                    + diffusion_rate
                    * (neighbor_average - grid[row][col])
                )

        grid = new_grid

    runtime = time.perf_counter() - start_time

    values = [
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
        "minimum_temperature": min(values),
        "maximum_temperature": max(values),
        "average_temperature": sum(values) / len(values),
        "center_temperature": grid[center][center],
        "runtime_seconds": runtime,
    }


def validate_results(results: dict[str, Any]) -> None:
    """Validate simulation output."""

    required_fields = [
        "case_id",
        "status",
        "grid_size",
        "iterations",
        "minimum_temperature",
        "maximum_temperature",
        "average_temperature",
        "center_temperature",
        "runtime_seconds",
    ]

    missing = [
        field
        for field in required_fields
        if field not in results
    ]

    if missing:
        raise ValueError(
            f"results.json is missing fields: {missing}"
        )

    if results["status"] != "completed":
        raise ValueError(
            "The simulation did not complete successfully."
        )

    numeric_fields = [
        "minimum_temperature",
        "maximum_temperature",
        "average_temperature",
        "center_temperature",
        "runtime_seconds",
    ]

    for field in numeric_fields:
        if not isinstance(results[field], (int, float)):
            raise TypeError(
                f"{field} must be numeric."
            )

    if results["minimum_temperature"] > results["maximum_temperature"]:
        raise ValueError(
            "Minimum temperature cannot exceed maximum temperature."
        )


def create_report(
    request: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Create a decision-facing report."""

    threshold = request["review_threshold"]
    maximum_temperature = results["maximum_temperature"]

    review_required = maximum_temperature > threshold

    return {
        "case_id": request["case_id"],
        "status": results["status"],
        "review_required": review_required,
        "review_threshold": threshold,
        "maximum_temperature": maximum_temperature,
        "average_temperature": results["average_temperature"],
        "center_temperature": results["center_temperature"],
        "runtime_seconds": results["runtime_seconds"],
        "message": (
            "Manual review required."
            if review_required
            else "Result is within the configured review threshold."
        ),
    }