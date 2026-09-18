REQUIRED_FIELDS = [
    "case_id",
    "grid_size",
    "initial_temperature",
    "boundary_temperature",
    "hotspot_temperature",
    "time_steps",
    "diffusion_rate",
    "review_threshold",
]

REQUIRED_PLAN_FIELDS = [
    "case_id",
    "simulation",
    "tool",
    "steps",
    "parameters",
    "constraints",
]

REQUIRED_RESULT_FIELDS = [
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

LIMITS = {
    "min_grid_size": 3,
    "max_grid_size": 100,
    "max_time_steps": 10_000,
    "min_diffusion_rate": 0.0,
    "max_diffusion_rate": 0.25,
}

def validate_request(request: dict)-> None:
    """Confirm that simulation_request.json contains all required fields."""
    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in request
    ]
    if missing:
        raise ValueError(
            f"simulation_request.json is missing fields: {missing}"
        )


def validate_plan(
    request: dict,
    plan: dict,
    limits: dict,
)-> None:
    """Confirm that plan.json matches the approved request and limits."""
    missing = [
    field
    for field in REQUIRED_PLAN_FIELDS
    if field not in plan
    ]
    if missing:
        raise ValueError(
            f"plan.json is missing fields: {missing}"
        )
    
    if plan["case_id"] != request["case_id"]:
        raise ValueError(
            "The plan case_id does not match "
            "simulation_request.json."
        )
    
    if plan["simulation"] != "two_dimensional_heat_diffusion":
        raise ValueError(
            "The plan contains an unsupported simulation."
        )
    
    if plan["tool"] != "heat_diffusion_simulation":
        raise ValueError(
            "The plan must use the heat_diffusion_simulation tool."
        )
    
    parameters = plan["parameters"]

    if not isinstance(parameters, dict):
        raise TypeError(
            "plan.json parameters must be an object."
        )

    required_parameters = [
        field
        for field in REQUIRED_FIELDS
        if field != "case_id"
    ]

    missing_parameters = [
        field
        for field in required_parameters
        if field not in parameters
    ]

    if missing_parameters:
        raise ValueError(
            "plan.json parameters are missing fields: "
            f"{missing_parameters}"
        )
    
    for field in required_parameters:
        if parameters[field] != request[field]:
            raise ValueError(
                f"The plan parameter '{field}' does not match "
                "simulation_request.json."
            )
        
    constraints = plan["constraints"]

    if not isinstance(constraints, dict):
        raise TypeError(
            "plan.json constraints must be an object."
        )

    
    required_constraints = [
        "min_grid_size",
        "max_grid_size",
        "max_time_steps",
        "max_diffusion_rate",
    ]

    missing_constraints = [
        field
        for field in required_constraints
        if field not in constraints
    ]

    if missing_constraints:
        raise ValueError(
            "plan.json constraints are missing fields: "
            f"{missing_constraints}"
        )
    
    grid_size = parameters["grid_size"]

    if not isinstance(grid_size, int):
        raise TypeError(
            "grid_size must be an integer."
        )
    
    if not (
        limits["min_grid_size"] <= grid_size <= limits["max_grid_size"]
    ):
        raise ValueError(
            "grid_size is outside the supported range: " 
            f"{grid_size}"
        )
    
    time_steps = parameters["time_steps"]

    if not isinstance(time_steps, int) or time_steps<= 0:
        raise ValueError(
            "time_steps must be a positive integer."
        )
    
    if time_steps> limits["max_time_steps"]:
        raise ValueError(
            "time_steps exceeds the configured cap: "
            f"{time_steps}> {limits['max_time_steps']}"
        )
    
    diffusion_rate = parameters["diffusion_rate"]

    if not isinstance(diffusion_rate, (int, float)):
        raise TypeError(
            "diffusion_rate must be numeric."
        )
    
    if not (
        limits["min_diffusion_rate"]
    < diffusion_rate
        <= limits["max_diffusion_rate"]
    ):
        raise ValueError(
            "diffusion_rate is outside the supported range: "
            f"{diffusion_rate}"
        )


def validate_results(results: dict)-> None:
    """Confirm that results.json matches the Chapter 5 result contract."""
    missing = [
        field
        for field in REQUIRED_RESULT_FIELDS
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
    
    if not isinstance(results["grid_size"], int):
        raise TypeError(
            "grid_size must be an integer."
        )
    
    if not isinstance(results["iterations"], int):
        raise TypeError(
        "iterations must be an integer."
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
    if (
        results["minimum_temperature"] > results["maximum_temperature"]
    ):
        
        raise ValueError(
            "The minimum temperature exceeds "
            "the maximum temperature."
        )

    if not (
    results["minimum_temperature"] <= results["average_temperature"] <= results["maximum_temperature"]
    ):
        raise ValueError(
            "The average temperature falls outside "
            "the reported temperature range."
        )
    
    if not ( results["minimum_temperature"] <= results["center_temperature"] <= results["maximum_temperature"]
    ):
        raise ValueError(
            "The center temperature falls outside "
            "the reported temperature range."
        )
    if results["runtime_seconds"]< 0:
        raise ValueError(
            "runtime_seconds must not be negative."
        )
    

"""Run basic quality checks for the heat-diffusion workflow."""
import math


def validate_results(results: dict)-> None:
    required_fields = [
        "case_id",
        "status",
        "minimum_temperature",
        "maximum_temperature",
        "average_temperature",
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
    
    numeric_fields = [
        "minimum_temperature",
        "maximum_temperature",
        "average_temperature",
    ]
    
    for field in numeric_fields:
        value = results[field]

        if not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        
    if results["minimum_temperature"]> results["maximum_temperature"]:
        raise ValueError(
            "minimum_temperature cannot exceed maximum_temperature"
        )
    
    if results["status"] != "completed":
        raise ValueError(
            f"simulation did not complete: {results['status']}"
        )

    
def run_smoke_test()-> None:
    request = {
        "case_id": "heat_plate_smoke",
        "grid_size": 20,
        "initial_temperature": 25.0,
        "boundary_temperature": 25.0,
        "hotspot_temperature": 100.0,
        "time_steps": 100,
        "diffusion_rate": 0.1,
        "review_threshold": 40.0,
    }

    # plan = {
    #     "case_id": request["case_id"],
    #     "tool": "heat_diffusion_simulation",
    #     "parameters": {
    #         "grid_size": request["grid_size"],
    #         "initial_temperature": request["initial_temperature"],
    #         "boundary_temperature": request["boundary_temperature"],
    #         "hotspot_temperature": request["hotspot_temperature"],
    #         "time_steps": request["time_steps"],
    #         "diffusion_rate": request["diffusion_rate"],
    #         },
    # }
    plan = {
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
            "min_grid_size": LIMITS["min_grid_size"],
            "max_grid_size": LIMITS["max_grid_size"],
            "max_time_steps": LIMITS["max_time_steps"],
            "max_diffusion_rate": LIMITS["max_diffusion_rate"],
        },
    }

    results = {
        "case_id": request["case_id"],
        "status": "completed",
        "minimum_temperature": 25.0,
        "maximum_temperature": 74.2,
        "average_temperature": 31.8,
    }

    print("smoke: validating simulation request")
    validate_request(request)

    print("smoke: validating execution plan")
    validate_plan(request, plan, LIMITS)

    print("smoke: validating simulation results")
    validate_results(results)

    print("smoke: PASS")

    
if __name__ == "__main__":
 run_smoke_test()
