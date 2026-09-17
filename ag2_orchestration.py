import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

from typing import Annotated, Any
from autogen import ConversableAgent, LLMConfig

from orchestration_demo import (
    create_report,
    read_json,
    run_heat_simulation,
    validate_request,
    validate_results,
    write_json,
)


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to your .env file."
    )


llm_config = LLMConfig(
    {
        "api_type": "google",
        "model": "gemini-3.5-flash",
        "api_key": GEMINI_API_KEY,
        "temperature": 0,
    }
)


def parse_json(text: str)-> dict[str, Any]:
    """Extract one JSON object from an agent response."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("Agent response did not contain a JSON object.")
    
    return json.loads(match.group())

planner = ConversableAgent(
    name="planner",
    system_message=(
        "Create a compact heat-diffusion execution plan. "
        "Return JSON only with case_id, simulation, steps, and constraints. "
        "Use these steps: validate_request, run_heat_simulation, "
        "validate_results, generate_report."
    ),
    llm_config=llm_config,
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message=(
        "Review the supplied heat-diffusion request and plan. "
        "Check required steps, grid size, time_steps, temperatures, and "
        "0< diffusion_rate<= 0.25. "
        'Return JSON only: {"decision": "approve" or "revise", '
        '"issues": ["..."]}.'
    ),
    llm_config=llm_config,
)

executor = ConversableAgent(
    name="executor",
    system_message=(
        "Execute only an approved heat-diffusion request. "
        "Call heat_diffusion_simulation exactly once with the supplied values."
    ),
    llm_config=llm_config,
)

controller = ConversableAgent(
    name="controller",
    human_input_mode="NEVER",
    llm_config=False,
    code_execution_config=False,
)


@executor.register_for_llm(
    description="Run an approved two-dimensional heat-diffusion simulation."
)


@controller.register_for_execution()
def heat_diffusion_simulation(
    case_id: Annotated[str, "Unique case identifier"],
    grid_size: Annotated[int, "Plate width and height"],
    initial_temperature: Annotated[float, "Initial plate temperature"],
    boundary_temperature: Annotated[float, "Fixed boundary temperature"],
    hotspot_temperature: Annotated[float, "Initial center temperature"],
    time_steps: Annotated[int, "Number of solver iterations"],
    diffusion_rate: Annotated[float, "Diffusion rate from 0 to 0.25"],
    review_threshold: Annotated[float, "Threshold for manual review"],
)-> dict[str, Any]:
    
    request = {
        "case_id": case_id,
        "grid_size": grid_size,
        "initial_temperature": initial_temperature,
        "boundary_temperature": boundary_temperature,
        "hotspot_temperature": hotspot_temperature,
        "time_steps": time_steps,
        "diffusion_rate": diffusion_rate,
        "review_threshold": review_threshold,
    }

    results = run_heat_simulation(request)
    validate_results(results)
    report = create_report(request, results)

    # write_json("results.json", results)
    # write_json("report.json", report)
    write_json(Path("results.json"), results)
    write_json(Path("report.json"), report)

    return {"results": results, "report": report}


def main()-> None:
    # request = read_json("simulation_request.json")
    request = read_json(Path("simulation_request.json"))

    plan_chat = controller.initiate_chat(
        recipient=planner,
        message=json.dumps(request),
        max_turns=1,
        summary_method="last_msg",
    )

    plan = parse_json(plan_chat.summary)
    plan["constraints"] = {
        "min_grid_size": 3,
        "max_grid_size": 100,
        "max_time_steps": 10000,
        "max_diffusion_rate": 0.25,
    }
    # write_json("plan.json", plan)
    write_json(Path("plan.json"), plan)

    review_chat = controller.initiate_chat(
        recipient=reviewer,
        message=json.dumps({"request": request, "plan": plan}),
        max_turns=1,
        summary_method="last_msg",
    )
    review = parse_json(review_chat.summary)


    # Deterministic control gate: prompts do not replace hard limits.
    gate = validate_request(request, plan)

    if review["decision"] != "approve":
        gate = {
            "status": "failed",
            "checks": {"review": "failed"},
            "message": "; ".join(review["issues"]),
        }

    # write_json("gate_pre.json", gate)
    write_json(Path("gate_pre.json"), gate)

    if gate["status"] != "passed":
        print(f"Execution stopped: {gate['message']}")
        return
    
    controller.initiate_chat(
        recipient=executor,
        message=(
            "The controller approved this request. "
            "Call heat_diffusion_simulation with exactly these values:\n"
            f"{json.dumps(request)}"
        ),
        max_turns=2,
    )

    print("AG2 orchestration completed.")
    print("Created: plan.json")
    print("Created: gate_pre.json")
    print("Created: results.json")
    print("Created: report.json")


if __name__ == "__main__":
    main()