import statistics
from io import BytesIO

import click
import PIL
from pydantic_ai import BinaryContent

from agents import et_af_agent, et_as_agent, ht_af_agent, ht_as_agent
from prep_task import EASY_PYDANTIC_PAINTING_DATASET, HARD_PYDANTIC_PAINTING_DATASET


# --- Easy Task Functions ---
async def solve_et_af(image: PIL.Image.Image) -> int:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    result = await et_af_agent.run(
        [
            "What is the style of this painting?",
            BinaryContent(data=image_bytes, media_type="image/jpeg"),
        ]
    )
    return result.output.label


async def solve_et_as(image: PIL.Image.Image) -> int:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    result = await et_as_agent.run(
        [
            "What is the style of this painting?",
            BinaryContent(data=image_bytes, media_type="image/jpeg"),
        ]
    )
    return result.output.label


# --- Hard Task Functions ---
async def solve_ht_af(image: PIL.Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    result = await ht_af_agent.run(
        [
            "What is the style of this painting?",
            BinaryContent(data=image_bytes, media_type="image/jpeg"),
        ]
    )
    return result.output.answer


async def solve_ht_as(image: PIL.Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    result = await ht_as_agent.run(
        [
            "What is the style of this painting?",
            BinaryContent(data=image_bytes, media_type="image/jpeg"),
        ]
    )
    return result.output.answer


@click.command()
@click.option(
    "--task",
    type=click.Choice(["easy", "hard"]),
    required=True,
    help="Task type to run: easy or hard",
)
@click.option(
    "--agent",
    type=click.Choice(["af", "as"]),
    required=True,
    help="Agent type: answer first (af) or answer second (as)",
)
@click.option(
    "--num-runs",
    type=int,
    default=1,
    show_default=True,
    help="Number of evaluation runs to perform",
)
def main(task: str, agent: str, num_runs: int):
    print(f"Solving task '{task}' with agent '{agent}' over {num_runs} run(s)...")

    if task == "easy":
        dataset = EASY_PYDANTIC_PAINTING_DATASET
        solve_fn_af = solve_et_af
        solve_fn_as = solve_et_as
        task_name = "Easy Task"
    else:  # task == "hard"
        dataset = HARD_PYDANTIC_PAINTING_DATASET
        solve_fn_af = solve_ht_af
        solve_fn_as = solve_ht_as
        task_name = "Hard Task"

    all_assertions = []
    last_report = None

    for i in range(num_runs):
        print(f"Starting run {i + 1}/{num_runs}...")
        if agent == "af":
            report = dataset.evaluate_sync(
                solve_fn_af,
                name=f"{task_name} Answer First Agent (Run {i + 1})",
                max_concurrency=50,
            )
        else:  # agent == "as"
            report = dataset.evaluate_sync(
                solve_fn_as,
                name=f"{task_name} Answer Second Agent (Run {i + 1})",
                max_concurrency=50,
            )

        averages = report.averages()
        if averages and hasattr(averages, "assertions"):
            assertion_value = averages.assertions
            assert isinstance(assertion_value, float), (
                "We assume the average assertions result is a float"
            )
            all_assertions.append(assertion_value)

        last_report = report
        print(f"Finished run {i + 1}/{num_runs}.")

    if last_report:
        print("--- Report from Last Run ---")
        last_report.print(
            include_input=False,
            include_output=True,
            include_durations=False,
            include_expected_output=True,
        )
    else:
        print("No successful runs completed.")

    if len(all_assertions) > 0:
        mean_assertions = statistics.mean(all_assertions)
        print(f"\n--- Overall Statistics ({len(all_assertions)} runs) ---")
        print(f"Mean Assertions: {mean_assertions:.4f}")
        if len(all_assertions) > 1:
            stdev_assertions = statistics.stdev(all_assertions)
            print(f"Standard Deviation Assertions: {stdev_assertions:.4f}")
        else:
            print("Standard Deviation: N/A (only 1 run)")
    else:
        print("No assertion results collected.")


if __name__ == "__main__":
    main()
