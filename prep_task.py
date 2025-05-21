import random
import string

import PIL
import PIL.Image
from datasets import load_dataset
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

PAINTING_DATASET = load_dataset("keremberke/painting-style-classification", "full")
LABEL_NAMES = PAINTING_DATASET["train"].features["labels"].names

# random strings that are used in the harder version of the task
AUXILIARY_LS = ["".join(random.choices(string.ascii_lowercase, k=i)) for i in range(30)]

EASY_PYDANTIC_PAINTING_DATASET = Dataset[PIL.Image.Image, int](
    cases=[
        Case(
            inputs=PAINTING_DATASET["test"][i]["image"],
            expected_output=PAINTING_DATASET["test"][i]["labels"],
        )
        for i in range(len(PAINTING_DATASET["test"]))
    ],
    evaluators=[
        EqualsExpected(),
    ],
)


HARD_PYDANTIC_PAINTING_DATASET = Dataset[PIL.Image.Image, int](
    cases=[
        Case(
            inputs=PAINTING_DATASET["test"][i]["image"],
            expected_output=AUXILIARY_LS[PAINTING_DATASET["test"][i]["labels"] + 2],
        )
        for i in range(len(PAINTING_DATASET["test"]))
    ],
    evaluators=[
        EqualsExpected(),
    ],
)
