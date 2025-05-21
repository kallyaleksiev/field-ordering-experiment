# field-ordering-experiment
Small experiment with AI

## Setup

To install the dependencies and setup your environment it is recommended to use `uv`:

```
uv sync
```

## Running

To run the experiment you can do use the `run_task` script.

```
$ python run_task.py --help
Usage: run_task.py [OPTIONS]

Options:
  --task [easy|hard]  Task type to run: easy or hard  [required]
  --agent [af|as]     Agent type: answer first (af) or answer second (as)
                      [required]
  --num-runs INTEGER  Number of evaluation runs to perform  [default: 1]
  --help              Show this message and exit.
```

so for example to run the easy task with the agent that cosniders the answer second a total of 4 times do:

```
LOGFIRE_CONSOLE=false python run_task.py --task easy --agent as --num-runs 4
```

## Results

Experiments ran with a command similar to `MODEL_NAME=openai:gpt-4.1-mini NUM_RUNS=10 ./run.sh`

###### Easy Task

| Model        | Answer First | Answer Second |
| ------------ | ------------ | ------------- |
| gpt 4.1      | 0.5227       | 0.5040        |
| gpt 4.1-mini | 0.4556       | 0.4515        |
| gpt 4o       | 0.4960       | 0.5103        |
| gpt 4o-mini  | 0.4205       | 0.4213        |

###### Hard Task

| Model        | Answer First | Answer Second |
| ------------ | ------------ | ------------- |
| gpt 4.1      | 0.0777       | 0.0647        |
| gpt 4.1-mini | 0.0385       | 0.1017        |
| gpt 4o       | 0.0696       | 0.0684        |
| gpt 4o-mini  | 0.0607       | 0.0787        |
