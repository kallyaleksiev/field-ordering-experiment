#!/bin/bash

set -e

MODEL_NAME=${MODEL_NAME:-"openai:gpt-4.1-mini"} 
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | sed 's/:/_/g')
# Define the number of runs - use environment variable NUM_RUNS if set, otherwise default.
NUM_RUNS=${NUM_RUNS:-5}

# Fixed
TASKS=("easy" "hard")
AGENTS=("af" "as")


RESULTS_DIR="results/${MODEL_NAME_SAFE}"
mkdir -p "$RESULTS_DIR"

echo "Using model: $MODEL_NAME"
echo "Running $NUM_RUNS evaluation runs for each configuration."


for task in "${TASKS[@]}"; do
  for agent in "${AGENTS[@]}"; do
    OUTPUT_FILE="$RESULTS_DIR/${task}_${agent}.txt"
    
    echo "----------------------------------------------------"
    echo "Running Task: $task, Agent: $agent"
    echo "Output will be saved to: $OUTPUT_FILE"
    echo "----------------------------------------------------"
    
    # Run the python script, pipe output to tee to stream and save
    LOGFIRE_CONSOLE=false MODEL_NAME="$MODEL_NAME" uv run python -u run_task.py \
      --task "$task" \
      --agent "$agent" \
      --num-runs "$NUM_RUNS" 2>&1 | tee "$OUTPUT_FILE"
      
    echo "Finished Task: $task, Agent: $agent. Results saved."
    echo "----------------------------------------------------"
    sleep 30
  done
done

echo "All experiments completed."
echo "Results are stored in the '$RESULTS_DIR' directory." 