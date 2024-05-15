#!/usr/bin/env bash
# This script sets up the environment and runs the Python project.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define project-related paths
DATASET_REAL="data/real/classifier_train"
DATASET_VALIDATION="data/real/validation"
DATASET_SYNTH_NOISY="data/synthetic/classifier_train"
MODEL_DUMP_DIR="trained_classifiers/increasing_samples_2"
RESULTS_DIR="results"

# Check if required commands are available
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 is required but it's not installed. Aborting."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo >&2 "pip is required but it's not installed. Aborting."; exit 1; }


# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Train the classifier on real data
echo "Training the classifier on real data..."
python3 src/train_classifier_steps.py --dataset_path "$DATASET_REAL" --model_dump_path "$MODEL_DUMP_DIR"

# Evaluate the classifier on real data (eval)
echo "Evaluating the classifier on real data..."
python3 src/evaluate_classifier_plot.py --model_path "$MODEL_DUMP_DIR" --dataset_path "$DATASET_VALIDATION" --plot_path "$RESULTS_DIR/" --metrics_path "$RESULTS_DIR/" --metrics_file_name "svc_eval_real" --plot_name "SVC_tested_eval_real"

# Evaluate the classifier on noisy synthetic data
echo "Evaluating the classifier on noisy synthetic data..."
python3 src/evaluate_classifier_plot.py --model_path "$MODEL_DUMP_DIR" --dataset_path "$DATASET_SYNTH_NOISY" --plot_path "$RESULTS_DIR/" --metrics_path "$RESULTS_DIR/" --metrics_file_name "svc_eval_synth_noisy" --plot_name "SVC_tested_eval_synth_noisy"


echo "Done."
