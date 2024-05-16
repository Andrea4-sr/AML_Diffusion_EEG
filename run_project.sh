#!/usr/bin/env bash
# This script sets up the environment and runs the Python project.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define project-related paths
TRAIN_DATASET_REAL="data/real/classifier_train"
VALIDATION_REAL="data/real/validation"
TRAIN_DATASET_SYNTH="data/synthetic/classifier_train"
VALIDATION_SYNTH="data/synthetic/validation"

REAL_MODEL_DUMP_DIR="tutors/trained_classifiers/trained_on_real"
SYNTH_MODEL_DUMP_DIR="tutors/trained_classifiers/trained_on_synth"
RESULTS_REAL_DIR="tutors/results/eval_trained_real"
RESULTS_SYNTH_DIR="tutors/results/eval_trained_synth"


# Check if required commands are available
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 is required but it's not installed. Aborting."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo >&2 "pip is required but it's not installed. Aborting."; exit 1; }


# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Ensure necessary directories exist
echo "Checking directories..."

directories=(
  "$TRAIN_DATASET_REAL"
  "$VALIDATION_REAL"
  "$TRAIN_DATASET_SYNTH"
  "$VALIDATION_SYNTH"
  "$RESULTS_REAL_DIR"
  "$RESULTS_SYNTH_DIR"
)

for dir in "${directories[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Directory $dir does not exist. Creating..."
    mkdir -p "$dir"
  else
    echo "Directory $dir already exists."
  fi
done



# Train the classifier on real data
echo "Training the classifier on real data..."
python3 src/train_classifier_steps.py --dataset_path "$TRAIN_DATASET_REAL" --model_dump_path "$REAL_MODEL_DUMP_DIR"

# Evaluate the classifier on real data (eval)
echo "Evaluating the classifier on real data..."
python3 src/evaluate_classifier_plot.py --model_path "$REAL_MODEL_DUMP_DIR" --dataset_path "$VALIDATION_REAL" --plot_path "$RESULTS_REAL_DIR" --metrics_path "$RESULTS_REAL_DIR/" --metrics_file_name "svc_eval_real_real" --plot_name "svc_eval_real_real"

# Evaluate the classifier on  synthetic data
echo "Evaluating the classifier on synthetic data..."
python3 src/evaluate_classifier_plot.py --model_path "$REAL_MODEL_DUMP_DIR" --dataset_path "$VALIDATION_SYNTH" --plot_path "$RESULTS_REAL_DIR" --metrics_path "$RESULTS_REAL_DIR" --metrics_file_name "svc_eval_real_synth" --plot_name "svc_eval_real_synth"


echo "Now we train the classifier on synthetic data and evaluate on real and synthetic data."

# Train the classifier on real data
echo "Training the classifier on synthetic data..."
python3 src/train_classifier_steps.py --dataset_path "$TRAIN_DATASET_SYNTH" --model_dump_path "$SYNTH_MODEL_DUMP_DIR"

echo "Evaluating the classifier on real data"
python3 src/evaluate_classifier_plot.py --model_path "$SYNTH_MODEL_DUMP_DIR" --dataset_path "$VALIDATION_REAL" --plot_path "$RESULTS_SYNTH_DIR" --metrics_path "$RESULTS_SYNTH_DIR" --metrics_file_name "svc_eval_synth_real" --plot_name "svc_eval_synth_real"

echo "Evaluating the classifier on synthetic data"
python3 src/evaluate_classifier_plot.py --model_path "$SYNTH_MODEL_DUMP_DIR" --dataset_path "$VALIDATION_SYNTH" --plot_path "$RESULTS_SYNTH_DIR" --metrics_path "$RESULTS_SYNTH_DIR" --metrics_file_name "svc_eval_synth_synth" --plot_name "svc_eval_synth_synth"


echo "Done."
