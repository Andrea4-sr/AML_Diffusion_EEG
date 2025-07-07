#!/usr/bin/env bash

set -e

TRAIN_DATASET_REAL="data/real/classifier_train"
VALIDATION_REAL="data/real/validation"
TRAIN_DATASET_SYNTH="data/synthetic/classifier_train"
VALIDATION_SYNTH="data/synthetic/validation"

REAL_MODEL_DUMP_DIR="results/classifer_real"
SYNTH_MODEL_DUMP_DIR="results/classifer_synth"
SYNTH_MODEL_20k_DUMP_DIR="results/classifer_synth_20k"
RESULTS_REAL_DIR="results/"
RESULTS_SYNTH_DIR="results/"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Train the classifier on real data
echo "Training the classifier on real data..."
python3 src/train_classifier_steps.py --dataset_path "$TRAIN_DATASET_REAL" --model_dump_path "$REAL_MODEL_DUMP_DIR"

# Evaluate the classifier on real data (eval)
echo "Evaluating the classifier on real data..."
python3 src/evaluate_classifier_plot.py --model_path "$REAL_MODEL_DUMP_DIR" --dataset_path "$VALIDATION_REAL" --plot_path "$RESULTS_REAL_DIR" --metrics_path "$RESULTS_REAL_DIR/" --metrics_file_name "SVM_train_real_test_real" --plot_name "SVM_train_real_test_real"

# Evaluate the classifier on synthetic data
echo "Evaluating the classifier on synthetic data..."
python3 src/evaluate_classifier_plot.py --model_path "$REAL_MODEL_DUMP_DIR" --dataset_path "$VALIDATION_SYNTH" --plot_path "$RESULTS_REAL_DIR" --metrics_path "$RESULTS_REAL_DIR" --metrics_file_name "SVM_train_real_test_synth" --plot_name "SVM_train_real_test_synth"

# Train the classifier on synthetic data
echo "Training the classifier on synthetic data..."
python3 src/train_classifier_steps.py --dataset_path "$TRAIN_DATASET_SYNTH" --model_dump_path "$SYNTH_MODEL_DUMP_DIR"

# Evaluate the classifier on synthetic data
echo "Evaluating the classifier on synthetic data..."
python3 src/evaluate_classifier_plot.py --model_path "$SYNTH_MODEL_DUMP_DIR" --dataset_path "$VALIDATION_SYNTH" --plot_path "$RESULTS_SYNTH_DIR" --metrics_path "$RESULTS_SYNTH_DIR" --metrics_file_name "SVM_train_synth_test_synth" --plot_name "SVM_train_synth_test_synth"

# Train the classifier on synthetic data with 20k steps
echo "Training the classifier on synthetic data with 20k steps... (this might take 5-10 mins)"
python3 src/train_classifier_steps.py --dataset_path "$TRAIN_DATASET_SYNTH" --model_dump_path "$SYNTH_MODEL_20k_DUMP_DIR" --n_samples 20000

# Evaluate the classifier on real data
echo "Evaluating the classifier on real data..."
python3 src/evaluate_classifier_plot.py --model_path "$SYNTH_MODEL_20k_DUMP_DIR" --dataset_path "$VALIDATION_REAL" --plot_path "$RESULTS_SYNTH_DIR" --metrics_path "$RESULTS_SYNTH_DIR" --metrics_file_name "SVM_train_synth_test_real" --plot_name "SVM_train_synth_test_real"


echo "Done."
