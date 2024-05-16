# EEG Classifier Project

This project contains scripts to train and evaluate machine learning classifiers for EEG data. The classifiers are trained on both real and synthetic data, and evaluated on both real and synthetic validation sets.

## Directory Structure

.
├── data
│ ├── real
│ │ ├── classifier_train
│ │ └── validation
│ ├── synthetic
│ │ ├── classifier_train
│ │ └── validation
├── results
│ ├── eval_trained_real
│ └── eval_trained_synth
├── trained_classifiers
│ ├── trained_on_real
│ └── trained_on_synth
├── src
│ ├── train_classifier_steps.py
│ └── evaluate_classifier_plot.py
├── requirements.txt
└── run_project.sh


## Setup

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

## Running the project

To reproduce our results, you can use our run_project.sh file: 

chmod +x run_project.sh

./run_project.sh


