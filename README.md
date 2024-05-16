# EEG Classifier Project

In this project we used EEG data (healthy and seizure data) to train a diffusion model, which then generates synthetic EEG (healhty and seizure) data. We train two simple SVM on Fast Fourier Transform features from the EEG signals: the first trained on real data and the second on synthetic data, and we evaluate their AUROCs every 200 samples on both real and synthetic data. 

This repo contains scripts to train and evaluate the SVMs. 

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

To reproduce our results, you can use our run_project.sh file. Running this file will create a folder called "tutors" where the SVMs trained on increasing number of samples will be stored, as well as the evaluation results (metrix.txt and auroc plots).

   ```sh
   chmod +x run_project.sh

   ./run_project.sh


