Log into Bouchet and set up the core machine learning environment.

# 1. Navigate to your specific project folder
cd /nfs/roberts/project/cpsc4520/cpsc4520_jcp87/factor-model

# 2. Load the Conda module and create the environment
module load miniconda
conda create -n factor_env python=3.9 -y
conda activate factor_env

# 3. Install ONLY the core ML libraries and NVIDIA-specific PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install pyqlib pandas numpy scipy

Request a quick interactive session just to download the standard 2000–2020 dataset safely.

# 1. Request an interactive node
salloc --time=01:00:00 --mem=8G --cpus-per-task=2

# --- Wait for the prompt to change to a compute node ---

# 2. Reactivate your environment
module load miniconda
conda activate factor_env

# 3. Download the base 2000-2020 dataset to your home directory
python -c "from qlib.tests.data import GetData; GetData().qlib_data(target_dir='~/.qlib/qlib_data/us_data', region='us')"

# 4. Exit the interactive node to return to the login node
exit

Go back to your main project folder and create your Slurm script.

cd /nfs/roberts/project/cpsc4520/cpsc4520_jcp87/factor-model
nano run_factor.sh

Paste this configuration (we can drop the time limit to 12 hours since we aren't processing the extra 6 years of scraped data):

#!/bin/bash
#SBATCH --job-name=factor_vae
#SBATCH --output=factor_vae_%j.out   # Standard output and error log
#SBATCH --partition=gpu              # Request the GPU partition
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --cpus-per-task=8            # 8 cores to speed up data loading
#SBATCH --mem=64G                    # 64GB RAM for the 158 features
#SBATCH --time=12:00:00              # Safely under limits

# 1. Load modules and activate your environment
module load miniconda
conda activate factor_env

# 2. Navigate exactly to your project directory
cd /nfs/roberts/project/cpsc4520/cpsc4520_jcp87/factor-model

# 3. Execute the code
echo "Starting FactorVAE Training on Bouchet GPU..."
python main.py

(If you want to run a quick test job just to see the Rank IC without re-training, simply change the execution line at the bottom of your Slurm script to:
python main.py --mode eval)

(Press Ctrl + O, hit Enter to save, then press Ctrl + X to exit).

Submit your job to the supercomputer:

sbatch run_factor.sh

Watch your model train live by reading the output file:

tail -f factor_vae_*.out