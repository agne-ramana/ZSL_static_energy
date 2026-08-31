#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=cp2k_run_%j.out
#SBATCH --error=cp2k_run_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --account=e89-camm
#SBATCH --partition=standard
#SBATCH --qos=short

# Enforce strict error handling
set -e

# Load the exact same Python module used during environment creation
module load cray-python

# Activate the virtual environment from the /work directory
source /work/e89/e89/agne_r/my_compute_env/bin/activate

# Execute the script, utilizing your argparse structure for inputs
# Ensure your Python script explicitly saves the plot (e.g., plt.savefig('output.png'))
srun --exact -c 1 python collect_and_plot.py

echo "Done!"
