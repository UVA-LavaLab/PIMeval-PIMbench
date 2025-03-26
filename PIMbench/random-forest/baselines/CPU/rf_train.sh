#!/bin/bash
#SBATCH --job-name="GPU_rf_train"
#SBATCH --error="/u/kmd2zjw/pim_research_project/PIMeval-PIMbench/PIMbench/random-forest/baselines/CPU/GPU_random-forest.err"
#SBATCH --output="/u/kmd2zjw/pim_research_project/PIMeval-PIMbench/PIMbench/random-forest/baselines/CPU/GPU_random-forest.out"
#SBATCH --mem=100G                                                                                    
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -p gpu

#source /etc/profile.d/modules.sh

echo "start module load"
module load gcccore/13.3.0
module load python/3.12.3  
pip install scikit-learn


cd $SLURM_SUBMIT_DIR  # Ensure we're in the correct directory
echo "done with cd"
pwd
python --version
python train_DT.py
