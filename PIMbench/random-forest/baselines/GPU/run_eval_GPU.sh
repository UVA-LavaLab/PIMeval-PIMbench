#!/bin/bash
#SBATCH --job-name="GPU_rf_eval"
#SBATCH --error="/u/kmd2zjw/pim_research_project/PIMeval-PIMbench/PIMbench/random-forest/baselines/GPU/GPU_random-forest.err"
#SBATCH --output="/u/kmd2zjw/pim_research_project/PIMeval-PIMbench/PIMbench/random-forest/baselines/GPU/GPU_random-forest.out"
#SBATCH --mem=10G                                                                                    
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_40gb
#SBATCH -n 1
#SBATCH -p gpu


echo "start module load"
module load gcccore/13.3.0
module load python/3.12.3  
pip install scikit-learn


# cd $SLURM_SUBMIT_DIR  # Ensure we're in the correct directory
# echo "done with cd"
# pwd
# python --version
# python train_DT.py


echo "start module load"
module load gcccore/13.3.0
module load python/3.12.3  
module load cuda/12.0.0
module load apptainer 
echo "done with module"

pip uninstall -y numpy
pip install numpy==1.26
apptainer exec --nv rapidsai_base.sif python /u/kmd2zjw/pim_research_project/PIMeval-PIMbench/PIMbench/random-forest/baselines/GPU/test_model.py