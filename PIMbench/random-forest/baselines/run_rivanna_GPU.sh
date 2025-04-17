#SBATCH -A LAVAlab
#SBATCH --job-name="GPU_a100_rf"
#SBATCH --error="./GPU_a100_rf.err"
#SBATCH --output="./GPU_a100_rf.out"
##SBATCH --mem=20G                                                                                    
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH -n 1
#SBATCH -p gpu 

module load apptainer 
module load rapidsai/24.06  # if running on cs, will need to build the image locally first

nvidia-smi

echo "done with module"

pip install scikit-learn
pip uninstall -y numpy
pip install numpy==1.26

apptainer exec nvidia-smi
apptainer exec --nv $CONTAINERDIR/rapidsai-24.06.sif python ./benchmark_rf.py -cuda