#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=60
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=all

test_dirs=("$@")

pim_arch_config_prefix="./../../../configs/iiswc"
pim_arch_configs=("PIMeval_Bank_Rank32.cfg" "PIMeval_BitSerial_Rank32.cfg" "PIMeval_Fulcrum_Rank32.cfg")

for test_dir in "${test_dirs[@]}"; do
    for config_file in "${pim_arch_configs[@]}"; do
        echo "~~~~~~~~~~~~Starting Test~~~~~~~~~~~~"
        echo "${test_dir}"
        echo "${config_file}"
        echo "_____________________________________"
        ./string-match.out -k ./../dataset/${test_dir}/keys.txt -t ./../dataset/${test_dir}/text.txt -v t -c ${pim_arch_config_prefix}/${config_file}
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    done
done