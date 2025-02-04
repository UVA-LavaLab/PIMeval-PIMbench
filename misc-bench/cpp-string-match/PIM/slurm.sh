#!/bin/bash
#SBATCH --job-name=pim_string_match_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=60
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=keg9ve@virginia.edu

make clean
make --directory=./../../.. clean
make --directory=./../../.. USE_OPENMP=1
make USE_OPENMP=1

pim_arch_config_prefix="./../../../configs/iiswc"
pim_arch_configs=("PIMeval_Bank_Rank32.cfg" "PIMeval_BitSerial_Rank32.cfg" "PIMeval_Fulcrum_Rank32.cfg")

prefix="./../dataset/"
for dir in "$prefix"*/; do
    test_dir=("$(basename "$dir")")
    for config_file in "${key_lens[@]}"; do
        echo "~~~~~~~~~~~~Starting Test~~~~~~~~~~~~"
        echo "${test_dir}"
        echo "${config_file}"
        echo "_____________________________________"
        ./string-match.out -k ${test_dir}/keys.txt -t ${test_dir}/text.txt -v t -c ${pim_arch_config_prefix}/${config_file}
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    done
done