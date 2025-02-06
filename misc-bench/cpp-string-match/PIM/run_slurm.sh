#!/bin/bash
sbatch --output=pim-text_len.log --job-name=text_len_pim_string_match slurm.sh "1mil_l-10_nk-10_kl" "10mil_l-10_nk-10_kl" "100mil_l-10_nk-10_kl" "1bil_l-10_nk-10_kl"
sbatch --output=pim-num_keys.log --job-name=num_keys_pim_string_match slurm.sh "1mil_l-1_nk-10_kl" "1mil_l-10_nk-10_kl" "1mil_l-100_nk-10_kl" "1mil_l-1000_nk-10_kl"
sbatch --output=pim-key_len.log --job-name=key_len_pim_string_match slurm.sh "1mil_l-10_nk-1_kl" "1mil_l-10_nk-10_kl" "1mil_l-10_nk-100_kl" "1mil_l-10_nk-1000_kl"