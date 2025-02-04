#!/bin/sh
text_lens=("1000000" "10000000" "100000000" "1000000000")
text_lens_names=("1mil" "10mil" "100mil" "1bil")
num_keys=("1" "10" "100" "1000")
key_lens=("1" "10" "100" "1000")

for text_len_ind in "${!text_lens[@]}"; do
    for num_key in "${num_keys[@]}"; do
        for key_len in "${key_lens[@]}"; do
            name="${text_lens_names[$text_len_ind]}_l-${num_key}_nk-${key_len}_kl"
            ./data-generator.out -l ${text_lens[$text_len_ind]} -n ${num_key} -m ${key_len} -x ${key_len} -o ${name}
            echo "Generated: ${name}"
        done
    done
done