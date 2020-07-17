#!/usr/bin/env bash
scons -j 32

plots_folder='plots/papers/biexciton1'
final_folder='/storage/Reference/Work/University/PhD/OwnPapers/biexcitons1/figures'
#file_id='g1-G6zBRT0qYcnFlfl6k3A'
#file_id='HBITYggKRuOdz3QjfzF65w'
file_id='tgHpID7rSQ2ijHeclWlnMw'

args_list=()

# All, proportion
args_list+=("{\"include_free_charges\":1,\"include_excitons\":1,\"include_biexcitons\":1,\"degeneracy\":0,\"total_density\":0,\"draw_all_lines\":1,\"show_fig\":0,\"file_id\":\"$file_id\"}")

# All, proportion, sqrt
args_list+=("{\"include_free_charges\":1,\"include_excitons\":1,\"include_biexcitons\":1,\"degeneracy\":0,\"total_density\":0,\"draw_all_lines\":1,\"show_fig\":0,\"norm_sqrt\":1,\"file_id\":\"$file_id\"}")

# Free charges, proportion
args_list+=("{\"include_free_charges\":1,\"include_excitons\":0,\"include_biexcitons\":0,\"degeneracy\":0,\"total_density\":0,\"show_fig\":0,\"file_id\":\"$file_id\"}")

# Excitons, proportion
args_list+=("{\"include_free_charges\":0,\"include_excitons\":1,\"include_biexcitons\":0,\"degeneracy\":0,\"total_density\":0,\"show_fig\":0,\"file_id\":\"$file_id\"}")

# Biexcitons, proportion
args_list+=("{\"include_free_charges\":0,\"include_excitons\":0,\"include_biexcitons\":1,\"degeneracy\":0,\"total_density\":0,\"show_fig\":0,\"file_id\":\"$file_id\"}")

# Excitons, degeneracy
args_list+=("{\"include_free_charges\":0,\"include_excitons\":1,\"include_biexcitons\":0,\"degeneracy\":1,\"total_density\":0,\"show_fig\":0,\"file_id\":\"$file_id\"}")

# Biexcitons, degeneracy
args_list+=("{\"include_free_charges\":0,\"include_excitons\":0,\"include_biexcitons\":1,\"degeneracy\":1,\"total_density\":0,\"show_fig\":0,\"file_id\":\"$file_id\"}")

for ((i=0;i<${#args_list[@]};++i)); do
    printf "Plotting %s\n" "${args_list[i]}"
    python bin/biexciton_density_2d_v2.py ${args_list[i]}
    printf "\n"
done

cp $plots_folder/* $final_folder
