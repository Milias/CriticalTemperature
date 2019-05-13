#!/usr/bin/env bash

scons -j 32

func_list=("density_result" "eb_photo_density" "cond_fit" "mobility_2d_integ")
plots_list=("loglog" "semilogx" "plot" "loglog")

for ((i=0;i<${#func_list[@]};++i)); do
    printf "Plotting %s as %s\n" "${func_list[i]}" "${plots_list[i]}"
    python bin/paper_exciton1_plots.py "${func_list[i]}" "${plots_list[i]}"
    printf "\n"
done

#python bin/plasmons_density_ht_v.py
