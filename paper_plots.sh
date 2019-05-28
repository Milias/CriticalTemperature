#!/usr/bin/env bash

scons -j 32

func_list=()
plots_list=()

#func_list+=("scr_length_density")
#plots_list+=("loglog")

#func_list+=("real_space_lwl_potential")
#plots_list+=("plot")

#func_list+=("real_space_mb_potential_density")
#plots_list+=("plot")

#func_list+=("energy_level_mb_density")
#plots_list+=("semilogx")

#func_list+=("density_result")
#plots_list+=("loglog")

#func_list+=("eb_photo_density")
#plots_list+=("semilogx")

#func_list+=("cond_fit")
#plots_list+=("plot")

func_list+=("mobility_2d_sample")
plots_list+=("semilogx")

#func_list+=("mobility_2d_integ")
#plots_list+=("loglog")

for ((i=0;i<${#func_list[@]};++i)); do
    printf "Plotting %s as %s\n" "${func_list[i]}" "${plots_list[i]}"
    python bin/paper_exciton1_plots.py "${func_list[i]}" "${plots_list[i]}"
    printf "\n"
done

