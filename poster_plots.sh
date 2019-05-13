#!/usr/bin/env bash

scons -j 32

func_list=("real_space_mb_potential_density" "energy_level_mb_density" "density_result" "cond_fit")
plots_list=("plot" "semilogx" "loglog" "plot")
size_x=(108.4 108.4 167 142)
size_y=(132.5 132.5 80 96)

for ((i=0;i<${#func_list[@]};++i)); do
    printf "Plotting %s as %s, size=(%.1f,%.1f)\n" "${func_list[i]}" "${plots_list[i]}" ${size_x[i]} ${size_y[i]}
    python bin/poster_symp2019_plots.py "${func_list[i]}" "${plots_list[i]}" "${size_x[i]}" "${size_y[i]}"
    printf "\n"
done

