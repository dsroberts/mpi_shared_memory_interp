#PBS -l ncpus=24
#PBS -l mem=700GB
#PBS -l walltime=3:00:00
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l jobfs=200GB
#PBS -q hugemem
#PBS -P xd2

cd $PBS_O_WORKDIR
mkdir -p /scratch/xd2/dr4292/downscaled_plumes/

module use /g/data/xd2/dr4292/apps/Modulefiles
module load firedrake

declare -a fns=( /g/data/xd2/rad552/FIREDRAKE_Simulations/DG_GPlates_Late_2024/GPlates_2e8/Cao/Output_Combined_CG/*tar.gz )
len="${#fns[@]}"

outfn=$( basename "${fns[0]}" .tar.gz )
echo "Untarring "$outfn


mkdir -p ${PBS_JOBFS}/${outfn}
tar -xf "${fns[0]}" -C ${PBS_JOBFS}/${outfn}

for i in $( seq 1 $(( "${len}" - 1 )) ); do
    
    nextfn=$( basename "${fns[$i]}" .tar.gz )
    echo "Untarring "$nextfn
    mkdir -p "${PBS_JOBFS}"/"${nextfn}"
    tar -xf "${fns[$i]}" -C "${PBS_JOBFS}"/"${nextfn}" &

    outfn=$( basename "${fns[$(( $i - 1 ))]}" .tar.gz )
    mpirun python3 mpi_shared_memory_interp.py --infile $PBS_JOBFS/${outfn}/output_0.pvtu --input_grid /scratch/xd2/dr4292/level5/level5_0.vtu --outfile /scratch/xd2/dr4292/downscaled_plumes/${outfn}.vtp --field Temperature_Deviation_CG
    
    rm -rf ${PBS_JOBFS}/${outfn}

    wait
done

outfn=$( basename "${fns[$(( $len - 1 ))]}" .tar.gz )
mkdir -p ${PBS_JOBFS}/${outfn}
mpirun python3 mpi_shared_memory_interp.py --infile $PBS_JOBFS/${outfn}/output_0.pvtu --input_grid /scratch/xd2/dr4292/level5/level5_0.vtu --outfile /scratch/xd2/dr4292/downscaled_plumes/${outfn}.vtp --field Temperature_Deviation_CG