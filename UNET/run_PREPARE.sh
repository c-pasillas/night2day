#!/bin/bash -l
#SBATCH --account=rda-goesstf
#SBATCH --partition=hera
#SBATCH --time=00:05:00
#SBATCH --ntasks=8

date

#module load intel
#module load impi

cd /scratch1/RDARCH/rda-goesstf/conus2/Code

MYPY=/scratch1/RDARCH/rda-goesstf/anaconda/bin/python

srun -n $SLURM_NTASKS $MYPY MAIN_PREPARE_SAVE_DATA.py configuration.txt

date
