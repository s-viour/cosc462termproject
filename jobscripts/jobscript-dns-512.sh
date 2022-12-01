#!/bin/bash

##SBATCH -J mwyatt14termproject ## Name of job, you can define it to be any word

#SBATCH -A isaac-utk0202        ## Information about the project account to be charged

#SBATCH --nodes=1               ## Number of nodes

#SBATCH --ntasks-per-node=512     ##-ntasks is used when we want to define total number of processors

#SBATCH --time=01:00:00         ## request resources for one hour  hh:mm:ss

#SBATCH --partition=condo-cs462 ## Campus-<name> are partitions that is shard between all UT campuses. use 'sinfo' to check available partitions

#SBATCH -e cannon.stderr        ## Errors will be written to this file

#SBATCH -o cannon.stdout        ## standard output written to this file. It is recommended that error and standard output be written to the same file

#SBATCH --qos=condo             ## must use the qos associated with the partition

# And finally run the job
srun ./build/dns 1024
