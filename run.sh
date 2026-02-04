#!/bin/bash
#PBS -q normal
#PBS -N serial_job
#PBS -l select=1:ncpus=128:mem=440G
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -P personal-le0003hi