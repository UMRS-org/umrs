#!/bin/bash
#SBATCH -J dsv41-pro6000
#SBATCH -p ai-pro6000
#SBATCH -N 2
#SBATCH --mem=0
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -t 23:00:00
#SBATCH -o /home/users/masaru.nagaso/tmp/dsv41/logs/pro6000-%j.log
# Queue holder for the ai-pro6000 leg (DeepSeek-V4.1-Flash-1017 NVFP4 on SGLang + benchmarks). What runs is read at
# start time from pro6000_entry.sh, so the driver can be finished while this job waits in the queue.
exec bash "$HOME/tmp/dsv41/pro6000_entry.sh"
