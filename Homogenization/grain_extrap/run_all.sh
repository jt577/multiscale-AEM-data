#!/bin/bash
#SBATCH -J meso_jobs                    # will be overridden per‐task
#SBATCH -N 1
#SBATCH -c 40
#SBATCH -p hi_mem2
#SBATCH --time=999:00:00
#SBATCH --array=1-8                  
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err

source /data2/jt577/COSMO-RS/myenv/bin/activate

# parameter lists
units_list=(1)
tiles_list=(1)
geom_list=(lamellae)
sample_list=(0)
obs_list=(conductivity elasticity)
suff_list=(_coarse _fine _finer _finest)

# compute zero‐based index
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

# peel off obs
o=$(( idx % ${#obs_list[@]} )); idx=$(( idx / ${#obs_list[@]} ))
# peel off sample
s=$(( idx % ${#sample_list[@]} )); idx=$(( idx / ${#sample_list[@]} ))
# peel off geom
g=$(( idx % ${#geom_list[@]} )); idx=$(( idx / ${#geom_list[@]} ))
# peel off tiles
t=$(( idx % ${#tiles_list[@]} )); idx=$(( idx / ${#tiles_list[@]} ))
# peel off units
u=$(( idx % ${#units_list[@]} ))
# peel off suffix
suff=$(( idx % ${#suff_list[@]} )); idx=$(( idx / ${#suff_list[@]} ))


# select actual values
units=${units_list[$u]}
tiles=${tiles_list[$t]}
geom=${geom_list[$g]}
sample=${sample_list[$s]}
obs=${obs_list[$o]}
suffix=${suff_list[$suff]}

# build short job name, e.g. "35g2e"
abbr_geom=${geom:0:1}   # l or g
abbr_obs=${obs:0:1}     # c or e
newname="${units}${tiles}${abbr_geom}${sample}${abbr_obs}${suffix}"

# rename this array task
scontrol update JobId=${SLURM_JOB_ID} Name=${newname}

echo "[$(date +'%H:%M:%S')] Running: units=$units tiles=$tiles geom=$geom sample=$sample obs=$obs suff=$suffix"

# launch your Python
python main.py \
  --units  $units \
  --tiles  $tiles \
  --geom   $geom  \
  --suff   $suffix  \
  --sample $sample \
  --obs    $obs
