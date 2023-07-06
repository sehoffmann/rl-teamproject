N_GPUS=$SLURM_NTASKS

TASK_ECHO="echo -e \"[\$SLURM_PROCID] (\$SLURM_NODEID.\$SLURM_LOCALID)\""
TASK_SERIAL_ECHO="sleep \`echo 0.5 \* \$SLURM_PROCID | bc\` && $TASK_ECHO"

srun_once_per_node() {
    # if you used gpus-per-task when submitting, you might need to overwrite it here
    # in order to allocate and be able to use all gpus
    # i.e. add --gpus-per-task=SLURM_GPUS_PER_NODE
    srun --ntasks-per-node=1 -n $SLURM_JOB_NUM_NODES  "$@"
}

copy_project() {
    # usage: copy_project PROJECT_FOLDER
    # copies PROJECT_FOLDER to $SCRATCH/project and prints the current state of the repo
    # to gurantee reproducibility
    local DIR=`pwd`

    echo "Copying $1 to SCRATCH"
    srun_once_per_node rclone copy $1 $SCRATCH/project

    cd $SCRATCH/project
    local GIT_DIFF=`git diff -U0 HEAD`
    echo "Most recent commit: `git log --oneline -n 1`"
    echo "Local diff:"
    git diff -U0 HEAD | paste /dev/null -

    cd $DIR
}


DEVICE_CHECKER="
import sys
for i in range(8):
    dev = '/dev/nvidia%d' % i
    try:
        open(dev, 'r').close()
        print(dev)
    except:
        pass
"

SBATCH_SCRIPT=`scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1`
SBATCH_SCRIPT_DIR=`dirname -- "$( readlink -f -- "$SBATCH_SCRIPT"; )"`


export SLURM_ALLOC_BIND="gn"

scontrol show job $SLURM_JOB_ID
echo "---------------------------------"

echo "Training distributed with $N_GPUS GPUs (/processes)"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "SBATCH_SCRIPT: $SBATCH_SCRIPT"
echo "SBATCH_SCRIPT_DIR: $SBATCH_SCRIPT_DIR"
echo "---------------------------------"

srun_once_per_node bash -c "$TASK_SERIAL_ECHO \"Topology:\n\`nvidia-smi topo -m\`\n\""
echo "---------------------------------"

echo "Available CPUs:"
srun_once_per_node bash -c "$TASK_SERIAL_ECHO \"\`cat /sys/fs/cgroup/cpuset/slurm/uid_$UID/job_$SLURM_JOB_ID/cpuset.cpus\`\""
echo "---------------------------------"

echo "Useable GPU devices:"
srun_once_per_node bash -c "$TASK_SERIAL_ECHO \`python -c \"$DEVICE_CHECKER\"\`   "
echo "---------------------------------"

echo "CPU Pinning:"
srun bash -c "$TASK_SERIAL_ECHO \"\t\`numactl -s | grep nodebind\` \t \`numactl -s | grep physcpubind\` \t \`numactl -s | grep membind\`\""
echo "---------------------------------"

echo "Slurm CPU bind:"
srun bash -c "$TASK_SERIAL_ECHO \"\$SLURM_CPU_BIND_TYPE \$SLURM_CPU_BIND_LIST\""
echo -e "============================================\n"


#horovodrun --check-build
