#!/bin/bash -xe

JOBID=$1
BASEDIR=$2
X509PATH=$3

# Set environment variables for xrootd and proxy certificate
export XRD_NETWORKSTACK=IPv4
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=$X509PATH

# Print proxy certificate details to verify validity and VOMS attributes
voms-proxy-info -all -file $X509PATH

# Save the current working directory where JSON configuration files are expected to be
WORKDIR=`pwd`

# Declare an associative array (dictionary) to hold job parameters
declare -A ARGS

# Extract specific job parameters from arguments.json
for key in workflow year output_path output_format dataset eos user; do
    ARGS[$key]=$(python3 -c "import json; print(json.load(open('$WORKDIR/arguments.json'))['$key'])")
done

# Build the full set of command-line parameters for submit.py (Add the JOBID suffix to the dataset name to uniquely identify the output)
CMD_ARGS="--workflow ${ARGS[workflow]} --year ${ARGS[year]} --output_path ${ARGS[output_path]} --output_format ${ARGS[output_format]} --dataset ${ARGS[dataset]}_$JOBID --user ${ARGS[user]}"

if [ "${ARGS[eos]}" = "True" ]; then
    CMD_ARGS="$CMD_ARGS --eos"
fi

# From partitions.json (which contains the partitioning of the full dataset across jobs),
# extract only the subset assigned to the current JOBID and save it as partition_fileset.json.
# This ensures each job processes a unique subset of the full dataset
python3 -c "import json; json.dump(json.load(open('$WORKDIR/partitions.json'))['$JOBID'], open('$WORKDIR/partition_fileset.json', 'w'), indent=4)"

# Add the newly created partition file to the list of arguments to pass to submit.py
CMD_ARGS="$CMD_ARGS --partition_json $WORKDIR/partition_fileset.json"
echo $CMD_ARGS

# Call the submit.py script with all collected job parameters
cd $BASEDIR
python3 submit.py $CMD_ARGS