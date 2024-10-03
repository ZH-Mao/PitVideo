#!/bin/bash

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pitVideo

# Define input directory and output log directory
INDIR="/home/zhehua/codes/PitVideo-Segment-Landmark/experiments/pituitary"
OUTDIR="/home/zhehua/codes/PitVideo-Segment-Landmark/logs"

# Check if output directory exists, if not create it
if [ ! -d "$OUTDIR" ]; then
    echo "Output directory does not exist. Creating it now..."
    mkdir -p "$OUTDIR"
    if [ $? -eq 0 ]; then
        echo "Output directory created successfully."
    else
        echo "Failed to create output directory. Please check permissions and path."
        exit 1
    fi
else
    echo "Output directory already exists."
fi

# Loop through configuration files and run the Python script sequentially, saving output to a log file
for CONFIG in "video_hrnet_ssstan_w48_2stage_fold1.yaml" "video_hrnet_ssstan_w48_2stage_fold2.yaml" "video_hrnet_ssstan_w48_2stage_fold3.yaml" "video_hrnet_ssstan_w48_2stage_fold4.yaml" "video_hrnet_ssstan_w48_2stage_fold5.yaml"
do
    BASENAME=$(basename $CONFIG .yaml)  # Get the base name of the config file
    LOGFILE="$OUTDIR/${BASENAME}_output.txt"  # Define log file name

    # Run the Python script in the background and redirect output
    nohup python /home/zhehua/codes/PitVideo-Segment-Landmark/PituVideo_sstan_test_consecutive_IOU.py --cfg $INDIR/$CONFIG --gpu "1" > $LOGFILE 2>&1 &

    # Store the PID of the background process
    PID=$!

    # Wait for the current task to finish
    wait $PID

    echo "Process for $CONFIG completed."
done

echo "All processes have been executed sequentially."

# # run the script with "nohup bash test_sstan_scripts_fold1_2_3_4_5.sh > test_sstan_scripts_fold1_2_3_4_5.log 2>&1 &"