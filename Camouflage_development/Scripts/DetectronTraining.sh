#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 8
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH --gres=gpu:V100:1
#SBATCH --job-name=development1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=o.fernandez@oist.jp


# create a temporary directory for this job and save the name
tempdir=$(mktemp -d /work/ReiterU/detectron.XXXXXX)
mkdir /work/ReiterU/temp_videos

# enter the temporary directory
cd $tempdir

# Start 'myprog' with input from bucket,
# and output to our temporary directory
#myprog /bucket/MyUnit/mydata.dat -o output.dat

conda activate texture4
cp /apps/unit/ReiterU/olivier/temp/DetectronTraining.py $tempdir
python3 DetectronTraining.py



# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
scp -r $tempdir saion:/bucket/ReiterU/olivier/

# Clean up by removing our temporary directory
rm -r $tempdir

