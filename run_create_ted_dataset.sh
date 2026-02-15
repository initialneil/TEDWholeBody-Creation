#!/bin/bash

# Set TEDWB1k_ROOT to your dataset path
export TEDWB1k_ROOT=${TEDWB1k_ROOT:-"/path/to/ted1k_dataset"}

# Download TED videos
./run_ted0_download_ted_videos.sh

# Run the extraction script
./run_ted1_extract.sh
