#!/bin/bash

# Check if TED1k_ROOT is set
if [ -z "$TED1k_ROOT" ]; then
    echo "Error: TED1k_ROOT environment variable is not set"
    exit 1
fi

# Run the extraction script
python ted1_extract_0.5fps.py \
    --videos_dir "$TED1k_ROOT/videos" \
    --keyframes_dir "$TED1k_ROOT/keyframes_0.5fps_shots"

