#!/bin/bash

# Check if TED1k_ROOT is set
if [ -z "$TED1k_ROOT" ]; then
    echo "Error: TED1k_ROOT environment variable is not set"
    exit 1
fi

# Run the download script
python ted0_download_ted_videos.py \
    --video_dir "$TED1k_ROOT/videos" \
    --cookies_fn "www.youtube.com_cookies.txt" \
    --video_list_json "ted_video_list.json" \
    --workers 2

