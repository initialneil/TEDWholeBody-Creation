import argparse
import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor

def find_video_by_id(video_id, video_dir):
    """Find if a video with the given ID already exists in the directory."""
    pattern = os.path.join(video_dir, f"{video_id}.*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def parse_args():
    parser = argparse.ArgumentParser(description="Download TED videos from list")
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Path to the video directory')
    parser.add_argument('--cookies_fn', type=str, required=True,
                        help='Path to the cookies file')
    parser.add_argument('--video_list_json', type=str, required=True,
                        help='Path to the JSON file containing video IDs')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel workers for downloading (default: 2)')
                        
    args = parser.parse_args()
    return args

def download_video_full(video_id, video_dir, cookies_fn):
    cmd = ' '.join([
        'yt-dlp',
        '--cookies', f'{cookies_fn}',
        '-f', 'bv+ba/best[ext=mp4]',
        '--merge-output-format', 'mp4',
        '--output', f'"{video_dir}/{video_id}.%(resolution)s.%(ext)s"',
        f'https://www.youtube.com/watch?v={video_id}'
    ])
    print('--------------------------------------------------')
    print(f"{cmd}")
    os.system(cmd)

def process_video(args_tuple):
    """Process a single video download task."""
    idx, video_id, total, video_dir, cookies_fn = args_tuple
    
    # Check if video already exists
    video_fn = find_video_by_id(video_id, video_dir)
    if video_fn is not None:
        print(f"[{idx+1}/{total}] Video {video_id} already exists: {video_fn}. Skipping download.")
        return video_id, 'skipped'
    
    print(f'[{idx+1}/{total}] Downloading video {video_id}')
    download_video_full(video_id, video_dir, cookies_fn)
    return video_id, 'downloaded'

if __name__ == "__main__":
    args = parse_args()
    print(f"Video directory: {args.video_dir}")
    print(f"Cookies file: {args.cookies_fn}")
    print(f"Video list JSON: {args.video_list_json}")
    print(f"Workers: {args.workers}")

    video_dir = args.video_dir
    os.makedirs(video_dir, exist_ok=True)

    # Load video IDs from JSON file
    with open(args.video_list_json, 'r', encoding='utf-8') as f:
        video_ids = json.load(f)
    
    print(f"Loaded {len(video_ids)} video IDs from JSON")
    print()
    
    # Prepare task tuples for parallel processing
    tasks = [(idx, video_id, len(video_ids), video_dir, args.cookies_fn) 
             for idx, video_id in enumerate(video_ids)]
    
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_video, tasks))
    
    # Print summary
    downloaded = sum(1 for _, status in results if status == 'downloaded')
    skipped = sum(1 for _, status in results if status == 'skipped')
    print()
    print(f"{'='*60}")
    print(f"Download complete!")
    print(f"Total videos: {len(video_ids)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already exist): {skipped}")


