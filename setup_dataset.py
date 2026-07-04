import os
import shutil
import glob
import kagglehub
from pathlib import Path

def setup_kaggle_dataset():
    input_dir = Path("input_videos")
    input_dir.mkdir(exist_ok=True)
    
    # Check if we already have videos
    existing_videos = list(input_dir.glob("*.mp4"))
    if len(existing_videos) > 0:
        print(f"Found {len(existing_videos)} videos in input_videos. Skipping download.")
        return

    print("Downloading high-quality road crossing dataset from Kaggle...")
    print("This might take a minute, but Colab has extremely fast internet!")
    
    # Download dataset (kagglehub handles public datasets without login)
    dataset_path = kagglehub.dataset_download('saisirishan/car-number-plate-video')
    print(f"Downloaded to {dataset_path}")
    
    # Find all videos in the dataset (case-insensitive for extensions)
    exts = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV', '*.mkv']
    all_videos = []
    for ext in exts:
        all_videos.extend(glob.glob(f"{dataset_path}/**/{ext}", recursive=True))
    
    if not all_videos:
        print("No videos found in the dataset! Try manually checking the Kaggle dataset contents.")
        return
        
    # Sort videos by file size descending (to get the highest quality/longest ones)
    all_videos.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    # Pick the top 20 best videos
    best_videos = all_videos[:20]
    
    print(f"Extracting the {len(best_videos)} highest quality videos...")
    for idx, vid_path in enumerate(best_videos, 1):
        # We'll rename them neatly
        ext = Path(vid_path).suffix
        dest_path = input_dir / f"kaggle_video_{idx:02d}{ext}"
        shutil.copy2(vid_path, dest_path)
        
    print(f"Successfully loaded {len(best_videos)} videos into {input_dir}!")

if __name__ == "__main__":
    setup_kaggle_dataset()
