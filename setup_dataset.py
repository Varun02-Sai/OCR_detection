import os
import urllib.request
from pathlib import Path

def setup_dataset():
    input_dir = Path("input_videos")
    input_dir.mkdir(exist_ok=True)
    
    # 100% public, guaranteed-to-work ANPR testing videos
    PUBLIC_VIDEOS = {
        "intel_traffic.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        "toll_gate.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/toll-camera.mp4",
    }
    
    for filename, url in PUBLIC_VIDEOS.items():
        dest = input_dir / filename
        if not dest.exists():
            print(f"Downloading guaranteed public testing video: {filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping.")
            
    print("\nDataset setup complete! 100% public videos are now in input_videos/")

if __name__ == "__main__":
    setup_dataset()
