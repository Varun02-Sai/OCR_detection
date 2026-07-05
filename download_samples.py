import os
import subprocess
from pathlib import Path

# URLs and time ranges for test videos
TEST_VIDEOS = {
    "traffic_camera_test.mp4": {
        "url": "https://www.youtube.com/watch?v=wqctLW0Hb_0",
        "sections": "*00:00:05-00:00:20"
    },
    "toll_gate_test.mp4": {
        "url": "https://www.youtube.com/watch?v=2CIhGxkSCmo",
        "sections": "*00:00:10-00:00:25"
    },
    "highway_test.mp4": {
        "url": "https://www.youtube.com/watch?v=MNn9qKG2UFI",
        "sections": "*00:00:02-00:00:17"
    }
}

def check_yt_dlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def main():
    print("=" * 60)
    print("  ANPR Pipeline - Downloading Test Videos")
    print("=" * 60)

    if not check_yt_dlp():
        print("\n[ERROR] yt-dlp is not installed or not in PATH.")
        print("Please install it using: pip install yt-dlp")
        return

    sample_dir = Path(__file__).resolve().parent / "sample_videos"
    sample_dir.mkdir(exist_ok=True)
    print(f"\nDownloading to: {sample_dir}\n")

    for filename, info in TEST_VIDEOS.items():
        dest_path = sample_dir / filename
        
        if dest_path.exists():
            print(f"  [SKIP] {filename} already exists.")
            continue

        print(f"  [DOWNLOADING] {filename}...")
        try:
            # Command to download specific sections in optimal quality, saved as mp4
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--download-sections", info["sections"],
                "--output", str(dest_path),
                info["url"]
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"  [OK] Successfully downloaded {filename}")
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] Error downloading {filename}: {e}")
        except Exception as e:
            print(f"  [FAIL] Unexpected error: {e}")

    print("\n" + "=" * 60)
    print("  Download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
