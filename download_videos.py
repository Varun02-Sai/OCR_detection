import os
import subprocess

def download_videos():
    output_dir = "input_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use yt-dlp to search and download 20 short traffic camera videos
    # Limiting duration to < 60 seconds to avoid huge downloads and slow processing
    search_query = 'ytsearch20:nyc traffic camera intersection CCTV vehicles'
    command = [
        "python", "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/mp4",
        "--output", f"{output_dir}/nyc_traffic_%(autonumber)s.%(ext)s",
        "--match-filter", "duration < 90",
        search_query
    ]
    
    print("Downloading 20 NYC traffic camera videos from YouTube. This may take a few minutes...")
    try:
        subprocess.run(command, check=True)
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading videos: {e}")

if __name__ == "__main__":
    download_videos()
