import os
import subprocess

def download_videos():
    output_dir = "input_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use yt-dlp to search and download 10 short traffic/dashcam videos
    # Limiting duration to < 300 seconds (5 mins) to avoid huge downloads
    search_query = 'ytsearch10:dashcam traffic license plates'
    command = [
        "python", "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/mp4",
        "--output", f"{output_dir}/video_%(autonumber)s.%(ext)s",
        "--match-filter", "duration < 300",
        search_query
    ]
    
    print("Downloading 10 traffic videos from YouTube. This may take a few minutes...")
    try:
        subprocess.run(command, check=True)
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading videos: {e}")

if __name__ == "__main__":
    download_videos()
