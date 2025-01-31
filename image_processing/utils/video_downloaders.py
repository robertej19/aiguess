import yt_dlp
import os
import re

def detect_platform(url):
    """
    Detects whether a given URL belongs to YouTube or Dailymotion.
    
    :param url: The video URL.
    :return: "youtube", "dailymotion", or None if unknown.
    """
    if re.search(r'(youtube\.com|youtu\.be)', url):
        return "youtube"
    elif re.search(r'dailymotion\.com', url):
        return "dailymotion"
    else:
        return None

def download_video(url, output_path="downloads"):
    """
    Downloads a video from YouTube or Dailymotion using yt-dlp,
    ensuring the video is not encoded in AV1.

    :param url: The video URL.
    :param output_path: The directory where the video will be saved.
    """
    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec!=av01]+bestaudio[ext=m4a]/mp4',  # Avoid AV1 codec
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),  # Filename pattern
        'merge_output_format': 'mp4',  # Ensure final output is MP4
        'noplaylist': True,  # Download only a single video
        'quiet': False,  # Show progress
        'no_warnings': True,
        'restrictfilenames': True,  # Use safe filenames
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Download completed! Video saved to '{output_path}' directory.")
    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading video: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    video_url = input("Enter the video URL: ").strip()
    download_directory = input("Enter the download directory (default 'downloads'): ").strip() or "downloads"
    download_video(video_url, download_directory)
