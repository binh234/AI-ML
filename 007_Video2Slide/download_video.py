import mimetypes
import pytube
import tempfile
import requests


def download_video_from_url(url):
    try:
        # Send a GET request to the URL to retrieve the video content
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Get the MIME type from the response headers
        content_type = response.headers.get("content-type")
        file_extension = mimetypes.guess_extension(content_type)

        # Create a temporary file to save the video
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name

        # Write the video content to the temporary file
        with open(temp_file_path, "wb") as file:
            file.write(response.content)

        # Return the path of the downloaded video file
        return temp_file_path

    except requests.exceptions.RequestException as e:
        print("An error occurred while downloading the video:", str(e))
        return None


def download_video_from_youtube(url):
    # Create a temporary file to save the video
    temp_file = tempfile.NamedTemporaryFile(suffix="mp4", delete=False)
    temp_file_path = temp_file.name

    try:
        # Create a YouTube object and retrieve the video
        youtube = pytube.YouTube(url)
        video = youtube.streams.first()

        # Download the video to the temporary file
        video.download(output_path=temp_file.name, filename="temp_video")

        # Return the path of the downloaded video file
        return temp_file_path

    except Exception as e:
        print("An error occurred while downloading the video:", str(e))
        return None


if __name__ == "__main__":
    youtube_link = "https://www.youtube.com/watch?v=2OTq15A5s0Y"
    temp_video_path = download_video_from_youtube(youtube_link)

    if temp_video_path is not None:
        print("Video downloaded successfully to:", temp_video_path)
    else:
        print("Failed to download the video.")
