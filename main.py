import cv2
import os
import subprocess
import requests
from custom_logger import CustomLogger
from logmod import logs
import common
import sys

# Initialize logging
logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# Load the base data folder from the config
base_data_folder = common.get_configs("data")

# Define paths using os.path.join
final_data = os.path.join(base_data_folder, "final")
original_data = os.path.join(base_data_folder, "original")
img2_output_data = os.path.join(base_data_folder, "img2turbo_output")
compare_folders = os.path.join(base_data_folder, "compare")
transformation = common.get_configs("transformation")


def video_to_frames(video_path):
    """
    Extracts frames from a video and saves them as image files.

    Args:
        video_path (str): Path to the video file to process.
    """
    # Get the directory where the video is located
    output_folder = os.path.dirname(video_path)

    # Extract video file name without extension to create frame names
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        # If the frame was read correctly
        if ret:
            # Save the frame as an image file in the same directory as the video
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)

            frame_count += 1
        else:
            break

    # Release the video capture object
    cap.release()
    logger.info(f"Extracted {frame_count} frames to {output_folder}")


def process_videos_in_directory(root_folder):
    """
    Processes all video files in a directory by extracting their frames.

    Args:
        root_folder (str): Path to the root directory containing video files.
    """
    # Walk through all subdirectories and files
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.mp4'):
                # Full path to the video file
                video_path = os.path.join(dirpath, file)

                # Process the video file, saving frames in the same directory
                video_to_frames(video_path)


def process_all_folders(frame_folder, fps=30):
    """
    Converts a sequence of image frames in a folder into a video.

    Args:
        frame_folder (str): Path to the folder containing image frames.
        fps (int): Frames per second for the output video.
    """
    # Get the list of all frames sorted by filename (assumes sequential naming)
    frames = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])

    # Check if there are frames to process
    if not frames:
        return  # Skip if no frames found in this folder

    # Extract the folder name to use as the video file name
    folder_name = os.path.basename(os.path.normpath(frame_folder))
    output_video_path = os.path.join(frame_folder, f"{folder_name}.mp4")

    # Get the full path of the first frame to read its dimensions
    first_frame_path = os.path.join(frame_folder, frames[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    # Define the codec and create the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate over the sorted frame list and write each one to the video
    for frame_filename in frames:
        frame_path = os.path.join(frame_folder, frame_filename)
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the video writer object
    out.release()
    logger.info(f"Video saved as {output_video_path}")


def frames_to_video(root_folder, fps=30):
    """
    Processes all directories in a root folder containing image frames and creates videos.

    Args:
        root_folder (str): Path to the root directory containing image frames.
        fps (int): Frames per second for the output videos.
    """
    # Walk through all subdirectories and files
    for dirpath, _, filenames in os.walk(root_folder):
        # Check if the current directory contains any frame files
        if any(filename.endswith('.png') for filename in filenames):
            # Process the frames in this directory and create a video
            process_all_folders(dirpath, fps=fps)


def run_inference_on_frames(base_input_dir, base_output_dir, model_name=transformation):
    """
    Runs inference on image frames using a specified model.

    Args:
        base_input_dir (str): Path to the input directory containing image frames.
        base_output_dir (str): Path to the output directory for inference results.
        model_name (str): Name of the model to use for inference.
    """
    # Dictionary to map pretrained names to URLs
    model_urls = {
        "day_to_night": "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl",
        "night_to_day": "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl",
        "clear_to_rainy": "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl",
        "rainy_to_clear": "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
    }

    # Ensure the model_name is valid
    if model_name not in model_urls:
        raise ValueError(f"Invalid model_name '{model_name}'. Valid options are: {list(model_urls.keys())}")

    # Define checkpoints directory
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Get the URL for the specified model_name
    url = model_urls[model_name]

    # Define the file path to save the model
    file_path = os.path.join(checkpoints_dir, f"{model_name}.pkl")

    # Download the file if it doesn't already exist
    if not os.path.exists(file_path):
        print(f"Downloading {model_name} model...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model '{model_name}' downloaded and saved to '{file_path}'.")
        else:
            raise Exception(f"Failed to download the model. HTTP Status Code: {response.status_code}")
    else:
        print(f"Model '{model_name}' already exists at '{file_path}'.")

    # Ensure the target folder exists in base_input_dir
    target_folder = os.path.join(base_input_dir, model_name)
    if not os.path.isdir(target_folder):
        logger.error(f"Folder '{model_name}' does not exist in the input directory '{base_input_dir}'")
        return

    # Traverse through all subdirectories and files in base_input_dir
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            # Check if the file is an image (e.g., ends with .png, .jpg, etc.)
            if file.endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(root, file)

                # Get the relative path of the file's directory with respect to base_input_dir
                relative_dir = os.path.relpath(root, base_input_dir)

                # Construct the corresponding output directory path
                output_dir = os.path.join(base_output_dir, relative_dir)

                # Create the output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Define the command with input image and output directory
                command = [
                    "python", "img2turbo/src/inference_unpaired.py",
                    "--model_name", model_name,
                    "--input_image", input_image_path,
                    "--output_dir", output_dir
                ]

                try:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               bufsize=1, universal_newlines=True)

                    # Stream stdout in real time
                    for stdout_line in iter(process.stdout.readline, ""):
                        sys.stdout.write(stdout_line)  # Write to stdout
                        sys.stdout.flush()  # Force flush to ensure real-time output

                    # Stream stderr in real time
                    for stderr_line in iter(process.stderr.readline, ""):
                        sys.stderr.write(stderr_line)  # Write to stderr
                        sys.stderr.flush()  # Force flush to ensure real-time output

                    process.stdout.close()
                    process.stderr.close()

                    return_code = process.wait()
                    if return_code:
                        logger.error(f"Error processing {input_image_path}. Command exited with {return_code}")

                except subprocess.CalledProcessError as e:
                    logger.error(f"Error occurred while processing {input_image_path}: {e}")
                    logger.error("Command Error Output:", e.stderr)


def download_file(url, output_dir):
    """
    Downloads a file from a given URL and saves it to the specified directory.

    Args:
        url (str): URL of the file to download.
        output_dir (str): Path to the directory where the file will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the file name from the URL
    file_name = os.path.join(output_dir, url.split("/")[-1])

    # Download the file
    logger.info(f"Downloading {file_name}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Write the content to the file
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        logger.info(f"Download completed: {file_name}")
    else:
        logger.error(f"Failed to download. Status code: {response.status_code}")


def run_realesrgan_inference(model_name, input_dir, output_base_dir, face_enhance=False):
    """
    Runs Real-ESRGAN inference on image files for super-resolution.

    Args:
        model_name (str): Name of the model to use for inference.
        input_dir (str): Path to the directory containing input images.
        output_base_dir (str): Path to the directory for saving output images.
        face_enhance (bool): Whether to enable face enhancement during inference.
    """

    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    output_dir = os.path.join("realesrgan_main", "weights")  # Folder to save the file
    download_file(url, output_dir)

    # Traverse through all subdirectories and files in input_dir
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is an image (e.g., ends with .png, .jpg, etc.)

            if file.endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(root, file)

                # Get the relative path of the file's directory with respect to input_dir
                relative_dir = os.path.relpath(root, input_dir)

                # Construct the corresponding output directory path
                output_dir = os.path.join(output_base_dir, relative_dir)

                # Create the output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Define the command with input image and output directory
                command = [
                    "python", "realesrgan_main/inference_realesrgan.py",
                    "-n", model_name,
                    "-i", input_image_path,
                    "-o", output_dir  # Define the output directory
                ]

                # Add the --face_enhance flag if face enhancement is enabled
                if face_enhance:
                    command.append("--face_enhance")

                try:
                    # Execute the command
                    result = subprocess.run(command, check=True, capture_output=True, text=True)

                    # Print the output from the command
                    logger.info(f"Processed {input_image_path} -> {os.path.join(output_dir, file)}")
                    logger.info("Command Output:", result.stdout)

                except subprocess.CalledProcessError as e:
                    logger.error(f"Error occurred while processing {input_image_path}: {e}")
                    logger.error("Command Error Output:", e.stderr)


if __name__ == "__main__":

    process_videos_in_directory(original_data)
    # process_videos_in_directory(img2_output_data)
    # process_videos_in_directory(final_data)

    run_inference_on_frames(original_data, img2_output_data)
    frames_to_video(img2_output_data, fps=30)

    run_realesrgan_inference('RealESRGAN_x4plus', img2_output_data, final_data, face_enhance=True)
    frames_to_video(final_data, fps=30)

    subprocess.run(["python", "evaluate.py"])
