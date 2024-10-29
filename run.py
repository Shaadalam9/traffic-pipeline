import cv2
import os
import subprocess
import requests
from custom_logger import CustomLogger
from logmod import logs
import common
import sys

logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
data_folder = common.get_configs("data")
original_data = common.get_configs("data_original")
img2_output_data = common.get_configs("data_img2_output")
final_data = common.get_configs("data_final")


def video_to_frames(video_path):
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
    # Walk through all subdirectories and files
    print("## entering root folder")
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.mp4'):
                # Full path to the video file
                video_path = os.path.join(dirpath, file)

                # Process the video file, saving frames in the same directory
                video_to_frames(video_path)


def process_all_folders(frame_folder, fps=30):
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
    frame = cv2.imread(first_frame_path)  # type: ignore
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
    print(f"Video saved as {output_video_path}")


def frames_to_video(root_folder, fps=30):
    # Walk through all subdirectories and files
    for dirpath, _, filenames in os.walk(root_folder):
        # Check if the current directory contains any frame files
        if any(filename.endswith('.png') for filename in filenames):
            # Process the frames in this directory and create a video
            process_all_folders(dirpath, fps=fps)


def run_inference_on_frames(base_input_dir, base_output_dir, model_name="day_to_night"):
    # Traverse through all subdirectories and files in base_input_dir
    for root, dirs, files in os.walk(base_input_dir):
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
                    "python", "src/inference_unpaired.py",
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
                    "python", "inference_realesrgan.py",
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

    # process_videos_in_directory(data_folder)

    run_inference_on_frames(original_data, img2_output_data)

    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    output_dir = "weights"  # Folder to save the file
    download_file(url, output_dir)

    run_realesrgan_inference('RealESRGAN_x4plus', img2_output_data, final_data, face_enhance=True)

    # subprocess.run(["python", "evaluate.py"])

    # frames_to_video(img2_output_data, fps=30)

    # subprocess.run(["python", "analysis.py"])
