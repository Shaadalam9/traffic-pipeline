import torch
import torchvision.transforms as T
import torchvision.models as models
from scipy.linalg import sqrtm
from PIL import Image
import timm
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
from moviepy.editor import VideoFileClip
import tensorflow as tf
import common
from custom_logger import CustomLogger
from logmod import logs
import os

# Initialize custom logger with color-coded output
logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)

# Load configurations from a common utility
template = common.get_configs("plotly_template")
base_data_folder = common.get_configs("data")

# Define paths for data organization
final_data = os.path.join(base_data_folder, "final")
original_data = os.path.join(base_data_folder, "original")
img2_output_data = os.path.join(base_data_folder, "img2turbo_output")
compare_folders = os.path.join(base_data_folder, "compare")


# --------------------------
# T-SSIM Metric Calculation
# --------------------------
def compute_t_ssim(video_path):
    """
    Compute the Temporal Structural Similarity Index (T-SSIM) for a video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        None: Logs T-SSIM and its variance for the video.
    """
    cap = cv2.VideoCapture(video_path)  # Open video file # type: ignore
    if not cap.isOpened():
        logger.error("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Error: Could not read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # type: ignore
    ssim_values = []

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)  # type: ignore
        ssim_value, _ = ssim(prev_gray, curr_gray, full=True)
        ssim_values.append(ssim_value)
        prev_gray = curr_gray

    cap.release()
    t_ssim_value = np.mean(ssim_values)
    t_ssim_variance = np.var(ssim_values)
    logger.info(f"T-SSIM for the video at {video_path} is {t_ssim_value:.4f} and variance is {t_ssim_variance:.4f}")


# ---------------------------
# Frechet Video Distance (FVD)
# ---------------------------
def load_i3d_model():
    """
    Load the pre-trained Inception V3 model for feature extraction.

    Returns:
        Model: TensorFlow Inception V3 model.
    """
    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(224, 224, 3))
    return model


def preprocess_video(video_path, target_size=(224, 224)):
    """
    Preprocess a video to extract frames resized to a target size.

    Args:
        video_path (str): Path to the video file.
        target_size (tuple): Desired frame size (width, height).

    Returns:
        np.ndarray: Array of processed video frames.
    """
    clip = VideoFileClip(video_path)
    frames = [cv2.resize(frame, target_size) for frame in clip.iter_frames()]
    frames = np.array(frames) / 255.0
    return frames


def extract_features(video_frames, model):
    """
    Extract features for video frames using the specified model.

    Args:
        video_frames (np.ndarray): Array of video frames.
        model: Feature extraction model.

    Returns:
        np.ndarray: Extracted feature vectors.
    """
    features = model.predict(video_frames)
    return features


def calculate_fvd(real_features, generated_features):
    """
    Calculate the Frechet Video Distance (FVD).

    Args:
        real_features (np.ndarray): Features from real video.
        generated_features (np.ndarray): Features from generated video.

    Returns:
        float: FVD value.
    """
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)
    diff = mu_real - mu_generated
    covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = np.sum(diff**2) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fvd


def compute_fvd(original_video_path, generated_video_path):
    """
    Compute the Frechet Video Distance (FVD) between two videos.

    Args:
        original_video_path (str): Path to the original video.
        generated_video_path (str): Path to the generated video.

    Returns:
        None: Logs the FVD value.
    """
    model = load_i3d_model()
    real_video_frames = preprocess_video(original_video_path)
    generated_video_frames = preprocess_video(generated_video_path)
    real_features = extract_features(real_video_frames, model)
    generated_features = extract_features(generated_video_frames, model)
    fvd_value = calculate_fvd(real_features, generated_features)
    logger.info(f"Frechet Video Distance (FVD) for {generated_video_path} is {fvd_value:.4f}")


# ---------------------------
# Peak Signal-to-Noise Ratio (PSNR)
# ---------------------------
def calculate_mse(frame1, frame2):
    """
    Calculate the Mean Squared Error (MSE) between two frames.

    Args:
        frame1 (np.ndarray): First frame.
        frame2 (np.ndarray): Second frame.

    Returns:
        float: MSE value.
    """
    mse = np.mean((frame1 - frame2) ** 2)
    return mse


def calculate_psnr(frame1, frame2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two frames.

    Args:
        frame1 (np.ndarray): First frame.
        frame2 (np.ndarray): Second frame.

    Returns:
        float: PSNR value in dB.
    """
    mse = calculate_mse(frame1, frame2)
    if mse == 0:
        return float('inf')  # Infinite PSNR if frames are identical
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


# Calculate PSNR for an entire video
def calculate_video_psnr(original_video_path, processed_video_path):
    """
    Calculate the average PSNR for an entire video by comparing frames.

    Args:
        original_video_path (str): Path to the original video.
        processed_video_path (str): Path to the processed video.

    Returns:
        None: Logs the average PSNR value for the video.
    """
    original_cap = cv2.VideoCapture(original_video_path)  # type: ignore
    processed_cap = cv2.VideoCapture(processed_video_path)  # type: ignore

    if not original_cap.isOpened() or not processed_cap.isOpened():
        logger.info("Error: Could not open video files.")
        return

    psnr_values = []

    # Get the resolution of the original video
    ret, original_frame = original_cap.read()
    if not ret:
        logger.error("Error: Could not read frames from the original video.")
        return

    target_size = (original_frame.shape[1], original_frame.shape[0])  # (width, height)

    # Reset the video captures to the beginning
    original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore
    processed_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore

    while True:
        ret1, original_frame = original_cap.read()
        ret2, processed_frame = processed_cap.read()
        if not ret1 or not ret2:
            break

        # Resize processed frame to match the original frame size
        processed_frame_resized = cv2.resize(processed_frame, target_size)  # type: ignore

        psnr_value = calculate_psnr(original_frame, processed_frame_resized)
        psnr_values.append(psnr_value)

    original_cap.release()
    processed_cap.release()

    if psnr_values:
        average_psnr = np.mean(psnr_values)
        logger.info(f"Average PSNR for the video {processed_video_path} is {average_psnr:.4f} dB")
    else:
        logger.info(f"No PSNR values calculated for the video {processed_video_path}.")


# ---------------------------
# Video Perceptual Quality (VPQ)
# ---------------------------
def calculate_ssim(frame1, frame2):
    """
    Calculate the Structural Similarity Index (SSIM) between two frames.

    Args:
        frame1 (np.ndarray): First frame.
        frame2 (np.ndarray): Second frame.

    Returns:
        float: SSIM value.
    """
    min_dim = min(frame1.shape[0], frame1.shape[1])

    # Set win_size to be the minimum of 7 or the smallest dimension, ensuring it's odd
    win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)

    # If the smallest dimension is less than the minimum win_size, set win_size to that dimension
    if min_dim < win_size:
        win_size = min_dim if min_dim % 2 != 0 else min_dim - 1

    # Calculate SSIM with channel_axis set to the correct axis for multichannel images
    return ssim(frame1, frame2, win_size=win_size, channel_axis=-1)


def calculate_vpq(original_video_path, processed_video_path):
    """
    Compute Video Perceptual Quality (VPQ) as a combination of PSNR and SSIM.

    Args:
        original_video_path (str): Path to the original video.
        processed_video_path (str): Path to the processed video.

    Returns:
        None: Logs the VPQ score.
    """
    original_cap = cv2.VideoCapture(original_video_path)  # type: ignore
    processed_cap = cv2.VideoCapture(processed_video_path)  # type: ignore

    if not original_cap.isOpened() or not processed_cap.isOpened():
        logger.error("Error: Could not open video files.")
        return

    ssim_values = []
    psnr_values = []

    # Get the resolution of the original video
    ret, original_frame = original_cap.read()
    if not ret:
        logger.error("Error: Could not read frames from the original video.")
        return

    target_size = (original_frame.shape[1], original_frame.shape[0])  # (width, height)

    # Reset the video captures to the beginning
    original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore
    processed_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore

    while True:
        ret1, original_frame = original_cap.read()
        ret2, processed_frame = processed_cap.read()
        if not ret1 or not ret2:
            break

        # Resize processed frame to match the original frame size
        processed_frame_resized = cv2.resize(processed_frame, target_size)  # type: ignore

        # Calculate SSIM
        ssim_value = calculate_ssim(original_frame, processed_frame_resized)
        ssim_values.append(ssim_value)

        # Calculate PSNR
        psnr_value = calculate_psnr(original_frame, processed_frame_resized)
        psnr_values.append(psnr_value)

    original_cap.release()
    processed_cap.release()

    if ssim_values and psnr_values:
        average_ssim = np.mean(ssim_values)
        average_psnr = np.mean(psnr_values)
        vpq_score = (average_ssim + average_psnr) / 2
        logger.info(f"Video Perceptual Quality (VPQ) for video {processed_video_path} is {vpq_score:.4f}")
    else:
        logger.info(f"No VPQ values calculated for the video {processed_video_path}.")


# ---------------------------
# DINO-Based Image Comparison
# ---------------------------
def load_dino_model():
    """
    Load the Vision Transformer (ViT) model pre-trained with DINO.

    Returns:
        Model: Pre-trained DINO model.
    """
    model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
    model.eval()
    return model


# Preprocess the image for DINO
def preprocess_image(image_path, image_size=224):
    """
    Preprocess an image for the DINO model.

    Args:
        image_path (str): Path to the image file.
        image_size (int): Desired size for resizing.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # type: ignore # Add batch dimension


# Extract features from the model
def extract_features_DINO(model, image_tensor):
    """
    Extract features from an image using the DINO model.

    Args:
        model: Pre-trained DINO model.
        image_tensor (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Flattened feature vector.
    """
    with torch.no_grad():
        features = model.forward_features(image_tensor)  # Use forward_features to get embeddings
    return features.flatten(start_dim=1)  # Flatten feature map for comparison


# Compute cosine similarity
def compute_similarity(features1, features2):
    """
    Compute cosine similarity between two feature vectors.

    Args:
        features1 (torch.Tensor): Feature vector 1.
        features2 (torch.Tensor): Feature vector 2.

    Returns:
        float: Cosine similarity score.
    """
    features1 = features1 / features1.norm(dim=1, keepdim=True)  # Normalize
    features2 = features2 / features2.norm(dim=1, keepdim=True)
    similarity = (features1 @ features2.T).squeeze().item()  # Dot product
    return similarity


def filter_image_files(file_list, folder_path):
    """
    Filter valid image files from a list of filenames.

    Args:
        file_list (list): List of filenames.
        folder_path (str): Path to the folder containing the files.

    Returns:
        list: List of valid image filenames.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return [file for file in file_list if file.lower().endswith(tuple(valid_extensions)) and os.path.isfile(os.path.join(folder_path, file))]


def match_files_by_basename(folder_a_files, folder_b_files):
    """
    Match files between two folders based on their basenames.

    Args:
        folder_a_files (list): List of filenames in folder A.
        folder_b_files (list): List of filenames in folder B.

    Returns:
        list of tuple: Pairs of matching filenames.
    """
    files_a_basenames = {os.path.splitext(file)[0]: file for file in folder_a_files}
    files_b_basenames = {os.path.splitext(file)[0]: file for file in folder_b_files}
    common_basenames = set(files_a_basenames.keys()).intersection(files_b_basenames.keys())
    return [(files_a_basenames[base], files_b_basenames[base]) for base in common_basenames]


# Compare all pairs in two folders
def compare_two_folders(folder_a, folder_b):
    """
    Compare all matching image files in two folders using DINO-based similarity.

    Args:
        folder_a (str): Path to folder A.
        folder_b (str): Path to folder B.

    Returns:
        list of tuple: Results with matching filenames and their similarity distances.
    """
    model = load_dino_model()
    results = []

    # Get list of files in both folders
    files_a = filter_image_files(os.listdir(folder_a), folder_a)
    files_b = filter_image_files(os.listdir(folder_b), folder_b)

    # Ensure both folders have matching files
    common_files = match_files_by_basename(files_a, files_b)

    for file_a, file_b in common_files:
        path_a = os.path.join(folder_a, file_a)
        path_b = os.path.join(folder_b, file_b)

        # Preprocess images
        image_a = preprocess_image(path_a)
        image_b = preprocess_image(path_b)

        # Extract features
        features_a = extract_features_DINO(model, image_a)
        features_b = extract_features_DINO(model, image_b)

        # Compute distance
        similarity = compute_similarity(features_a, features_b)
        distance = 1 - similarity  # Convert similarity to distance (1 means identical)

        results.append((f"{file_a} -> {file_b}", distance))

    return results


def compare_folders_with_others(parent_dir, folder_a_name):
    """
    Compare a specified folder with all other folders in the parent directory.

    Args:
        parent_dir (str): Path to the parent directory.
        folder_a_name (str): Name of the folder to compare.

    Returns:
        dict: Comparison results grouped by folder names.
    """
    folder_a = os.path.join(parent_dir, folder_a_name)
    other_folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir,
                                                                                     f)) and f != folder_a_name]

    all_results = {}

    for folder_b_name in other_folders:
        folder_b = os.path.join(parent_dir, folder_b_name)
        results = compare_two_folders(folder_a, folder_b)
        all_results[folder_b_name] = results

    return all_results


# ---------------------------
# Main Execution Block
# ---------------------------
if __name__ == "__main__":
    logger.info("Analysis started")

    # Process all MP4 videos in the final_data and img2_output_data directories to compute T-SSIM
    for directory in [final_data, img2_output_data]:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    compute_t_ssim(video_path)

    print("\n")

    # Calculate FVD (Frechet Video Distance) for videos in the final_data directory
    for root, _, files in os.walk(final_data):
        for file in files:
            if file.endswith('.mp4'):
                # Define paths for original and generated videos
                relative_path = os.path.relpath(root, final_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # If the original video exists, compute FVD
                if os.path.exists(original_video_path):
                    compute_fvd(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")
    print("\n")

    # Compute FVD for videos in the img2_output_data directory
    for root, _, files in os.walk(img2_output_data):
        for file in files:
            if file.endswith('.mp4'):
                # Define paths for original and generated videos
                relative_path = os.path.relpath(root, img2_output_data)
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # If the original video exists, compute FVD
                if os.path.exists(original_video_path):
                    compute_fvd(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    print("\n")

    # Calculate PSNR for videos in final_data
    for root, _, files in os.walk(img2_output_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, img2_output_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Compute PSNR if original video is available
                if os.path.exists(original_video_path):
                    # Calculate PSNR
                    calculate_video_psnr(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    # Calculate PSNR for videos in final_data
    for root, _, files in os.walk(final_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, final_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Compute PSNR if original video is available
                if os.path.exists(original_video_path):
                    # Calculate PSNR
                    calculate_video_psnr(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    print("\n")

    # Calculate VPQ for videos in img2_output_data
    for root, _, files in os.walk(final_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, final_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Compute VPQ if original video is available
                if os.path.exists(original_video_path):
                    # Calculate VPQ
                    calculate_vpq(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    print("\n")

    # Calculate VPQ for videos in img2_output_data
    for root, _, files in os.walk(img2_output_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, img2_output_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Compute VPQ if original video is available
                if os.path.exists(original_video_path):
                    # Calculate VPQ
                    calculate_vpq(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    # Perform DINO-based comparisons for a specified folder (Original) against others

    all_comparisons = compare_folders_with_others(compare_folders, "Original")

    # Log the results of the comparisons
    for folder_b_name, results in all_comparisons.items():
        logger.info(f"\nComparing Original with {folder_b_name}:")
        for file_name, distance in results:
            logger.info(f"File: {file_name}, DINO-Struct-Distance: {distance}")
