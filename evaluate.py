import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
from moviepy.editor import VideoFileClip
import tensorflow as tf
import common
from custom_logger import CustomLogger
from logmod import logs
import os

logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs("plotly_template")
data_folder = common.get_configs("data")
final_data = common.get_configs("data_final")
original_data = common.get_configs("data_original")
img2_output_data = common.get_configs("data_img2_output")


# T-SSIM (Temporal Structural Similarity Index)
def compute_t_ssim(video_path):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Error: Could not read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    ssim_values = []

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        ssim_value, _ = ssim(prev_gray, curr_gray, full=True)
        print(ssim_value)
        ssim_values.append(ssim_value)
        prev_gray = curr_gray

    cap.release()
    t_ssim_value = np.mean(ssim_values)
    logger.info(f"T-SSIM for the video at {video_path} is {t_ssim_value:.4f}")


# FVD (Frechet Video Distance)
def load_i3d_model():
    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(224, 224, 3))
    return model


def preprocess_video(video_path, target_size=(224, 224)):
    clip = VideoFileClip(video_path)
    frames = [cv2.resize(frame, target_size) for frame in clip.iter_frames()]
    frames = np.array(frames) / 255.0
    return frames


def extract_features(video_frames, model):
    features = model.predict(video_frames)
    return features


def calculate_fvd(real_features, generated_features):
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
    model = load_i3d_model()
    real_video_frames = preprocess_video(original_video_path)
    generated_video_frames = preprocess_video(generated_video_path)
    real_features = extract_features(real_video_frames, model)
    generated_features = extract_features(generated_video_frames, model)
    fvd_value = calculate_fvd(real_features, generated_features)
    logger.info(f"Frechet Video Distance (FVD) for {generated_video_path} is {fvd_value:.4f}")


# PSNR (Peak Signal-to-Noise Ratio)

# Calculate Mean Squared Error (MSE) between two frames
def calculate_mse(frame1, frame2):
    mse = np.mean((frame1 - frame2) ** 2)
    return mse


# Calculate PSNR between two frames
def calculate_psnr(frame1, frame2):
    mse = calculate_mse(frame1, frame2)
    if mse == 0:
        return float('inf')  # Infinite PSNR if frames are identical
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


# Calculate PSNR for an entire video
def calculate_video_psnr(original_video_path, processed_video_path):
    original_cap = cv2.VideoCapture(original_video_path)
    processed_cap = cv2.VideoCapture(processed_video_path)

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
    original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    processed_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret1, original_frame = original_cap.read()
        ret2, processed_frame = processed_cap.read()
        if not ret1 or not ret2:
            break

        # Resize processed frame to match the original frame size
        processed_frame_resized = cv2.resize(processed_frame, target_size)

        psnr_value = calculate_psnr(original_frame, processed_frame_resized)
        psnr_values.append(psnr_value)

    original_cap.release()
    processed_cap.release()

    if psnr_values:
        average_psnr = np.mean(psnr_values)
        logger.info(f"Average PSNR for the video {processed_video_path} is {average_psnr:.4f} dB")
    else:
        logger.info(f"No PSNR values calculated for the video {processed_video_path}.")


# VPQ (Video Perceptual Quality)

# Calculate SSIM between two frames with a specified win_size
def calculate_ssim(frame1, frame2):
    # Determine the smaller dimension of the frames
    min_dim = min(frame1.shape[0], frame1.shape[1])

    # Set win_size to be the minimum of 7 or the smallest dimension, ensuring it's odd
    win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)

    # If the smallest dimension is less than the minimum win_size, set win_size to that dimension
    if min_dim < win_size:
        win_size = min_dim if min_dim % 2 != 0 else min_dim - 1

    # Calculate SSIM with channel_axis set to the correct axis for multichannel images
    return ssim(frame1, frame2, win_size=win_size, channel_axis=-1)


def calculate_vpq(original_video_path, processed_video_path):
    original_cap = cv2.VideoCapture(original_video_path)
    processed_cap = cv2.VideoCapture(processed_video_path)

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
    original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    processed_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret1, original_frame = original_cap.read()
        ret2, processed_frame = processed_cap.read()
        if not ret1 or not ret2:
            break

        # Resize processed frame to match the original frame size
        processed_frame_resized = cv2.resize(processed_frame, target_size)

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


# Example usage
if __name__ == "__main__":
    logger.info("Analysis started")

    # Process the videos
    for directory in [final_data, img2_output_data]:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    compute_t_ssim(video_path)

    print("\n")

    # Iterate through videos in the generated directory for FVD
    for root, _, files in os.walk(img2_output_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, img2_output_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Check if the corresponding original video exists
                if os.path.exists(original_video_path):
                    calculate_video_psnr(original_video_path, processed_video_path)
                else:
                    print(f"Original video not found for: {processed_video_path}")
    print("\n")

    # Iterate through videos in the generated directory for FVD
    for root, _, files in os.walk(final_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, final_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Check if the corresponding original video exists
                if os.path.exists(original_video_path):
                    compute_fvd(original_video_path, processed_video_path)
                else:
                    print(f"Original video not found for: {processed_video_path}")
    print("\n")

    # Iterate through videos in the generated directory
    for root, _, files in os.walk(final_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct paths for the original and generated video files
                relative_path = os.path.relpath(root, final_data)
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Check if the corresponding original video exists
                if os.path.exists(original_video_path):
                    compute_fvd(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    print("\n")

    # Iterate through videos in the generated directory
    for root, _, files in os.walk(img2_output_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, img2_output_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Check if the corresponding original video exists
                if os.path.exists(original_video_path):
                    # Calculate PSNR
                    calculate_video_psnr(original_video_path, processed_video_path)
                else:
                    print(f"Original video not found for: {processed_video_path}")

    # Iterate through videos in the final_data directory
    for root, _, files in os.walk(final_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, final_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Check if the corresponding original video exists
                if os.path.exists(original_video_path):
                    # Calculate PSNR
                    calculate_video_psnr(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    print("\n")

    # Iterate through videos in the final_data directory
    for root, _, files in os.walk(final_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, final_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Check if the corresponding original video exists
                if os.path.exists(original_video_path):
                    # Calculate VPQ
                    calculate_vpq(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")

    print("\n")

    # Iterate through videos in the img2_output_data directory again
    for root, _, files in os.walk(img2_output_data):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the relative path of the file
                relative_path = os.path.relpath(root, img2_output_data)

                # Construct paths for the original and generated video files
                original_video_path = os.path.join(original_data, relative_path, file)
                processed_video_path = os.path.join(root, file)

                # Check if the corresponding original video exists
                if os.path.exists(original_video_path):
                    # Calculate VPQ
                    calculate_vpq(original_video_path, processed_video_path)
                else:
                    logger.error(f"Original video not found for: {processed_video_path}")
