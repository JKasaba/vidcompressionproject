import cv2
import numpy as np
from cv2 import Canny
from scipy.ndimage import median_filter

def denoise_motion_vectors(motion_vectors):
    return median_filter(motion_vectors, size=(3, 3, 1))


width, height, fps = 960, 540, 30
filename = "3.rgb"
block_size = 16
search_range = 4
threshold = .1

def read_frame(file, width, height):
    frame_data = file.read(width * height * 3)
    if not frame_data:
        return None
    return np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))

def smooth_motion_vectors(motion_vectors, prev_motion_vectors, alpha=0.8):
    return alpha * motion_vectors + (1 - alpha) * prev_motion_vectors

def filter_uniform_regions(frame, low_threshold=50, high_threshold=150):
    edges = Canny(frame, low_threshold, high_threshold)
    return edges > 0  # Mask of non-uniform regionss


def compute_motion_vector_tss(curr_frame, prev_frame, block_size=16, search_range=4):
    h, w, _ = curr_frame.shape
    curr_frame_r = curr_frame[:, :, 0]
    prev_frame_r = prev_frame[:, :, 0]
    uniform_mask = filter_uniform_regions(curr_frame_r)
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size      
    motion_vectors = np.zeros((num_blocks_h, num_blocks_w, 2))

    for i in range(0, num_blocks_h * block_size, block_size):
        for j in range(0, num_blocks_w * block_size, block_size):
            block_mask = uniform_mask[i:i + block_size, j:j + block_size]
            if not block_mask.any():  # Skip uniform blocks
                continue
            curr_block = curr_frame_r[i:i + block_size, j:j + block_size]
            center_x, center_y = i, j
            step_size = search_range
            min_cost = float('inf')
            best_vector = (0, 0)

            while step_size >= 1:
                for x in range(-step_size, step_size + 1, step_size):
                    for y in range(-step_size, step_size + 1, step_size):
                        ref_i, ref_j = center_x + x, center_y + y
                        if 0 <= ref_i <= h - block_size and 0 <= ref_j <= w - block_size:
                            ref_block = prev_frame_r[ref_i:ref_i + block_size, ref_j:ref_j + block_size]
                            cost = np.sum(np.abs(curr_block - ref_block))

                            if cost < min_cost:
                                min_cost = cost
                                best_vector = (x, y)

                center_x += best_vector[0]
                center_y += best_vector[1]
                step_size //= 2

            motion_vectors[i // block_size, j // block_size] = best_vector

    return motion_vectors

def classify_macroblocks_r(motion_vectors, threshold=2, max_foreground_percentage=4):
    background = []
    foreground = []
    total_blocks = motion_vectors.shape[0] * motion_vectors.shape[1]
    max_foreground_blocks = int((max_foreground_percentage / 100) * total_blocks)

    for i in range(motion_vectors.shape[0]):
        for j in range(motion_vectors.shape[1]):
            motion_vector = motion_vectors[i, j]
            magnitude = np.linalg.norm(motion_vector)
            if magnitude < threshold:
                background.append((i, j))
            else:
                foreground.append((i, j))

    # Adjust if the number of foreground blocks exceeds the limit
    if len(foreground) > max_foreground_blocks:
        # Sort by motion vector magnitude and move excess to background
        sorted_foreground = sorted(foreground, key=lambda idx: np.linalg.norm(motion_vectors[idx[0], idx[1]]))
        excess_foreground = sorted_foreground[max_foreground_blocks:]
        foreground = sorted_foreground[:max_foreground_blocks]
        background.extend(excess_foreground)

    return background, foreground


def visualize_segmentation(frame, background, foreground, block_size=16):
    vis_frame = frame.copy()
    for (i, j) in foreground:
        y, x = i * block_size, j * block_size
        vis_frame[y:y + block_size, x:x + block_size] = [255, 0, 0]
    return vis_frame

if __name__ == "__main__":
    prev_motion_vectors = None  # Initialize a variable to store previous motion vectors

    with open(filename, "rb") as file:
        prev_frame = read_frame(file, width, height)  # I-frame
        background, foreground = [], []
        frame_count = 0

        while True:
            curr_frame = read_frame(file, width, height)
            if curr_frame is None:
                break

            motion_vectors = compute_motion_vector_tss(curr_frame, prev_frame, block_size, search_range)
            motion_vectors = denoise_motion_vectors(motion_vectors)




            background, foreground = classify_macroblocks_r(motion_vectors, threshold)

            segmented_frame = visualize_segmentation(curr_frame, background, foreground, block_size)

            cv2.imshow("Segmented Video", cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR))
            prev_frame = curr_frame

            if cv2.waitKey(int(1000 / fps)) == 27:  # Escape key to exit
                break

            frame_count += 1

    cv2.destroyAllWindows()

