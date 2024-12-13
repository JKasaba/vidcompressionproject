import numpy as np
from scipy.ndimage import gaussian_filter
from dct import perform_dct
from Step1_code import compute_motion_vector_tss, classify_macroblocks_r, read_frame

def write_cmp_header_binary(outfile, width, height, padded_width, padded_height, n1, n2):
    """Writes the header containing video dimensions and quantization parameters to the .cmp file in binary."""
    header = np.array([width, height, padded_width, padded_height], dtype=np.uint32)
    quant_params = np.array([n1, n2], dtype=np.uint8)
    outfile.write(header.tobytes())
    outfile.write(quant_params.tobytes())

def write_macroblock_binary(outfile, block_type, coefficients):
    """Writes the macroblock type and quantized coefficients to the .cmp file in binary."""
    outfile.write(np.uint8(block_type).tobytes())  # Write block type as a single byte
    outfile.write(coefficients.astype(np.int16).tobytes())  # Write coefficients as 16-bit integers

def quantize_block(block, quant_param):
    """Applies uniform quantization to a block of DCT coefficients."""
    quantized = np.round(block / (2 ** quant_param))
    return quantized

def pad_frame(frame, target_height, target_width):
    """Pads the frame to the target dimensions."""
    padded_frame = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    padded_frame[:frame.shape[0], :frame.shape[1], :] = frame
    return padded_frame

def denoise_motion_vectors(motion_vectors):
    """Denoises motion vectors using Gaussian filtering."""
    return gaussian_filter(motion_vectors, sigma=1)

def process_video_binary(input_file, output_file, width, height, n1, n2):
    """Processes the video file and writes the compressed .cmp file using binary encoding."""
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8

    with open(input_file, 'rb') as file, open(output_file, 'wb') as outfile:
        # Write the binary header
        write_cmp_header_binary(outfile, width, height, padded_width, padded_height, n1, n2)

        prev_frame = read_frame(file, width, height)
        if prev_frame is None:
            print("No frames to process. Exiting.")
            return
        prev_frame = pad_frame(prev_frame, padded_height, padded_width)

        while True:
            curr_frame = read_frame(file, width, height)
            if curr_frame is None:
                break
            curr_frame = pad_frame(curr_frame, padded_height, padded_width)

            # Compute and denoise motion vectors
            motion_vectors = denoise_motion_vectors(
                compute_motion_vector_tss(curr_frame, prev_frame, block_size=16)
            )

            # Classify macroblocks
            background, foreground = classify_macroblocks_r(motion_vectors, threshold=0.1, max_foreground_percentage=4)

            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    # DCT and quantization
                    block_r = perform_dct(curr_frame[i:i+8, j:j+8, 0])
                    block_g = perform_dct(curr_frame[i:i+8, j:j+8, 1])
                    block_b = perform_dct(curr_frame[i:i+8, j:j+8, 2])

                    macroblock_index = (i // 16, j // 16)
                    block_type = 1 if macroblock_index in foreground else 0
                    quant_param = n1 if block_type == 1 else n2

                    coefficients = np.stack([
                        quantize_block(block_r, quant_param),
                        quantize_block(block_g, quant_param),
                        quantize_block(block_b, quant_param),
                    ])

                    write_macroblock_binary(outfile, block_type, coefficients)

            prev_frame = curr_frame

if __name__ == "__main__":
    input_video_path = '3.rgb'
    output_compressed_path = 'compressed_output.cmp'
    video_width = 960
    video_height = 540
    quantization_param_foreground = 0
    quantization_param_background = 7

    process_video_binary(input_video_path, output_compressed_path, video_width, video_height,
                         quantization_param_foreground, quantization_param_background)
