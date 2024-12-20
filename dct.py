import numpy as np
from PIL import Image

from scipy.fft import dct, idct

def perform_dct(block):
    return np.round(dct(dct(block.T, norm='ortho').T, norm='ortho'))

def perform_idct(block):
    return np.round(idct(idct(block.T, norm='ortho').T, norm='ortho'))

def read_image_rgb(file_path, width, height):

    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(data) != width * height * 3:
        raise ValueError("The file size does not match the specified dimensions.")

    # Separate the channels
    r = data[:width * height].reshape((height, width))
    g = data[width * height:2 * width * height].reshape((height, width))
    b = data[2 * width * height:].reshape((height, width))

    return r, g, b

def save_image(r, g, b, output_path):
    image = np.stack((r, g, b), axis=-1).astype(np.uint8)
    img = Image.fromarray(image, 'RGB')
    img.save(output_path)

def zigzag_order():
    #    i hard coded this
    order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    return order

# def perform_dct(block):
#     """
#     Performs Discrete Cosine Transform (DCT) on an 8x8 block.
#     """
#     return np.round(np.fft.dct(np.fft.dct(block.T, norm='ortho').T, norm='ortho'))

# def perform_idct(block):
#     """
#     Performs Inverse Discrete Cosine Transform (IDCT) on an 8x8 block.
#     """
#     return np.round(np.fft.idct(np.fft.idct(block.T, norm='ortho').T, norm='ortho'))

def apply_zigzag_and_zero(block, m):
    order = zigzag_order()
    flattened = np.zeros_like(block)
    for idx, (i, j) in enumerate(order):
        if idx < m:
            flattened[i, j] = block[i, j]
    return flattened

def process_image_dct(r, g, b, n):
    """
    Applies DCT and IDCT to 8x8 blocks of R, G, and B channels with zigzag coefficient selection.
    """
    height, width = r.shape
    m = round(n / (height * width / 64))  # Number of coefficients per block

    def process_channel(channel):
        processed_channel = np.zeros_like(channel)
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = channel[i:i+8, j:j+8]
                dct_block = perform_dct(block)
                dct_block = apply_zigzag_and_zero(dct_block, m)
                idct_block = perform_idct(dct_block)
                processed_channel[i:i+8, j:j+8] = idct_block
        return processed_channel

    r_processed = process_channel(r)
    g_processed = process_channel(g)
    b_processed = process_channel(b)

    return r_processed, g_processed, b_processed

# Main execution example
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 process_rgb.py <input_file> <n_coefficients>")
        sys.exit(1)

    input_file = sys.argv[1]
    n = int(sys.argv[2])

    # Dimensions of the input image
    width, height = 512, 512

    # Read the image
    r_channel, g_channel, b_channel = read_image_rgb(input_file, width, height)

    # Process the image with DCT and IDCT
    r_processed, g_processed, b_processed = process_image_dct(r_channel, g_channel, b_channel, n)

    # Save the processed image
    output_file = "output.png"
    save_image(r_processed, g_processed, b_processed, output_file)
    print(f"Processed image saved as {output_file}")
