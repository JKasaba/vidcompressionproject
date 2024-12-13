import numpy as np
import cv2
import wave
import pyaudio
from dct import perform_idct
import time


def preprocess_frames_binary(cmp_file):
    """Preprocess frames from the binary .cmp file and store them in a list."""
    frames = []

    with open(cmp_file, 'rb') as file:
        # Read header information
        header = np.frombuffer(file.read(16), dtype=np.uint32)
        quant_params = np.frombuffer(file.read(2), dtype=np.uint8)
        width, height, padded_width, padded_height = header
        n1, n2 = quant_params
        print(f"Header: width={width}, height={height}, "
              f"padded_width={padded_width}, padded_height={padded_height}, "
              f"n1={n1}, n2={n2}")

        block_size = 8

        while True:
            try:
                frame = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)

                for i in range(0, padded_height, block_size):
                    for j in range(0, padded_width, block_size):
                        # Read block type and coefficients
                        block_type_data = file.read(1)
                        if not block_type_data:
                            raise EOFError

                        block_type = np.frombuffer(block_type_data, dtype=np.uint8)[0]

                        coefficients_data = file.read(3 * block_size * block_size * 2)
                        if len(coefficients_data) < 3 * block_size * block_size * 2:
                            raise EOFError
                        coefficients = np.frombuffer(coefficients_data, dtype=np.int16).reshape(3, block_size, block_size)

                        # Determine quantization parameter
                        quant_param = n1 if block_type == 1 else n2

                        # Dequantize and perform IDCT
                        r_block = perform_idct(coefficients[0] * (2 ** quant_param))
                        g_block = perform_idct(coefficients[1] * (2 ** quant_param))
                        b_block = perform_idct(coefficients[2] * (2 ** quant_param))

                        # Place the block into the frame
                        frame[i:i+block_size, j:j+block_size, 0] = r_block.clip(0, 255)
                        frame[i:i+block_size, j:j+block_size, 1] = g_block.clip(0, 255)
                        frame[i:i+block_size, j:j+block_size, 2] = b_block.clip(0, 255)

                # Crop to original dimensions and add to frame list
                frames.append(frame[:height, :width, :])
            except EOFError:
                break

    return frames


def preprocess_audio(audio_file, fps, num_frames):
    """Preprocess the audio to align with video playback."""
    with wave.open(audio_file, 'rb') as wav:
        sample_rate = wav.getframerate()
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        # Calculate the desired audio duration
        video_duration = num_frames / fps
        total_audio_frames = int(video_duration * sample_rate)

        # Read audio data
        audio_data = wav.readframes(total_audio_frames)

    return audio_data, sample_rate, num_channels, sample_width


def play_audio_and_video(frames, fps, audio_file):
    """Synchronize audio and video playback with pause and step controls."""
    frame_duration = 1.0 / fps  # Frame duration in seconds

    # Preprocess audio
    audio_data, sample_rate, num_channels, sample_width = preprocess_audio(audio_file, fps, len(frames))

    # Initialize audio
    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = pyaudio_instance.open(format=pyaudio_instance.get_format_from_width(sample_width),
                                         channels=num_channels,
                                         rate=sample_rate,
                                         output=True)

    chunk_size = int(sample_rate * frame_duration) * num_channels * sample_width
    audio_idx = 0

    cv2.namedWindow("CMP Player", cv2.WINDOW_NORMAL)
    total_frames = len(frames)
    start_time = time.time()

    paused = False  # Playback state

    for frame_idx, frame in enumerate(frames):
        if paused:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('p'):  # Resume
                paused = False
                start_time = time.time() - frame_idx * frame_duration
                continue
            elif key == ord('s'):  # Step forward
                frame_bgr = cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR)
                cv2.imshow("CMP Player", frame_bgr)
                continue

        # Play audio in sync
        if audio_idx < len(audio_data):
            audio_chunk = audio_data[audio_idx:audio_idx + chunk_size]
            audio_stream.write(audio_chunk)
            audio_idx += chunk_size

        # Show video frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("CMP Player", frame_bgr)

        # Maintain synchronization
        elapsed_time = time.time() - start_time
        target_time = frame_idx * frame_duration
        if elapsed_time < target_time:
            time.sleep(target_time - elapsed_time)

        # Handle user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause
            paused = True

    # Cleanup
    audio_stream.stop_stream()
    audio_stream.close()
    pyaudio_instance.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cmp_file = "compressed_output.cmp"  # Binary .cmp file
    audio_file = "3.wav"  # Audio file
    fps = 30

    print("Preprocessing frames...")
    frames = preprocess_frames_binary(cmp_file)
    print(f"Total frames preprocessed: {len(frames)}")

    # Play audio and video together
    play_audio_and_video(frames, fps, audio_file)
