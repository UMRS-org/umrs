import argparse
import cv2
from pathlib import Path


def calculate_dimensions(width, height, original_width, original_height):
    if width is None and height is None:
        return original_width, original_height
    elif width is None:
        ratio = original_width / original_height
        width = int(height * ratio)
    elif height is None:
        ratio = original_height / original_width
        height = int(width * ratio)
    return width, height


def concat(input_files: list[Path], output_file: Path, fps: int == 15, rate: int == 4):
    """
    複数の入力動画を1つの動画に連結する
    :param input_files:
    :param output_file:
    :param fps:  fps of output video
    :param rate: resize rate of output video. ex) 2. NOTE: 2 means half size, 4 means quarter size
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    for input_file in input_files:
        if not input_file.exists():
            raise FileNotFoundError(f"{input_file} does not exist")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        cap = cv2.VideoCapture(str(input_file))
        print(f"reading {input_file}")
        if not cap.isOpened():
            print(f"failed to open {input_file}")
            continue

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = round(input_fps / fps)
        frame_count = 0

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width, height = int(original_width / rate), int(original_height / rate)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (width, height))

                if out is None:
                    out = cv2.VideoWriter(
                        str(output_file), fourcc, fps, (width, height)
                    )

                out.write(frame)

            frame_count += 1

        cap.release()
        print(f"finished proceccing {input_file}")

    if out is not None:
        out.release()

    print(f"finish writing to {output_file}")
