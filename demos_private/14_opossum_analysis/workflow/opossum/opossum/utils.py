import csv
import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

DATETIME_FORMT = "%Y-%m-%d %H:%M:%S"


def get_current_datetime() -> datetime:
    """
    現在の日時を取得する
    :return:
    """
    return datetime.now()


def get_datetime_from_filename(filename: str) -> datetime:
    """
    ファイル名をdatetimeに変換する
    :param filename:
        新カメラ（TS-WPTCAM2）:       20230916_121705.mp4
        旧カメラ（QWATCH TS-NS310W）: Record-2023-04-16_00-00-02.avi
    :param extension:
    :return:
    """
    date_str = filename.replace("Record-", "")
    date_str = date_str.replace(".mp4", "").replace(".avi", "")
    date_str = date_str.replace("-", "")
    return datetime.strptime(date_str, "%Y%m%d_%H%M%S")


def get_files_newer_than(
    file_paths: list[Path], datetime_threshold: datetime | None
) -> list[Path]:
    """
    datetime_thresholdより新しいファイルを返す
    :param file_paths:
    :param datetime_threshold:
    :return:
    """
    for file_path in file_paths:
        file_datetime = get_datetime_from_filename(file_path.name)
        if datetime_threshold is None or file_datetime > datetime_threshold:
            yield file_path


def get_last_target_datetime(last_run_file: Path) -> datetime | None:
    """
    ファイルから最後に解析を行ったファイルの撮影開始日時を取得する
    :param last_run_file:
    :return:
    """
    if os.path.exists(last_run_file):
        with open(last_run_file, "r") as f:
            last_run_datetime = datetime.strptime(f.read().strip(), DATETIME_FORMT)
    else:
        last_run_datetime = None
    return last_run_datetime


def update_last_target_datetime(last_run_file: Path, dt: datetime):
    """
    ファイルに最後に解析を行ったファイルの撮影開始日時を書き込む
    :param last_run_file:
    :param dt:
    :return:
    """
    if not last_run_file.parent.exists():
        last_run_file.parent.mkdir(parents=True)
    with open(last_run_file, "w") as f:
        f.write(dt.strftime(DATETIME_FORMT))


def save_to_tsv(file_path: Path, data: list[list[str]]):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)
    return


def get_duration(file_path: Path) -> float:
    """
    ファイルの長さ（秒数）を取得する
    :param file_path:
    :return:
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


def vfr2cfr(file_path: Path, frame_rate: int = 5):
    subprocess.run(["mkdir", "-p", "/tmp"])
    temp_output_path = Path("/tmp").joinpath(f"{file_path.stem}_temp_cfr.mp4")
    cmd_ffprobe = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    r_frame_rate = subprocess.check_output(cmd_ffprobe, text=True).strip()
    print(r_frame_rate)

    if r_frame_rate == "0/0":
        print(f"Converting VFR video '{file_path}' to CFR at {frame_rate}fps...")
        cmd_ffmpeg = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-vf",
            f"fps={frame_rate}",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            "-y",
            str(temp_output_path),
        ]
        subprocess.run(cmd_ffmpeg)
        temp_output_path.replace(file_path)
        print(
            f"Conversion completed and replaced the original with: {file_path}",
            file=sys.stderr,
        )
    else:
        print(f"The video '{file_path}' is already CFR.", file=sys.stderr)

    return file_path


def remove_white_noise(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ret_im = image.copy()
    denoised_image = cv2.morphologyEx(ret_im, cv2.MORPH_OPEN, kernel)
    return denoised_image


def draw_rectangle_on_image(image, rectangle_info, color=(0, 255, 0), thickness=2):
    start_point = (rectangle_info["x"], rectangle_info["y"])
    end_point = (
        rectangle_info["x"] + rectangle_info["width"],
        rectangle_info["y"] + rectangle_info["height"],
    )
    ret_im = image.copy()
    cv2.rectangle(ret_im, start_point, end_point, color, thickness)
    return ret_im


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def increase_contrast(image, alpha=1.5, beta=0):
    contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrasted_image


def generate_mask(frames, variance_threshold=1000):
    frames = np.array(frames)
    variance = np.var(frames, axis=0)
    mask = (variance <= variance_threshold).astype(np.uint8) * 255
    return mask


def convert_to_yellow_tint(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        colored_image = image.copy()
    hsv_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = 58
    yellow_tinted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return yellow_tinted_image


def overlay_mask_on_grayscale_image(image, mask):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_colored = image.copy()
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_blue = np.zeros_like(image_colored)
    mask_blue[:, :, 0] = mask  # 青色チャンネルにマスクをセット
    overlaid_image = cv2.addWeighted(image_colored, 0.5, mask_blue, 1, 0)
    return overlaid_image


def overlay_mask_on_image(image, mask, color=(0, 0, 255)):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_colored = image.copy()
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_colored = np.zeros_like(image_colored)
    for i in range(3):
        mask_colored[:, :, i] = mask * color[i] / 255
    overlaid_image = cv2.addWeighted(image_colored, 0.5, mask_colored, 1, 0)

    return overlaid_image


def overwrite_with_blue_tint(image, mask):
    # 白黒の画像をカラーに変換
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_colored = image.copy()

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # マスクが白の場所で元の画像の青色チャンネルを255に上書き
    image_colored[mask == 255, 0] = 255
    image_colored[mask == 255, 1] = 0
    image_colored[mask == 255, 2] = 0

    return image_colored


def white_pixel_ratio(image, threshold=200):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(binary_image == 255)
    total_pixels = image.size
    ratio = white_pixels / total_pixels
    return ratio


def binarize_image(image, threshold=90):
    # チャンネル数が1の場合、3チャンネルに変換
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)
    binary_colored = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    return binary_colored


def put_text_on_top_right(
    image, text, margin=10, font_scale=0.7, color=(0, 0, 255), thickness=2
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = image.shape[1] - text_size[0] - margin
    y = margin + text_size[1]
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    return image


def merge_adjacent_boxes(image, threshold=10):
    """
    画像内の近接するボックスを結合する関数
    Parameters:
    - image: 入力画像 (二値画像)
    - threshold: ボックスが近接しているとみなす距離 (px)

    Returns:
    - ボックスが結合された画像
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.uint8(image)

    # 輪郭を検出
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 各輪郭の外接矩形を取得
    rects = [cv2.boundingRect(contour) for contour in contours]

    # 近接するボックスを結合
    merged_rects = []
    merged_flags = [False] * len(rects)
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        for j, other_rect in enumerate(rects):
            if i == j:
                continue
            ox, oy, ow, oh = other_rect
            if abs(x - ox) <= threshold and abs(y - oy) <= threshold:
                merged_rects.append(
                    (
                        min(x, ox),
                        min(y, oy),
                        max(x + w, ox + ow) - min(x, ox),
                        max(y + h, oy + oh) - min(y, oy),
                    )
                )
                merged_flags[i] = True
                merged_flags[j] = True
                break

    # 結合されていないボックスを削除
    for i, flag in enumerate(merged_flags):
        if not flag:
            x, y, w, h = rects[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), 0, -1)

    # 結合されたボックスを新しい画像に描画
    output = np.zeros_like(image)
    for rect in merged_rects:
        x, y, w, h = rect
        cv2.rectangle(output, (x, y), (x + w, y + h), 255, -1)

    # 画像を3チャンネルのBGRカラーに変換
    output_colored = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    return output_colored


def fill_outside_rectangle(image, rectangle):
    output_image = image.copy()
    x = rectangle["x"]
    y = rectangle["y"]
    w = rectangle["width"]
    h = rectangle["height"]
    output_image[:y, :] = [0, 0, 0]  # 上
    output_image[y + h :, :] = [0, 0, 0]  # 下
    output_image[y : y + h, :x] = [0, 0, 0]  # 左
    output_image[y : y + h, x + w :] = [0, 0, 0]  # 右
    return output_image


def draw_grid(
    image,
    grid_size=10,
    color=(50, 150, 50),
    thickness=1,
    bold_interval=5,
    bold_thickness=2,
    alpha=0.5,
):
    output_image = image.copy()
    overlay = image.copy()
    height, width = image.shape[:2]

    for idx, x in enumerate(range(0, width, grid_size)):
        if idx % bold_interval == 0:
            cv2.line(overlay, (x, 0), (x, height), color, bold_thickness)
        else:
            cv2.line(overlay, (x, 0), (x, height), color, thickness)

    for idx, y in enumerate(range(0, height, grid_size)):
        if (idx) % bold_interval == 0:
            cv2.line(overlay, (0, y), (width, y), color, bold_thickness)
        else:
            cv2.line(overlay, (0, y), (width, y), color, thickness)

    # アルファブレンディングを適用
    cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

    return output_image


def add_yellow_border(image, thickness=5):
    output_image = image.copy()
    height, width = image.shape[:2]
    color = (0, 255, 255)  # BGRでの黄色
    cv2.rectangle(output_image, (0, 0), (width, height), color, thickness)
    return output_image


def calculate_mask_percentage_in_rectangle(mask_image, rectangle):
    """
    長方形の領域内のマスクされた割合を計算する関数
    Args:
        mask_image:
        rectangle:

    Returns:

    """
    x, y, w, h = rectangle["x"], rectangle["y"], rectangle["width"], rectangle["height"]
    roi = mask_image[y : y + h, x : x + w]
    masked_pixels = np.sum(roi[:, :, 0] == 255)  # 青チャンネルが255の画素数をカウント
    total_pixels = w * h
    percentage = (masked_pixels / total_pixels) * 100
    return percentage
    # return f"{round(percentage, 5)}%({masked_pixels} / {total_pixels})"


def calculate_mask_percentage_in_rectangle_str(mask_image, rectangle):
    x, y, w, h = rectangle["x"], rectangle["y"], rectangle["width"], rectangle["height"]
    roi = mask_image[y : y + h, x : x + w]
    masked_pixels = np.sum(roi[:, :, 0] == 255)  # 青チャンネルが255の画素数をカウント
    total_pixels = w * h
    percentage = (masked_pixels / total_pixels) * 100
    return f"{round(percentage, 5)}%({masked_pixels} / {total_pixels})"


def find_long_continuous_regions(data, l):
    """
    連続する非ゼロの領域を検出する関数
    Args:
        data: 1次元の配列 (e.g. [0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
            0は
        l:

    Returns:
    """
    regions = []
    start_idx = None
    count = 0

    for i in range(len(data)):
        if data[i] != 0:
            if start_idx is None:
                start_idx = i
            count += 1
        else:
            if count > l:
                regions.append((start_idx, i - 1))
            start_idx = None
            count = 0

    # 最後の継続フレーム数がlを超える場合、その領域をリストに追加
    if count > l:
        regions.append((start_idx, len(data) - 1))

    return regions


def seconds_to_hms(seconds: float) -> str:
    """
    秒数を時分秒の形式に変換する関数
    Args:
        seconds:

    Returns:

    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
