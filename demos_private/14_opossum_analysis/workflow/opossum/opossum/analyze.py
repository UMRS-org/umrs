import argparse
import csv
import cv2
from pathlib import Path
import pprint
import sys

from .utils import (
    binarize_image,
    remove_white_noise,
    merge_adjacent_boxes,
    calculate_mask_percentage_in_rectangle,
    overlay_mask_on_grayscale_image,
    add_yellow_border,
    concat_tile,
    draw_grid,
    fill_outside_rectangle,
    find_long_continuous_regions,
    seconds_to_hms,
    vfr2cfr,
)

pp = pprint.PrettyPrinter(indent=4)


def main():
    parser = argparse.ArgumentParser(description="オポッサム動画の解析")
    parser.add_argument("input1", metavar="VIDEO1", type=str, help="input movie file 1")
    parser.add_argument("input2", metavar="VIDEO2", type=str, help="input movie file 2")
    parser.add_argument(
        "-o", metavar="output_filename_prefix", type=str, help="output file name prefix"
    )
    args = parser.parse_args()
    input_path_1 = Path(args.input1)
    input_path_2 = Path(args.input2)
    pp.pprint(args)
    for i in get_duration(input_path_1, input_path_2, output_filename=args.o):
        print(",".join(i))


def get_duration(
    input_filename_1: Path,
    input_filename_2: Path,
    output_filename: str = None,
    rate_csv_file_path: Path = None,
    distance_threshold: float = 0.05,
    time_threshold: float = 15,
):
    """

    Args:
        input_filename_1:
        input_filename_2:
        output_filename:
        rate_csv_file_path:
        distance_threshold: 0.0 ~ 1.0 で指定 (動きのある矩形を併合する際の距離の閾値、解像度のwidthに対する比率で指定)
            1920 * 1080 に対して0.05なら 96ピクセル 以内の矩形が併合される
            640 * 360 に対して0.05なら 32ピクセル 以内の矩形が併合される
        time_threshold: 動きのあるフレームを連続とみなす時間の閾値 (秒)

    Returns:

    """
    vfr2cfr(input_filename_1)
    vfr2cfr(input_filename_2)
    cap_file_1 = cv2.VideoCapture(str(input_filename_1))
    cap_file_2 = cv2.VideoCapture(str(input_filename_2))
    frame_rate = cap_file_1.get(cv2.CAP_PROP_FPS)
    height = int(cap_file_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap_file_1.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(
        f"input file name: {input_filename_1}\topen: {cap_file_1.isOpened()}",
        file=sys.stderr,
    )
    print(
        f"CAP_PROP_FRAME_WIDTH  : {cap_file_1.get(cv2.CAP_PROP_FRAME_WIDTH)}",
        file=sys.stderr,
    )
    print(
        f"CAP_PROP_FRAME_HEIGHT : {cap_file_1.get(cv2.CAP_PROP_FRAME_HEIGHT)}",
        file=sys.stderr,
    )
    print(
        f"CAP_PROP_FPS          : {cap_file_1.get(cv2.CAP_PROP_FPS)}", file=sys.stderr
    )
    print(
        f"CAP_PROP_FRAME_COUNT  : {cap_file_1.get(cv2.CAP_PROP_FRAME_COUNT)}",
        file=sys.stderr,
    )
    print(
        f"CAP_PROP_POS_FRAMES   : {cap_file_1.get(cv2.CAP_PROP_POS_FRAMES)}",
        file=sys.stderr,
    )
    print(
        f"CAP_PROP_POS_MSEC     : {cap_file_1.get(cv2.CAP_PROP_POS_MSEC)}",
        file=sys.stderr,
    )

    if output_filename is not None:
        output_video_size = (width * 2, height * 2)
        fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # ファイル形式(ここではmp4)
        writer = cv2.VideoWriter(
            output_filename + "_movie.mp4", fmt, frame_rate, output_video_size
        )  # ライター作成

    previous_frame = None
    masked_pixel_rate = [[], []]
    print("Proccecing", file=sys.stderr)
    for cap_file in (cap_file_1, cap_file_2):
        ret, frame = cap_file.read()
        if previous_frame is None:
            previous_frame = frame.copy()
            ret, frame = cap_file.read()

        print(width * distance_threshold)
        while ret:
            sec = round(cap_file.get(cv2.CAP_PROP_POS_FRAMES) / frame_rate)
            """
            secがthretholdを超えていてかつcap_fileがcap_file_2のとき、breakする。
            """
            if sec > time_threshold and cap_file == cap_file_2:
                break

            binarized = binarize_image(frame)
            bdiff = cv2.absdiff(binarized, binarize_image(previous_frame))

            without_noize = remove_white_noise(bdiff, 5)
            merged = merge_adjacent_boxes(
                without_noize, threshold=width * distance_threshold
            )
            rate_0 = calculate_mask_percentage_in_rectangle(
                merged, {"x": 0, "y": 0, "width": width, "height": height}
            )
            if output_filename is not None:
                masked = overlay_mask_on_grayscale_image(frame, merged)
                fill_0 = add_yellow_border(
                    draw_grid(
                        fill_outside_rectangle(
                            masked, {"x": 0, "y": 0, "width": width, "height": height}
                        )
                    )
                )
                im = concat_tile(((frame, bdiff), (without_noize, fill_0)))
                writer.write(im)

            previous_frame = frame.copy()
            masked_pixel_rate[0].append(round(rate_0, 4))
            masked_pixel_rate[1].append(sec)

            ret, frame = cap_file.read()

    print("\nDone", file=sys.stderr)
    if output_filename is not None:
        writer.release()
    cap_file_1.release()
    cap_file_2.release()
    ret_array = [["start_time", "end_time"]]
    for each_region in find_long_continuous_regions(
        masked_pixel_rate[0], frame_rate * time_threshold
    ):
        print(each_region, file=sys.stderr)
        start_frame, end_frame = each_region
        start_sec = start_frame // frame_rate
        end_sec = end_frame // frame_rate
        start_time = seconds_to_hms(start_sec)
        end_time = seconds_to_hms(end_sec)
        ret_array.append([start_time, end_time])

    if rate_csv_file_path is not None:
        transposed_data = list(zip(*masked_pixel_rate))
        with open(rate_csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["rect0", "sec"])
            writer.writerows(transposed_data)

    return ret_array


if __name__ == "__main__":
    main()
