import csv
import argparse

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, data

def count_non_zero_regions(data, l):
    count = 0
    in_region = False

    for i in range(len(data)):
        if data[i] != 0:
            if not in_region:
                count += 1
                in_region = True
        else:
            if in_region and any(data[j] != 0 for j in range(i+1, min(i+1+l, len(data)))):
                continue
            in_region = False
    return count


def find_non_zero_regions(data, l):
    regions = []
    start_idx = None

    for i in range(len(data)):
        if data[i] != 0:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                # lの距離内に次の非ゼロの領域があるか確認
                next_non_zero_idx = next((j for j in range(i+1, min(i+1+l, len(data))) if data[j] != 0), None)
                if next_non_zero_idx:
                    i = next_non_zero_idx - 1  # インデックスを次の非ゼロの領域の前に移動
                    continue
                regions.append((start_idx, i-1))
                start_idx = None

    # 最後の非ゼロの領域をリストに追加
    if start_idx is not None:
        regions.append((start_idx, len(data)-1))

    return regions

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def main():
    parser = argparse.ArgumentParser(description="Calculate the proportion of non-zero regions")
    parser.add_argument("csv", metavar="csv", type=str, help="Input CSV file")
    parser.add_argument("-l", metavar="distance", type=int, default=5, help="Distance to connect non-zero regions (default = 5)")
    args = parser.parse_args()

    header, csv_data = read_csv(args.csv)

    for col_idx, col_name in enumerate(header[:-1]):  # Exclude 'sec' column
        column_data = [float(row[col_idx]) for row in csv_data]
        non_zero_count = count_non_zero_regions(column_data, args.l)
        #print(f"{col_name}: {non_zero_count} non-zero regions in {len(column_data)} frames({round(non_zero_count/len(column_data)*100, 3)}%)")
        #print(find_non_zero_regions(column_data, args.l))
        total_frames = 0
        for each_region in find_non_zero_regions(column_data, args.l):
            start_frame, end_frame = each_region
            start_frame    = start_frame // 5
            end_frame      = end_frame // 5
            start_time_str = seconds_to_hms(start_frame)
            end_time_str   = seconds_to_hms(end_frame)
            print(f"{col_name}, {start_time_str}, {end_time_str}")
if __name__ == "__main__":
    main()
