import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description="移動平均を求める")
    parser.add_argument("csv", metavar="csv", type=str, help="input csv file")
    parser.add_argument("-o", metavar="output_filename_prefix", type=str, default="result", help="output file name prefix (default = result)")
    args = parser.parse_args()

    with open(args.csv) as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーを読み込む
        data = [list(map(float, row[:3])) + [row[3]] for row in reader]  # 最後のカラムは数値に変換しない

    n = 5
    moving_average_data = []
    for i in range(len(data)):
        row = data[i]
        if i < len(data) - n + 1:
            avg = [sum(data[j][k] for j in range(i, i + n))/n for k in range(3)]  # 最初の3カラムのみ移動平均を計算
            avg = [round(a, 5) for a in avg]
            moving_average_data.append(avg + [row[3]])  # `sec`カラムを追加
        else:
            moving_average_data.append(row)

    with open(args.o, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)  # ヘッダーを書き込む
        writer.writerows(moving_average_data)

if __name__ == "__main__":
    main()
