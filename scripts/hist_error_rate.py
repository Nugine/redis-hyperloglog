import json
import matplotlib.pyplot as plt


# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# 提取每条数据的第三个值
def extract_third_values(data):
    third_values = [item[2] for item in data]
    return third_values


# 绘制直方图
def plot_histogram(values, bin_width):
    plt.figure(figsize=(10, 6))
    bins = int((max(values) - min(values)) / bin_width)
    plt.hist(values, bins=bins, alpha=0.7)
    plt.xlabel("Error Rate")
    plt.ylabel("Frequency")
    plt.title("Histogram of Error Rate")
    plt.grid(True)
    plt.show()


# 主函数
def main():
    file_path = "results.json"  # 替换为你的 JSON 文件路径
    data = read_json(file_path)
    third_values = extract_third_values(data)
    plot_histogram(third_values, bin_width=0.0001)


if __name__ == "__main__":
    main()
