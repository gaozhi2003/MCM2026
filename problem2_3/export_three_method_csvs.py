"""
问题二（3）：单独导出三种方法的排名表 CSV

输出三个 CSV 文件：
1. percentage_method_rankings.csv  - 百分比结合法
2. ranking_method_rankings.csv    - 排名结合法
3. new_method_rankings.csv        - 新方法（末尾两位+评委最低）

运行此脚本会执行完整数据流水线（加载、清洗、特征、模型、合成），
在生成上述三个 CSV 后即退出，不执行后续评估等步骤。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 默认导出到 data/，便于 three_method_comparison.py 等脚本读取
OUTPUT_DIR = PROJECT_ROOT / "data"


def main():
    from main import main as run_main
    print("=" * 60)
    print("导出三种方法排名表")
    print("=" * 60)
    run_main(export_only_three_csvs=True, output_dir=str(OUTPUT_DIR))
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("  - percentage_method_rankings.csv")
    print("  - ranking_method_rankings.csv")
    print("  - new_method_rankings.csv")


if __name__ == "__main__":
    main()
