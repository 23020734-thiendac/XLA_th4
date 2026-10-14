"""Tạo sơ đồ vị trí các thành phần tần số trong ma trận DCT 8x8."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OUTPUT = Path(__file__).resolve().parent / "output_bai2_dct"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-2.1, 9.2)
    ax.set_ylim(9.1, -2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Khung và lưới ma trận DCT 8x8.
    ax.add_patch(Rectangle((0, 0), 8, 8, facecolor="white",
                           edgecolor="black", linewidth=2.2))
    for i in range(1, 8):
        ax.plot([i, i], [0, 8], color="#c8c8c8", linewidth=0.8)
        ax.plot([0, 8], [i, i], color="#c8c8c8", linewidth=0.8)

    # Hệ số DC và các vùng tần số.
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eeeeee",
                           edgecolor="black", linewidth=1.2))
    ax.text(0.5, 0.5, "DC", ha="center", va="center",
            fontsize=17, fontweight="bold", color="black")
    ax.text(1.35, 1.35, "TẦN SỐ THẤP", ha="left", va="center",
            fontsize=15, fontweight="bold", color="black",
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(5.9, 7.25, "TẦN SỐ CAO", ha="center", va="center",
            fontsize=15, fontweight="bold", color="black",
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

    # Mũi tên chỉ chiều tăng chỉ số và tần số.
    ax.annotate("", xy=(8, -0.65), xytext=(0, -0.65),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
    ax.text(4, -1.0, "v tăng  —  tần số ngang tăng",
            ha="center", va="bottom", fontsize=13, color="black")

    ax.annotate("", xy=(-0.7, 8), xytext=(-0.7, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
    ax.text(-1.35, 4, "u tăng  —  tần số dọc tăng",
            ha="center", va="center", rotation=90, fontsize=13, color="black")

    # Đường chéo thể hiện tần số tổng quát tăng dần.
    ax.annotate("", xy=(7.55, 7.55), xytext=(1.15, 1.15),
                arrowprops=dict(arrowstyle="->", color="black", lw=2,
                                linestyle="--"))
    ax.text(4.15, 3.75, "Tần số tăng dần",
            rotation=-45, ha="center", va="bottom", fontsize=11,
            color="black", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.text(4, -2.0,
            "VỊ TRÍ CÁC THÀNH PHẦN TẦN SỐ TRONG MA TRẬN DCT 8×8",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color="black")

    duong_dan = OUTPUT / "03_so_do_tan_so_dct_8x8.png"
    fig.savefig(duong_dan, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"Đã lưu sơ đồ tại: {duong_dan}")


if __name__ == "__main__":
    main()
