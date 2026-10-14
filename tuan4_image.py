"""Chạy đủ các yêu cầu DFT/IDFT trên ảnh week4_image.png."""

from pathlib import Path
import sys

from bai1_dft_idft import HIEN_THI, plt, xu_ly


THU_MUC = Path(__file__).resolve().parent
ANH_WEEK4 = THU_MUC / "week4_image.png"
OUTPUT = THU_MUC / "output_bai1_week4_image"
KICH_THUOC = (32, 64)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not ANH_WEEK4.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {ANH_WEEK4}")

    OUTPUT.mkdir(exist_ok=True)
    ket_qua = []
    for k in KICH_THUOC:
        mse = xu_ly(k, ANH_WEEK4, OUTPUT, "week4_image")
        ket_qua.append(f"{k}x{k}: MSE = {mse:.12e}")

    (OUTPUT / "ket_qua_mse.txt").write_text(
        "\n".join(ket_qua), encoding="utf-8"
    )
    if HIEN_THI:
        plt.show()


if __name__ == "__main__":
    main()
