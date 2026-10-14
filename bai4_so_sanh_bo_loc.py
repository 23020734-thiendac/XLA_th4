"""Bài 4: so sánh Ideal, Gaussian và Butterworth trong miền tần số."""

from pathlib import Path
import sys

import matplotlib

HIEN_THI = "--hien-thi" in sys.argv
if not HIEN_THI:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Kế thừa các hàm tự xây dựng từ Bài 1 và Bài 3.
from bai1_dft_idft import dft_2d, idft_2d, dich_tam_pho
from bai3_bo_loc_tan_so import tinh_khoang_cach, bo_dich_tam_pho


THU_MUC = Path(__file__).resolve().parent
ANH_GOC = THU_MUC / "week4_image.png"
OUTPUT = THU_MUC / "output_bai4_so_sanh_bo_loc"
KICH_THUOC = 64
CAC_D0 = (10, 30, 50, 80)
CAC_BAC_N = (1, 2, 4)
HE_SO_LAM_SAC = 1.0


def ideal_low_pass(D, D0):
    """Bộ lọc thông thấp lý tưởng."""
    return (D <= D0).astype(float)


def gaussian_low_pass(D, D0):
    """Gaussian LPF: H = exp(-D^2 / (2*D0^2))."""
    return np.exp(-(D ** 2) / (2 * D0 ** 2))


def butterworth_low_pass(D, D0, n):
    """Butterworth LPF bậc n: H = 1 / (1 + (D/D0)^(2n))."""
    return 1.0 / (1.0 + (D / D0) ** (2 * n))


def loc_anh(pho_dich_tam, H):
    """Nhân mask với phổ, dịch ngược rồi dùng IDFT để thu ảnh lọc."""
    pho_sau_loc = H * pho_dich_tam
    anh_sau_loc = idft_2d(bo_dich_tam_pho(pho_sau_loc)).real
    return pho_sau_loc, anh_sau_loc


def tinh_mse_psnr(anh_goc, anh_loc):
    """Tính MSE và PSNR giữa ảnh gốc với ảnh sau lọc."""
    mse = np.mean((anh_goc - anh_loc) ** 2)
    psnr = np.inf if mse < 1e-15 else 10 * np.log10(255 ** 2 / mse)
    return mse, psnr


def hien_thi_thong_cao(ax, anh_hp, tieu_de):
    """Hiển thị ảnh thông cao theo thang đối xứng quanh giá trị 0."""
    gioi_han = max(np.max(np.abs(anh_hp)), 1e-12)
    ax.imshow(anh_hp, cmap="gray", vmin=-gioi_han, vmax=gioi_han)
    ax.set_title(tieu_de)
    ax.axis("off")


def luu_hinh(fig, ten_file):
    fig.savefig(OUTPUT / ten_file, dpi=160,
                bbox_inches="tight", pad_inches=0.15)
    if not HIEN_THI:
        plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not ANH_GOC.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {ANH_GOC}")
    OUTPUT.mkdir(exist_ok=True)

    # Đọc cùng một ảnh xám đầu vào như Bài 3.
    with Image.open(ANH_GOC) as im:
        im = im.convert("L").resize(
            (KICH_THUOC, KICH_THUOC), Image.Resampling.LANCZOS
        )
    anh = np.asarray(im, dtype=float)

    # Chỉ tính DFT một lần rồi dùng chung cho tất cả bộ lọc.
    pho = dft_2d(anh)
    pho_dich_tam = dich_tam_pho(pho)
    pho_log = np.log1p(np.abs(pho_dich_tam))
    D = tinh_khoang_cach(*anh.shape)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(anh, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Ảnh xám 64×64")
    ax[1].imshow(pho_log, cmap="gray")
    ax[1].set_title("Phổ Fourier đã dịch tâm")
    for truc in ax:
        truc.axis("off")
    fig.tight_layout()
    luu_hinh(fig, "01_anh_goc_va_pho.png")

    # Lưu kết quả bằng khóa (D0, tên bộ lọc) để tái sử dụng khi vẽ hình.
    ket_qua = {}

    for D0 in CAC_D0:
        cac_mask = {
            "Ideal": ideal_low_pass(D, D0),
            "Gaussian": gaussian_low_pass(D, D0),
            "Butterworth n=1": butterworth_low_pass(D, D0, 1),
            "Butterworth n=2": butterworth_low_pass(D, D0, 2),
            "Butterworth n=4": butterworth_low_pass(D, D0, 4),
        }

        for ten, H in cac_mask.items():
            pho_sau_loc, anh_lp = loc_anh(pho_dich_tam, H)
            mse, psnr = tinh_mse_psnr(anh, anh_lp)

            # H_HP = 1-H_LP. Theo tính tuyến tính của IDFT:
            # IDFT(H_HP*F) = ảnh gốc - IDFT(H_LP*F).
            H_hp = 1.0 - H
            pho_hp = H_hp * pho_dich_tam
            anh_hp = anh - anh_lp
            anh_sac = np.clip(anh + HE_SO_LAM_SAC * anh_hp, 0, 255)

            ket_qua[(D0, ten)] = {
                "H_lp": H,
                "H_hp": H_hp,
                "pho_lp": pho_sau_loc,
                "pho_hp": pho_hp,
                "anh_lp": anh_lp,
                "anh_hp": anh_hp,
                "anh_sac": anh_sac,
                "mse": mse,
                "psnr": psnr,
            }

    # Ý 5 và 7: cùng D0, so sánh mask, phổ và ảnh của ba bộ lọc.
    ten_so_sanh = ("Ideal", "Gaussian", "Butterworth n=2")
    for stt, D0 in enumerate(CAC_D0, start=2):
        fig, ax = plt.subplots(3, 3, figsize=(11, 10))
        for hang, ten in enumerate(ten_so_sanh):
            kq = ket_qua[(D0, ten)]
            ax[hang, 0].imshow(kq["H_lp"], cmap="gray", vmin=0, vmax=1)
            ax[hang, 0].set_title(f"{ten}\nMask thông thấp")
            ax[hang, 1].imshow(np.log1p(np.abs(kq["pho_lp"])), cmap="gray")
            ax[hang, 1].set_title("Phổ sau lọc")
            ax[hang, 2].imshow(kq["anh_lp"], cmap="gray", vmin=0, vmax=255)
            ax[hang, 2].set_title(
                f"Ảnh kết quả\nMSE={kq['mse']:.3f}, "
                f"PSNR={kq['psnr']:.2f} dB"
            )
        for truc in ax.ravel():
            truc.axis("off")
        fig.suptitle(f"So sánh ba bộ lọc thông thấp — D₀={D0}",
                     fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        luu_hinh(fig, f"0{stt}_so_sanh_LPF_D0_{D0}.png")

    # Ý 4: ảnh hưởng của bậc n trong Butterworth ở mọi D0.
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for hang, n in enumerate(CAC_BAC_N):
        for cot, D0 in enumerate(CAC_D0):
            kq = ket_qua[(D0, f"Butterworth n={n}")]
            ax[hang, cot].imshow(kq["H_lp"], cmap="gray", vmin=0, vmax=1)
            ax[hang, cot].set_title(f"n={n}, D₀={D0}")
            ax[hang, cot].axis("off")
    fig.suptitle("Mask Butterworth khi thay đổi bậc n và bán kính D₀",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    luu_hinh(fig, "06_mask_butterworth_n_D0.png")

    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for hang, n in enumerate(CAC_BAC_N):
        for cot, D0 in enumerate(CAC_D0):
            kq = ket_qua[(D0, f"Butterworth n={n}")]
            ax[hang, cot].imshow(kq["anh_lp"], cmap="gray", vmin=0, vmax=255)
            ax[hang, cot].set_title(
                f"n={n}, D₀={D0}\nMSE={kq['mse']:.2f}"
            )
            ax[hang, cot].axis("off")
    fig.suptitle("Ảnh Butterworth khi thay đổi bậc n và bán kính D₀",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    luu_hinh(fig, "07_anh_butterworth_n_D0.png")

    # Ý 6: bộ lọc thông cao tương ứng và ảnh làm sắc nét, chọn D0=30.
    D0_lam_sac = 30
    fig, ax = plt.subplots(3, 3, figsize=(11, 10))
    for hang, ten in enumerate(ten_so_sanh):
        kq = ket_qua[(D0_lam_sac, ten)]
        ax[hang, 0].imshow(kq["H_hp"], cmap="gray", vmin=0, vmax=1)
        ax[hang, 0].set_title(f"{ten}\nMask thông cao")
        hien_thi_thong_cao(
            ax[hang, 1], kq["anh_hp"], "Ảnh thông cao"
        )
        ax[hang, 2].imshow(kq["anh_sac"], cmap="gray", vmin=0, vmax=255)
        ax[hang, 2].set_title("Ảnh làm sắc nét")
        ax[hang, 2].axis("off")
    fig.suptitle("Bộ lọc thông cao và làm sắc nét — D₀=30",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    luu_hinh(fig, "08_thong_cao_va_lam_sac_D0_30.png")

    # So sánh ảnh làm sắc nét của ba bộ lọc tại mọi D0.
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for hang, ten in enumerate(ten_so_sanh):
        for cot, D0 in enumerate(CAC_D0):
            kq = ket_qua[(D0, ten)]
            ax[hang, cot].imshow(kq["anh_sac"], cmap="gray", vmin=0, vmax=255)
            ax[hang, cot].set_title(f"{ten}, D₀={D0}")
            ax[hang, cot].axis("off")
    fig.suptitle("So sánh ảnh làm sắc nét từ các bộ lọc thông cao",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    luu_hinh(fig, "09_so_sanh_anh_lam_sac.png")

    # Lưu bảng số liệu để dùng trong báo cáo.
    dong_ket_qua = ["D0;Bo loc;MSE;PSNR(dB)"]
    for D0 in CAC_D0:
        for ten in ("Ideal", "Gaussian", "Butterworth n=1",
                    "Butterworth n=2", "Butterworth n=4"):
            kq = ket_qua[(D0, ten)]
            dong = f"{D0};{ten};{kq['mse']:.6f};{kq['psnr']:.6f}"
            dong_ket_qua.append(dong)
            print(dong)
    (OUTPUT / "ket_qua_so_sanh.txt").write_text(
        "\n".join(dong_ket_qua), encoding="utf-8"
    )

    if HIEN_THI:
        plt.show()
    print(f"Kết quả được lưu tại: {OUTPUT}")


if __name__ == "__main__":
    main()
