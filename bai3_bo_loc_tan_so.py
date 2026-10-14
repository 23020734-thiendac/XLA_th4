"""Bài 3: xây dựng bộ lọc thông thấp và thông cao lý tưởng trong miền tần số."""

from pathlib import Path
import sys

import matplotlib

HIEN_THI = "--hien-thi" in sys.argv
if not HIEN_THI:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ý 1: dùng lại đúng các hàm tự xây dựng ở Bài 1.
from bai1_dft_idft import dft_2d, idft_2d, dich_tam_pho


THU_MUC = Path(__file__).resolve().parent
ANH_GOC = THU_MUC / "week4_image.png"
OUTPUT = THU_MUC / "output_bai3_bo_loc_tan_so"
KICH_THUOC = 64
CAC_D0 = (10, 30, 50)


def tinh_khoang_cach(m, n):
    """Ý 2: tính D(u,v) từ từng điểm đến tâm phổ (M/2, N/2)."""
    D = np.zeros((m, n), dtype=float)
    for u in range(m):
        for v in range(n):
            D[u, v] = np.sqrt((u - m / 2) ** 2 + (v - n / 2) ** 2)
    return D


def ideal_low_pass(D, D0):
    """Ý 3: ILPF bằng 1 trong đường tròn bán kính D0, ngoài ra bằng 0."""
    return (D <= D0).astype(float)


def ideal_high_pass(D, D0):
    """Ý 6: IHPF là phần bù của ILPF."""
    return 1.0 - ideal_low_pass(D, D0)


def bo_dich_tam_pho(pho_dich_tam):
    """Dịch phổ từ giữa ảnh về vị trí ban đầu, không dùng ifftshift()."""
    m, n = pho_dich_tam.shape
    ket_qua = np.empty_like(pho_dich_tam)
    for u in range(m):
        for v in range(n):
            ket_qua[(u - m // 2) % m, (v - n // 2) % n] = pho_dich_tam[u, v]
    return ket_qua


def luu_anh_xam(duong_dan, anh):
    """Lưu ảnh có miền giá trị mức xám 0..255."""
    anh_uint8 = np.clip(np.rint(anh), 0, 255).astype(np.uint8)
    Image.fromarray(anh_uint8).save(duong_dan)


def anh_thong_cao_de_hien_thi(anh):
    """Đưa giá trị 0 về xám giữa; giá trị âm/dương thành tối/sáng."""
    gioi_han = np.max(np.abs(anh))
    if gioi_han < 1e-12:
        return np.full(anh.shape, 127, dtype=np.uint8)
    ket_qua = 127.5 + 127.5 * anh / gioi_han
    return np.clip(np.rint(ket_qua), 0, 255).astype(np.uint8)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not ANH_GOC.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {ANH_GOC}")
    OUTPUT.mkdir(exist_ok=True)

    # Ý 1: đọc ảnh, chuyển ảnh xám và đổi kích thước thành 64x64.
    with Image.open(ANH_GOC) as im:
        im = im.convert("L").resize(
            (KICH_THUOC, KICH_THUOC), Image.Resampling.LANCZOS
        )
    anh = np.asarray(im, dtype=float)
    luu_anh_xam(OUTPUT / "anh_xam_64x64.png", anh)

    # DFT chỉ tính một lần; mọi bộ lọc đều dùng chung phổ này.
    pho = dft_2d(anh)

    # Output ý 1: lưu trực tiếp ma trận phổ phức F(u,v) kích thước 64x64.
    # Mỗi phần tử có dạng (phần_thực + phần_ảo*j).
    np.savetxt(OUTPUT / "ma_tran_pho_phuc_Fuv.txt", pho, fmt="%.6e")

    pho_dich_tam = dich_tam_pho(pho)
    pho_log = np.log1p(np.abs(pho_dich_tam))
    D = tinh_khoang_cach(*anh.shape)

    # Hình ảnh gốc và phổ Fourier đã dịch tâm.
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(anh, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Ảnh xám 64×64")
    ax[1].imshow(pho_log, cmap="gray")
    ax[1].set_title("Phổ Fourier đã dịch tâm\nlog(1 + |F(u,v)|)")
    for truc in ax:
        truc.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_anh_goc_va_pho.png", dpi=160,
                bbox_inches="tight", pad_inches=0.15)

    ket_qua = []

    # Ý 4, 5, 6: thử D0 = 10, 30, 50 và lọc trong miền tần số.
    for D0 in CAC_D0:
        H_lp = ideal_low_pass(D, D0)
        H_hp = ideal_high_pass(D, D0)

        G_lp_tam = H_lp * pho_dich_tam
        G_hp_tam = H_hp * pho_dich_tam

        # Dịch phổ về vị trí cũ trước khi dùng hàm IDFT của Bài 1.
        anh_lp = idft_2d(bo_dich_tam_pho(G_lp_tam)).real
        anh_hp = idft_2d(bo_dich_tam_pho(G_hp_tam)).real

        mse_lp = np.mean((anh - anh_lp) ** 2)
        nang_luong_hp = np.mean(anh_hp ** 2)
        ket_qua.append((D0, H_lp, H_hp, anh_lp, anh_hp,
                        mse_lp, nang_luong_hp))

        luu_anh_xam(OUTPUT / f"anh_thong_thap_D0_{D0}.png", anh_lp)
        Image.fromarray(anh_thong_cao_de_hien_thi(anh_hp)).save(
            OUTPUT / f"anh_thong_cao_D0_{D0}.png"
        )

        # Ý 7: hiển thị đủ ảnh gốc, phổ, hai mask và hai ảnh sau lọc.
        fig, ax = plt.subplots(2, 3, figsize=(11, 7.5))
        noi_dung = [
            (anh, "Ảnh gốc", "gray", 0, 255),
            (pho_log, "Phổ Fourier đã dịch tâm", "gray", None, None),
            (H_lp, f"Mask thông thấp, D₀ = {D0}", "gray", 0, 1),
            (anh_lp, f"Ảnh lọc thông thấp\nMSE = {mse_lp:.3f}",
             "gray", 0, 255),
            (H_hp, f"Mask thông cao, D₀ = {D0}", "gray", 0, 1),
            (anh_hp, f"Ảnh lọc thông cao\nNăng lượng = {nang_luong_hp:.3f}",
             "gray", -max(np.max(np.abs(anh_hp)), 1e-12),
             max(np.max(np.abs(anh_hp)), 1e-12)),
        ]
        for truc, (du_lieu, ten, cmap, vmin, vmax) in zip(ax.ravel(), noi_dung):
            truc.imshow(du_lieu, cmap=cmap, vmin=vmin, vmax=vmax)
            truc.set_title(ten)
            truc.axis("off")
        fig.suptitle(f"Bộ lọc thông thấp và thông cao lý tưởng — D₀ = {D0}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUTPUT / f"0{D0 // 20 + 2}_ket_qua_D0_{D0}.png",
                    dpi=160, bbox_inches="tight", pad_inches=0.15)

    # Hình tổng hợp giúp so sánh khi D0 thay đổi.
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    ax[0, 0].imshow(anh, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Ảnh gốc")
    ax[1, 0].imshow(pho_log, cmap="gray")
    ax[1, 0].set_title("Phổ Fourier")

    for cot, (D0, _, _, anh_lp, anh_hp, mse_lp, nang_luong_hp) in enumerate(
            ket_qua, start=1):
        ax[0, cot].imshow(anh_lp, cmap="gray", vmin=0, vmax=255)
        ax[0, cot].set_title(f"Thông thấp D₀={D0}\nMSE={mse_lp:.3f}")
        gioi_han = max(np.max(np.abs(anh_hp)), 1e-12)
        ax[1, cot].imshow(anh_hp, cmap="gray",
                          vmin=-gioi_han, vmax=gioi_han)
        ax[1, cot].set_title(
            f"Thông cao D₀={D0}\nNăng lượng={nang_luong_hp:.3f}"
        )

    for truc in ax.ravel():
        truc.axis("off")
    fig.suptitle("So sánh kết quả khi thay đổi bán kính cắt D₀",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT / "05_so_sanh_cac_D0.png", dpi=160,
                bbox_inches="tight", pad_inches=0.15)

    dong_ket_qua = []
    for D0, H_lp, _, _, _, mse_lp, nang_luong_hp in ket_qua:
        so_he_so_giu = int(np.sum(H_lp))
        dong = (f"D0={D0}: ILPF giữ {so_he_so_giu}/4096 hệ số, "
                f"MSE thông thấp={mse_lp:.6f}, "
                f"năng lượng thông cao={nang_luong_hp:.6f}")
        dong_ket_qua.append(dong)
        print(dong)
    (OUTPUT / "ket_qua_bai3.txt").write_text(
        "\n".join(dong_ket_qua), encoding="utf-8"
    )

    if HIEN_THI:
        plt.show()
    else:
        plt.close("all")
    print(f"Kết quả được lưu tại: {OUTPUT}")


if __name__ == "__main__":
    main()
