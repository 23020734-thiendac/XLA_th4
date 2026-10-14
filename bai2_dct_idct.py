"""Bài 2 - Ý 1 đến 6: chọn khối 8x8, tính DCT/IDCT và khảo sát hệ số."""

from pathlib import Path
import sys

import matplotlib

HIEN_THI = "--hien-thi" in sys.argv
if not HIEN_THI:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


THU_MUC = Path(__file__).resolve().parent
ANH_GOC = THU_MUC / "week4_image.png"
OUTPUT = THU_MUC / "output_bai2_dct"
N = 8
HANG, COT = 240, 240       # Góc trên trái của khối 8x8 được chọn


def alpha(k, n=N):
    """Hệ số chuẩn hóa alpha(k) trong công thức DCT."""
    return np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)


def dct_2d(khoi):
    """Ý 2: DCT 2D tự xây dựng đúng theo công thức tổng kép."""
    n = khoi.shape[0]
    C = np.zeros((n, n), dtype=float)

    for u in range(n):
        for v in range(n):
            tong = 0.0
            for x in range(n):
                for y in range(n):
                    cos_x = np.cos((2 * x + 1) * u * np.pi / (2 * n))
                    cos_y = np.cos((2 * y + 1) * v * np.pi / (2 * n))
                    tong += khoi[x, y] * cos_x * cos_y
            C[u, v] = alpha(u, n) * alpha(v, n) * tong
    return C


def idct_2d(C):
    """Ý 3: IDCT 2D tự xây dựng đúng theo công thức tổng kép."""
    n = C.shape[0]
    khoi_khoi_phuc = np.zeros((n, n), dtype=float)

    for x in range(n):
        for y in range(n):
            tong = 0.0
            for u in range(n):
                for v in range(n):
                    cos_x = np.cos((2 * x + 1) * u * np.pi / (2 * n))
                    cos_y = np.cos((2 * y + 1) * v * np.pi / (2 * n))
                    tong += (alpha(u, n) * alpha(v, n) * C[u, v]
                             * cos_x * cos_y)
            khoi_khoi_phuc[x, y] = tong
    return khoi_khoi_phuc


def dua_he_so_nho_ve_0(C, ty_le):
    """Ý 6: đưa ty_le phần trăm hệ số có |C(u,v)| nhỏ nhất về 0."""
    C_moi = C.copy()
    so_luong_xoa = round(C_moi.size * ty_le / 100)

    # Trải ma trận thành 1 chiều, sắp xếp theo độ lớn rồi lấy các vị trí nhỏ nhất.
    vi_tri_nho_nhat = np.argsort(np.abs(C_moi).ravel())[:so_luong_xoa]
    C_moi.ravel()[vi_tri_nho_nhat] = 0
    return C_moi, so_luong_xoa


def ghi_so_len_o(ax, ma_tran, mau="black"):
    """Ghi giá trị của ma trận vào từng ô để dễ quan sát."""
    for i in range(ma_tran.shape[0]):
        for j in range(ma_tran.shape[1]):
            ax.text(j, i, f"{ma_tran[i, j]:.0f}", ha="center", va="center",
                    fontsize=7, color=mau)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not ANH_GOC.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {ANH_GOC}")
    OUTPUT.mkdir(exist_ok=True)

    # Ý 1: đọc ảnh xám và chọn khối 8x8.
    with Image.open(ANH_GOC) as im:
        anh = np.asarray(im.convert("L"), dtype=float)
    khoi = anh[HANG:HANG + N, COT:COT + N]
    if khoi.shape != (N, N):
        raise ValueError("Khối 8x8 vượt ra ngoài biên ảnh")

    # Ý 2: tự tính DCT 2 chiều.
    he_so_dct = dct_2d(khoi)

    # Ý 3: tự tính IDCT để khôi phục khối ảnh ban đầu.
    khoi_khoi_phuc = idct_2d(he_so_dct)
    mse = np.mean((khoi - khoi_khoi_phuc) ** 2)
    sai_so_lon_nhat = np.max(np.abs(khoi - khoi_khoi_phuc))

    # Ý 5: lưu ma trận hệ số DCT 8x8 vào đúng định dạng .txt theo đề bài.
    np.savetxt(OUTPUT / "he_so_dct.txt", he_so_dct, fmt="%12.6f")

    # Hai file CSV chỉ dùng để mở nhanh bằng Excel/WPS; không bắt buộc ở ý 5.
    # Nếu CSV đang mở trong WPS, bỏ qua để chương trình vẫn chạy bình thường.
    try:
        np.savetxt(OUTPUT / "khoi_8x8.csv", khoi, fmt="%.0f", delimiter=",")
        np.savetxt(OUTPUT / "he_so_dct.csv", he_so_dct, fmt="%.6f", delimiter=",")
    except PermissionError:
        print("Không ghi đè CSV vì file đang được mở trong Excel/WPS.")
    Image.fromarray(khoi.astype(np.uint8)).resize((256, 256)).save(
        OUTPUT / "khoi_8x8_phong_to.png"
    )

    # Hình 1: vị trí khối trong ảnh và các giá trị pixel của khối.
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(anh, cmap="gray", vmin=0, vmax=255)
    ax[0].add_patch(Rectangle((COT, HANG), N, N, fill=False,
                              edgecolor="red", linewidth=2))
    ax[0].set_title(f"Ảnh xám — khối tại ({HANG}, {COT})")
    ax[0].axis("off")

    ax[1].imshow(khoi, cmap="gray", vmin=0, vmax=255)
    ghi_so_len_o(ax[1], khoi, "red")
    ax[1].set_title("Khối ảnh 8×8 và giá trị pixel")
    ax[1].set_xticks(range(N))
    ax[1].set_yticks(range(N))
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_chon_khoi_8x8.png", dpi=160)

    # Hình 2: hệ số DCT; signed-log chỉ dùng để màu dễ quan sát.
    dct_hien_thi = np.sign(he_so_dct) * np.log1p(np.abs(he_so_dct))
    gioi_han = np.max(np.abs(dct_hien_thi))
    fig, ax = plt.subplots(figsize=(7, 6))
    hinh = ax.imshow(dct_hien_thi, cmap="seismic",
                     vmin=-gioi_han, vmax=gioi_han)
    ghi_so_len_o(ax, he_so_dct)
    ax.set_title("Hệ số DCT 2D của khối 8×8\n(màu được nén log, số là giá trị thật)")
    ax.set_xlabel("v")
    ax.set_ylabel("u")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    fig.colorbar(hinh, ax=ax, label="sign(C)·log(1+|C|)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_he_so_dct.png", dpi=160)

    # Hình 3: so sánh khối gốc, khối sau IDCT và sai số tuyệt đối.
    sai_so = np.abs(khoi - khoi_khoi_phuc)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    ax[0].imshow(khoi, cmap="gray", vmin=0, vmax=255)
    ghi_so_len_o(ax[0], khoi)
    ax[0].set_title("Khối ảnh gốc 8×8")

    ax[1].imshow(khoi_khoi_phuc, cmap="gray", vmin=0, vmax=255)
    ghi_so_len_o(ax[1], khoi_khoi_phuc)
    ax[1].set_title(f"Khối sau IDCT\nMSE = {mse:.3e}")

    hinh = ax[2].imshow(sai_so, cmap="hot")
    ax[2].set_title(f"Sai số tuyệt đối\nMax = {sai_so_lon_nhat:.3e}")
    fig.colorbar(hinh, ax=ax[2], fraction=0.046, pad=0.04)

    for truc in ax:
        truc.set_xticks(range(N))
        truc.set_yticks(range(N))
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_so_sanh_idct.png", dpi=160)

    khoi_uint8 = np.clip(np.rint(khoi_khoi_phuc), 0, 255).astype(np.uint8)
    Image.fromarray(khoi_uint8).resize((256, 256)).save(
        OUTPUT / "khoi_8x8_sau_idct.png"
    )

    # Ý 4: hình tổng hợp khối gốc, DCT, DC, AC và khối sau IDCT.
    vi_tri_ac = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 2)]
    noi_dung_ac = "\n".join(
        f"C({u},{v}) = {he_so_dct[u, v]:.3f}" for u, v in vi_tri_ac
    )

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    ax[0, 0].imshow(khoi, cmap="gray", vmin=0, vmax=255)
    ghi_so_len_o(ax[0, 0], khoi)
    ax[0, 0].set_title("Khối ảnh gốc 8×8")

    ax[0, 1].imshow(dct_hien_thi, cmap="seismic",
                    vmin=-gioi_han, vmax=gioi_han)
    ghi_so_len_o(ax[0, 1], he_so_dct)
    ax[0, 1].set_title("Ma trận hệ số DCT 8×8")
    ax[0, 1].set_xlabel("v")
    ax[0, 1].set_ylabel("u")

    ax[0, 2].axis("off")
    ax[0, 2].text(0.5, 0.68, "HỆ SỐ DC", ha="center", va="center",
                  fontsize=18, fontweight="bold")
    ax[0, 2].text(0.5, 0.48, f"C(0,0) = {he_so_dct[0, 0]:.3f}",
                  ha="center", va="center", fontsize=17,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#eeeeee",
                            edgecolor="black"))
    ax[0, 2].text(0.5, 0.27, "Biểu diễn mức sáng trung bình\ncủa khối ảnh",
                  ha="center", va="center", fontsize=12)

    ax[1, 0].axis("off")
    ax[1, 0].text(0.5, 0.92, "MỘT SỐ HỆ SỐ AC", ha="center", va="top",
                  fontsize=17, fontweight="bold")
    ax[1, 0].text(0.5, 0.48, noi_dung_ac, ha="center", va="center",
                  fontsize=14, linespacing=1.5,
                  bbox=dict(boxstyle="round,pad=0.7", facecolor="white",
                            edgecolor="black"))

    ax[1, 1].imshow(khoi_khoi_phuc, cmap="gray", vmin=0, vmax=255)
    ghi_so_len_o(ax[1, 1], khoi_khoi_phuc)
    ax[1, 1].set_title(f"Khối sau IDCT\nMSE = {mse:.3e}")

    hinh_sai_so = ax[1, 2].imshow(sai_so, cmap="hot")
    ax[1, 2].set_title(f"Sai số tuyệt đối\nMax = {sai_so_lon_nhat:.3e}")
    fig.colorbar(hinh_sai_so, ax=ax[1, 2], fraction=0.046, pad=0.04)

    for truc in (ax[0, 0], ax[0, 1], ax[1, 1], ax[1, 2]):
        truc.set_xticks(range(N))
        truc.set_yticks(range(N))

    fig.suptitle("Ý 4 — Tổng hợp kết quả DCT/IDCT khối ảnh 8×8",
                 fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "04_hien_thi_tong_hop.png", dpi=160)

    # Ý 6: lần lượt xóa 50%, 75% và 90% hệ số DCT nhỏ nhất rồi IDCT.
    ty_le_xoa = [50, 75, 90]
    ket_qua_y6 = []

    for ty_le in ty_le_xoa:
        dct_sau_xoa, so_luong_xoa = dua_he_so_nho_ve_0(he_so_dct, ty_le)
        anh_sau_idct = idct_2d(dct_sau_xoa)
        mse_y6 = np.mean((khoi - anh_sau_idct) ** 2)
        ket_qua_y6.append((ty_le, so_luong_xoa, anh_sau_idct, mse_y6))

        anh_luu = np.clip(np.rint(anh_sau_idct), 0, 255).astype(np.uint8)
        Image.fromarray(anh_luu).resize((256, 256)).save(
            OUTPUT / f"khoi_8x8_idct_xoa_{ty_le}.png"
        )

    # So sánh trực quan theo bố cục 2x2 để hình không bị cắt mất nhãn trục.
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    cac_truc = ax.ravel()
    cac_truc[0].imshow(khoi, cmap="gray", vmin=0, vmax=255)
    ghi_so_len_o(cac_truc[0], khoi)
    cac_truc[0].set_title("Khối gốc")

    for truc, (ty_le, so_luong_xoa, anh_idct, mse_y6) in zip(
            cac_truc[1:], ket_qua_y6):
        truc.imshow(anh_idct, cmap="gray", vmin=0, vmax=255)
        ghi_so_len_o(truc, anh_idct)
        truc.set_title(
            f"Đưa {ty_le}% về 0 ({so_luong_xoa}/64)\nMSE = {mse_y6:.3f}"
        )

    for truc in cac_truc:
        truc.set_xticks(range(N))
        truc.set_yticks(range(N))
    fig.suptitle("Ý 6 — Ảnh sau IDCT khi loại các hệ số DCT nhỏ nhất",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT / "06_so_sanh_loai_he_so_dct.png", dpi=160,
                bbox_inches="tight", pad_inches=0.2)

    if HIEN_THI:
        plt.show()
    else:
        plt.close("all")

    print(f"Đã chọn khối 8x8 tại hàng {HANG}, cột {COT}")
    print(f"Hệ số DC C(0,0) = {he_so_dct[0, 0]:.6f}")
    print("Một số hệ số AC:")
    print(noi_dung_ac)
    print(f"Đã lưu ma trận DCT vào: {OUTPUT / 'he_so_dct.txt'}")
    print(f"MSE giữa khối gốc và khối sau IDCT = {mse:.12e}")
    print(f"Sai số tuyệt đối lớn nhất = {sai_so_lon_nhat:.12e}")
    print("Ý 6 - kết quả sau khi đưa các hệ số nhỏ nhất về 0:")
    for ty_le, so_luong_xoa, _, mse_y6 in ket_qua_y6:
        print(f"  {ty_le}%: xóa {so_luong_xoa}/64 hệ số, MSE = {mse_y6:.6f}")
    print(f"Kết quả được lưu tại: {OUTPUT}")


if __name__ == "__main__":
    main()
