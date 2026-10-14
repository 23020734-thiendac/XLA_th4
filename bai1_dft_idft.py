"""Bài 1: tự xây dựng DFT, dịch tâm phổ và IDFT 2 chiều."""

from pathlib import Path
import sys

import matplotlib

HIEN_THI = "--hien-thi" in sys.argv
if not HIEN_THI:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


THU_MUC = Path(__file__).resolve().parent
ANH_GOC = THU_MUC / "week4_image.png"
OUTPUT = THU_MUC / "output_bai1"
KICH_THUOC = (32, 64)


def dft_2d(anh):
    """Ý 2: tính từng F(u,v) theo đúng công thức tổng kép DFT."""
    m, n = anh.shape
    pho = np.zeros((m, n), dtype=complex)

    # Tính trước các số mũ để vòng lặp chạy nhanh hơn:
    # exp_hang[u,x] = e^(-j2*pi*u*x/M)
    # exp_cot[v,y]  = e^(-j2*pi*v*y/N)
    u, x = np.arange(m), np.arange(m)
    v, y = np.arange(n), np.arange(n)
    exp_hang = np.exp(-2j * np.pi * np.outer(u, x) / m)
    exp_cot = np.exp(-2j * np.pi * np.outer(v, y) / n)

    for u in range(m):
        for v in range(n):
            tong = 0j
            for x in range(m):
                for y in range(n):
                    # e^(-j2*pi*(u*x/M + v*y/N))
                    he_so = exp_hang[u, x] * exp_cot[v, y]
                    tong += anh[x, y] * he_so
            pho[u, v] = tong
    return pho


def idft_2d(pho):
    """Ý 5: tính từng f(x,y) theo đúng công thức tổng kép IDFT."""
    m, n = pho.shape
    anh_khoi_phuc = np.zeros((m, n), dtype=complex)

    # IDFT đổi dấu số mũ từ âm sang dương.
    x, u = np.arange(m), np.arange(m)
    y, v = np.arange(n), np.arange(n)
    exp_hang = np.exp(2j * np.pi * np.outer(x, u) / m)
    exp_cot = np.exp(2j * np.pi * np.outer(y, v) / n)

    for x in range(m):
        for y in range(n):
            tong = 0j
            for u in range(m):
                for v in range(n):
                    # e^(j2*pi*(u*x/M + v*y/N))
                    he_so = exp_hang[x, u] * exp_cot[y, v]
                    tong += pho[u, v] * he_so
            anh_khoi_phuc[x, y] = tong / (m * n)
    return anh_khoi_phuc


def dich_tam_pho(pho):
    """Ý 4: tự dịch tâm phổ, không dùng fftshift."""
    m, n = pho.shape
    ket_qua = np.empty_like(pho)
    for u in range(m):
        for v in range(n):
            ket_qua[(u + m // 2) % m, (v + n // 2) % n] = pho[u, v]
    return ket_qua


def hien_thi(ax, anh, tieu_de, cmap="gray", vmin=None, vmax=None):
    ax.imshow(anh, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(tieu_de)
    ax.axis("off")


def xu_ly(kich_thuoc, anh_goc=ANH_GOC, output=OUTPUT, ten_anh="robot"):
    """Xử lý đủ 6 ý; cho phép tái sử dụng với một ảnh khác."""
    thu_muc_kq = output / f"{kich_thuoc}x{kich_thuoc}"
    thu_muc_kq.mkdir(parents=True, exist_ok=True)

    # Ý 1: đọc ảnh xám và đổi kích thước.
    with Image.open(anh_goc) as im:
        im = im.convert("L").resize(
            (kich_thuoc, kich_thuoc), Image.Resampling.LANCZOS
        )
    im.save(thu_muc_kq / f"{ten_anh}_{kich_thuoc}x{kich_thuoc}.png")
    anh = np.asarray(im, dtype=float)

    # Ý 2, 3: DFT rồi tính phần thực, phần ảo, biên độ và pha.
    pho = dft_2d(anh)
    phan_thuc, phan_ao = pho.real, pho.imag
    bien_do = np.sqrt(phan_thuc**2 + phan_ao**2)
    pha = np.arctan2(phan_ao, phan_thuc)

    # Ý 4, 5: dịch tâm phổ và khôi phục ảnh bằng IDFT.
    pho_dich_tam = dich_tam_pho(pho)
    anh_khoi_phuc = idft_2d(pho).real
    mse = np.mean((anh - anh_khoi_phuc) ** 2)

    # Ý 3: hiển thị các thành phần DFT.
    # Nén log có dấu chỉ để vẽ rõ hơn, không thay đổi R(u,v), I(u,v) thật.
    thuc_log = np.sign(phan_thuc) * np.log1p(np.abs(phan_thuc))
    ao_log = np.sign(phan_ao) * np.log1p(np.abs(phan_ao))
    max_thuc = max(np.max(np.abs(thuc_log)), 1e-12)
    max_ao = max(np.max(np.abs(ao_log)), 1e-12)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    noi_dung = [
        (anh, "Ảnh xám", "gray", 0, 255),
        (thuc_log, "Phần thực R(u,v) — nén log", "seismic", -max_thuc, max_thuc),
        (ao_log, "Phần ảo I(u,v) — nén log", "seismic", -max_ao, max_ao),
        (np.log1p(bien_do), "Phổ biên độ log(1 + |F|)", "gray", None, None),
        (pha, "Phổ pha φ(u,v)", "twilight", -np.pi, np.pi),
    ]
    for truc, du_lieu in zip(ax.ravel(), noi_dung):
        hien_thi(truc, *du_lieu)
    ax.ravel()[-1].axis("off")
    fig.suptitle(f"Các thành phần DFT — ảnh {kich_thuoc}×{kich_thuoc}")
    fig.text(0.5, 0.01, "Phần thực/ảo: đỏ = dương, xanh = âm, trắng = gần 0",
             ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(thu_muc_kq / "01_cac_thanh_phan_dft.png", dpi=160)

    # Ý 6: ảnh gốc, phổ trước/sau dịch tâm và ảnh sau IDFT.
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    noi_dung = [
        (anh, "Ảnh gốc"),
        (np.log1p(np.abs(pho)), "Phổ trước dịch tâm"),
        (np.log1p(np.abs(pho_dich_tam)), "Phổ sau dịch tâm"),
        (anh_khoi_phuc, f"Ảnh sau IDFT\nMSE = {mse:.3e}"),
    ]
    for truc, (du_lieu, ten) in zip(ax, noi_dung):
        hien_thi(truc, du_lieu, ten)
    fig.suptitle(f"DFT/IDFT — ảnh {kich_thuoc}×{kich_thuoc}")
    fig.tight_layout()
    fig.savefig(thu_muc_kq / "02_ket_qua_tong_hop.png", dpi=160)

    anh_uint8 = np.clip(np.rint(anh_khoi_phuc), 0, 255).astype(np.uint8)
    Image.fromarray(anh_uint8).save(thu_muc_kq / "03_anh_sau_idft.png")
    if not HIEN_THI:
        plt.close("all")

    print(f"Ảnh {kich_thuoc}x{kich_thuoc}: MSE = {mse:.12e}")
    return mse


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.mkdir(exist_ok=True)

    ket_qua = [f"{k}x{k}: MSE = {xu_ly(k):.12e}" for k in KICH_THUOC]
    (OUTPUT / "ket_qua_mse.txt").write_text("\n".join(ket_qua), encoding="utf-8")
    if HIEN_THI:
        plt.show()


if __name__ == "__main__":
    main()
