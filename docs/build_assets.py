# -*- coding: utf-8 -*-
"""Sinh các sơ đồ minh hoạ cho slide/báo cáo (phong cách giống mẫu DevOps)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#0F172A"
BLUE = "#1D4ED8"
TEAL = "#0D9488"
PURPLE = "#7E22CE"
GREEN = "#16A34A"
AMBER = "#D97706"
SLATE = "#334155"
MUTED = "#64748B"
BORDER = "#E2E8F0"
SOFT = "#F1F5F9"


def canvas(w, h, bg="#FFFFFF"):
    fig = plt.figure(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_facecolor(bg)
    return fig, ax


def rbox(ax, x, y, w, h, text="", fc="#FFFFFF", ec=BORDER, tc=NAVY, fs=9,
         bold=False, lw=1.1, radius=0.07, pad=0.05, va="center", ha="center", ls=1.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2))
    if text:
        ax.text(x + w / 2 if ha == "center" else x + pad,
                y + h / 2 if va == "center" else y + pad,
                text, ha=ha, va=va, fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=ls, zorder=3)


def arrow(ax, p1, p2, color=BLUE, lw=1.5, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- cover art
def cover():
    w, h = 6.49, 4.99
    fig, ax = canvas(w, h, NAVY)
    for gx in range(0, 66):
        for gy in range(0, 51):
            ax.add_patch(Circle((gx * 0.1 + 0.05, gy * 0.1 + 0.05), 0.008,
                                color="#1E293B", zorder=1))

    rbox(ax, 0.35, 0.55, 5.80, 1.05, "", fc="#111C31", ec="#1E293B", radius=0.12)
    ax.text(0.55, 0.80, "DEEP LEARNING  ×  DEVOPS", fontsize=10, color="#60A5FA",
            fontweight="bold", zorder=3)
    ax.text(0.55, 1.10, "Transformer Seq2Seq (ViT5) phục vụ qua hạ tầng container",
            fontsize=11, color="#E2E8F0", zorder=3)

    ax.text(0.45, 2.00, "Hôm nai chời mưa to qúa", fontsize=19, color="#F87171",
            fontweight="bold", zorder=3)
    ax.text(0.45, 2.42, "Hôm nay trời mưa to quá", fontsize=19, color="#34D399",
            fontweight="bold", zorder=3)
    ax.text(0.45, 2.95, "Sửa 3 lỗi: nai → nay · chời → trời · qúa → quá", fontsize=9.5,
            color="#94A3B8", zorder=3)

    items = [("9.341", "câu VSEC"), ("11.202", "lỗi thật"), ("250M", "tham số")]
    for i, (big, small) in enumerate(items):
        x = 0.45 + i * 1.95
        ax.text(x, 3.35, big, fontsize=21, color="#38BDF8", fontweight="bold", zorder=3)
        ax.text(x, 3.66, small, fontsize=9.5, color="#94A3B8", zorder=3)

    rbox(ax, 0.35, 4.10, 5.80, 0.58, "Docker  ·  Nginx  ·  GitHub Actions  ·  Ubuntu 24.04 LTS",
         fc="#0B1220", ec="#1E293B", tc="#CBD5E1", fs=9.5, radius=0.1)
    save(fig, "cover.png")


# ---------------------------------------------------------------- inference
def inference():
    w, h = 6.90, 1.95
    fig, ax = canvas(w, h, "#FFFFFF")
    steps = [
        ("Văn bản lỗi", BLUE),
        ("Chuẩn hoá NFC", TEAL),
        ('Prefix\n"spelling: "', TEAL),
        ("Tokenizer\nSentencePiece", PURPLE),
        ("ViT5 12+12\nEncoder–Decoder", BLUE),
        ("Beam search\nnum_beams=4", PURPLE),
        ("Sequence\nMatcher", GREEN),
    ]
    bw, gap = 0.82, 0.135
    x = 0.10
    for i, (label, col) in enumerate(steps):
        rbox(ax, x, 0.45, bw, 0.95, label, fc="#FFFFFF", ec=col, tc=NAVY, fs=7.2)
        if i < len(steps) - 1:
            arrow(ax, (x + bw, 0.92), (x + bw + gap, 0.92), color="#94A3B8", lw=1.2)
        x += bw + gap
    ax.text(0.10, 1.66, "Câu đã sửa + danh sách vị trí sai (từ gốc → từ thay thế) trả về qua /api/correct_spelling",
            fontsize=7.6, color=MUTED)
    save(fig, "inference.png")


# ---------------------------------------------------------------- docker stages
def docker_stages():
    w, h = 6.40, 2.44
    fig, ax = canvas(w, h, "#FFFFFF")
    boxes = [
        ("BUILDER", "python:3.10-slim\n+ torch CPU\n+ transformers", SOFT, "#CBD5E1", MUTED),
        ("STAGE 1", "pip install\n--no-cache-dir\nsite-packages", "#FFFFFF", BLUE, BLUE),
        ("RUNTIME", "slim + site-packages\n+ app.py + src/\nuser non-root", "#EFF6FF", BLUE, BLUE),
        ("SERVING", "app.py :8000\nwarm-up 1 câu\nhealthcheck", "#ECFDF5", TEAL, TEAL),
    ]
    bw, gap, x = 1.38, 0.24, 0.15
    for i, (tag, body, fc, ec, tc) in enumerate(boxes):
        rbox(ax, x, 0.20, bw, 1.30, "", fc=fc, ec=ec, radius=0.08, lw=1.3)
        ax.text(x + bw / 2, 0.40, tag, ha="center", fontsize=7.4, color=tc, fontweight="bold")
        ax.text(x + bw / 2, 0.90, body, ha="center", va="center", fontsize=6.6, color=SLATE)
        if i < 3:
            arrow(ax, (x + bw, 0.85), (x + bw + gap, 0.85), color="#94A3B8")
        x += bw + gap
    rbox(ax, 0.15, 1.68, 6.10, 0.46,
         "Checkpoint 903,8 MB mount read-only  ─  không bake vào image",
         fc="#FFFBEB", ec="#FCD34D", tc="#92400E", fs=7.4, radius=0.06)
    ax.text(0.15, 2.28, "Image nhẹ, thay model không cần build lại; container chạy bằng user thường",
            fontsize=7.0, color=MUTED)
    save(fig, "docker_stages.png")


# ---------------------------------------------------------------- topology
def topology():
    w, h = 4.50, 2.55
    fig, ax = canvas(w, h, "#FFFFFF")
    rbox(ax, 0.10, 0.95, 0.95, 0.66, "Client\nHTTPS", fc=SOFT, ec="#CBD5E1", fs=7.2, tc=SLATE)
    arrow(ax, (1.05, 1.28), (1.32, 1.28), color="#94A3B8")
    rbox(ax, 1.34, 0.42, 1.20, 1.72, "", fc="#FFFFFF", ec=BLUE, lw=1.3)
    ax.text(1.94, 0.62, "Nginx", ha="center", fontsize=8.4, color=BLUE, fontweight="bold")
    ax.text(1.94, 1.10, "443 → TLS\n/api/* → 8000\nstatic + gzip\nrate limit",
            ha="center", va="center", fontsize=6.9, color=SLATE)
    arrow(ax, (2.54, 1.28), (2.80, 1.28), color="#94A3B8")
    rbox(ax, 2.82, 0.42, 1.56, 1.72, "", fc="#F5F3FF", ec=PURPLE, lw=1.3)
    ax.text(3.60, 0.62, "spellcheck_api", ha="center", fontsize=8.0, color=PURPLE, fontweight="bold")
    ax.text(3.60, 1.18, "app.py :8000\nViT5 (CPU)\n+ Rule engine", ha="center",
            va="center", fontsize=6.9, color=SLATE)
    rbox(ax, 1.34, 2.24, 1.20, 0.24, "certbot", fc="#ECFDF5", ec="#A7F3D0", tc=TEAL, fs=7.0)
    rbox(ax, 2.82, 2.24, 1.56, 0.24, "volume: checkpoints (ro)", fc=SOFT, ec="#CBD5E1",
         tc=SLATE, fs=6.8)
    rbox(ax, 0.10, 0.10, 4.28, 0.24, "Ubuntu 24.04 LTS  ·  Docker network  ·  chỉ mở 22 / 80 / 443",
         fc="#EFF6FF", ec="#BFDBFE", tc=BLUE, fs=6.9)
    save(fig, "topology.png")


# ---------------------------------------------------------------- architecture
def architecture():
    w, h = 5.50, 2.55
    fig, ax = canvas(w, h, "#FFFFFF")
    rows = [
        ("1. Trình bày", "Trình duyệt → Nginx (TLS) → HTML/CSS/JS trong app.py", BLUE, "#EFF6FF"),
        ("2. Suy luận", "app.py :8000 → ViT5 (beam 4)  |  Rule engine (N-gram)", PURPLE, "#F5F3FF"),
        ("3. Dữ liệu", "checkpoints/ (ro)  ·  data/processed/*.jsonl  ·  logs/", TEAL, "#ECFDF5"),
    ]
    y = 0.12
    for name, body, col, fc in rows:
        rbox(ax, 0.10, y, 5.30, 0.60, "", fc=fc, ec=col, lw=1.2)
        ax.text(0.24, y + 0.30, name, fontsize=7.8, color=col, fontweight="bold", va="center")
        ax.text(1.38, y + 0.30, body, fontsize=6.8, color=NAVY, va="center")
        if y < 1.0:
            arrow(ax, (2.75, y + 0.60), (2.75, y + 0.82), color="#94A3B8", lw=1.2)
        y += 0.80
    save(fig, "architecture.png")


# ---------------------------------------------------------------- pipeline
def pipeline():
    w, h = 5.03, 2.35
    fig, ax = canvas(w, h, "#FFFFFF")
    ax.text(0.10, 0.22, "GitHub Actions", fontsize=8.2, color=NAVY, fontweight="bold")
    stages = [("Push /\nPull Request", BLUE), ("Lint +\nsmoke test", TEAL),
              ("Build\nimage", PURPLE), ("Push\nGHCR", AMBER),
              ("SSH\ndeploy", GREEN), ("Health\ncheck", BLUE)]
    bw, gap, x = 0.72, 0.075, 0.10
    for i, (label, col) in enumerate(stages):
        rbox(ax, x, 0.45, bw, 0.85, label, fc="#FFFFFF", ec=col, fs=6.8, tc=NAVY)
        if i < len(stages) - 1:
            arrow(ax, (x + bw, 0.87), (x + bw + gap, 0.87), color="#94A3B8", lw=1.1)
        x += bw + gap
    rbox(ax, 0.10, 1.52, 5.03 - 0.20, 0.52,
         "Cổng chất lượng: test import + nạp tokenizer + 20 câu hồi quy (EM ≥ ngưỡng)",
         fc="#FFFBEB", ec="#FCD34D", tc="#92400E", fs=6.9)
    rbox(ax, 0.10, 2.10, 2.35, 0.22, "Nhánh main → deploy staging", fc=SOFT, ec="#CBD5E1", tc=SLATE, fs=6.6, radius=0.05)
    rbox(ax, 2.60, 2.10, 2.33, 0.22, "Rollback = deploy image SHA cũ", fc="#FEF2F2", ec="#FECACA", tc="#B91C1C", fs=6.6, radius=0.05)
    save(fig, "pipeline.png")


# ---------------------------------------------------------------- devops loop
def devops_loop():
    w, h = 3.28, 1.70
    fig, ax = canvas(w, h, "#FFFFFF")
    top = ["Plan", "Code", "Build", "Test"]
    bot = ["Release", "Deploy", "Operate", "Monitor"]
    bw, gap = 0.735, 0.055
    for i, lab in enumerate(top):
        x = 0.08 + i * (bw + gap)
        rbox(ax, x, 0.10, bw, 0.42, lab, fc="#EFF6FF", ec="#BFDBFE", tc=BLUE, fs=8, radius=0.05)
        if i < 3:
            arrow(ax, (x + bw, 0.31), (x + bw + gap, 0.31), color="#94A3B8", lw=1.1)
    for i, lab in enumerate(reversed(bot)):
        x = 0.08 + i * (bw + gap)
        rbox(ax, x, 0.68, bw, 0.42, lab, fc="#F5F3FF", ec="#DDD6FE", tc=PURPLE, fs=8, radius=0.05)
        if i > 0:
            arrow(ax, (x + bw, 0.89), (x + bw + gap, 0.89), color="#94A3B8", lw=1.1)
    arrow(ax, (0.08 + 3 * (bw + gap) + bw / 2, 0.52), (0.08 + 3 * (bw + gap) + bw / 2, 0.68), color="#94A3B8", lw=1.1)
    arrow(ax, (0.08 + bw / 2, 0.68), (0.08 + bw / 2, 0.52), color="#94A3B8", lw=1.1)
    ax.text(w / 2, 1.28, "8 giai đoạn khép kín, lặp liên tục", ha="center", fontsize=7.2, color=MUTED)
    ax.text(w / 2, 1.55, "Dev + Ops cùng chịu trách nhiệm", ha="center", fontsize=7.4, color=NAVY, fontweight="bold")
    save(fig, "devops_loop.png")


# ---------------------------------------------------------------- dora
def dora():
    w, h = 3.28, 1.53
    fig, ax = canvas(w, h, "#FFFFFF")
    rows = [("Lead Time", 0.34, BLUE), ("Deployment Freq.", 0.55, TEAL),
            ("Change Failure Rate", 0.22, AMBER), ("MTTR", 0.18, PURPLE)]
    y = 0.10
    for name, val, col in rows:
        ax.text(0.06, y + 0.13, name, fontsize=7.0, color=SLATE, va="center")
        ax.add_patch(FancyBboxPatch((1.72, y + 0.05), 1.38, 0.16,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=SOFT, edgecolor="none"))
        ax.add_patch(FancyBboxPatch((1.72, y + 0.05), 1.38 * val + 0.10, 0.16,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=col, edgecolor="none"))
        y += 0.35
    ax.text(0.06, 1.44, "Đo tốc độ & độ tin cậy thay vì cảm tính", fontsize=6.8, color=MUTED)
    save(fig, "dora.png")


# ---------------------------------------------------------------- before after
def before_after():
    w, h = 3.28, 1.80
    fig, ax = canvas(w, h, "#FFFFFF")
    rbox(ax, 0.05, 0.05, 1.52, 0.30, "Trước", fc="#FEF2F2", ec="#FECACA", tc="#B91C1C", fs=8, radius=0.05)
    rbox(ax, 1.71, 0.05, 1.52, 0.30, "Sau (DevOps)", fc="#ECFDF5", ec="#A7F3D0", tc=GREEN, fs=8, radius=0.05)
    rows = [("Cài thư viện thủ công", "Docker image bất biến"),
            ("Deploy copy file qua SSH", "Pipeline tự động"),
            ("Model nằm rải rác", "Checkpoint versioned"),
            ("Sự cố sửa tay", "Rollback theo image SHA")]
    y = 0.42
    for a, b in rows:
        ax.text(0.07, y + 0.25, "• " + a, fontsize=6.6, color="#7F1D1D", va="center")
        ax.text(1.73, y + 0.25, "• " + b, fontsize=6.6, color="#065F46", va="center")
        y += 0.34
    save(fig, "before_after.png")


# ---------------------------------------------------------------- terminal
def terminal():
    w, h = 3.07, 3.20
    fig, ax = canvas(w, h, "#FFFFFF")
    rbox(ax, 0.05, 0.05, 2.97, 3.10, "", fc=NAVY, ec=NAVY, radius=0.10)
    ax.add_patch(FancyBboxPatch((0.05, 0.05), 2.97, 0.30,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#111C31", edgecolor="none", zorder=2))
    for i, c in enumerate(["#F87171", "#FBBF24", "#34D399"]):
        ax.add_patch(Circle((0.18 + i * 0.12, 0.20), 0.028, color=c, zorder=3))
    ax.text(1.55, 0.20, "ubuntu@spellcheck:~", ha="center", va="center", fontsize=6.6,
            color="#94A3B8", zorder=3, family="Consolas")
    lines = [
        ("$ ls -la checkpoints/", "#38BDF8"),
        ("drwxr-xr-x  best_model", "#94A3B8"),
        ("-rw-r--r--  model.safetensors", "#94A3B8"),
        ("             903.8 MB", "#64748B"),
        ("$ free -m", "#38BDF8"),
        ("Mem: 3 720 / 3 918 MB used", "#94A3B8"),
        ("$ docker compose ps", "#38BDF8"),
        ("spellcheck_api  Up (healthy)", "#34D399"),
        ("spellcheck_nginx Up (healthy)", "#34D399"),
        ("$ curl -s localhost/api/status", "#38BDF8"),
        ('{"model_loaded": true}', "#FBBF24"),
    ]
    y = 0.50
    for text, col in lines:
        ax.text(0.14, y, text, fontsize=6.2, color=col, va="top", family="Consolas")
        y += 0.213
    save(fig, "terminal.png")


for fn in (cover, inference, docker_stages, topology, architecture, pipeline,
           devops_loop, dora, before_after, terminal):
    fn()
print("DONE")
