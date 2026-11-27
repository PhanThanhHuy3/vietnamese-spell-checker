"""Hệ thống Web UI Đồ Án: Phát Hiện và Tự Động Sửa Lỗi Chính Tả Tiếng Việt
Kiến trúc: Transformer Seq2Seq (ViT5) Finetuned trên VSEC Corpus.
Hỗ trợ:
  1. Kiểm tra và sửa lỗi chính tả tương tác trực quan (Interactive AI Corrector kiểu Grammarly)
  2. So sánh đối sánh song song (ViT5 Deep Learning vs Rule Engine)
  3. Bộ sinh lỗi nhân tạo mở rộng dữ liệu (Data Augmentation & Noise Synthesizer)
  4. Báo cáo thực nghiệm khoa học & Đồ thị hội tụ Loss chuẩn mực
"""

import http.server
import json
import os
import re
import socketserver
import sys
import time
import unicodedata
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Đảm bảo UTF-8 an toàn tuyệt đối trên mọi console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.noise_generator import VietnameseNoiseGenerator
from src.smart_corrector import SmartVietnameseCorrector

# Khởi tạo corrector và nạp mô hình AI ViT5
print("[*] Đang khởi động hệ thống Sửa lỗi chính tả tiếng Việt...")
smart_corrector = SmartVietnameseCorrector()
ai_loaded = smart_corrector.load_ai_model()
noise_gen = VietnameseNoiseGenerator(seed=42)

print(f"[*] Trạng thái AI Model: {'ĐÃ NẠP THÀNH CÔNG (ViT5 Checkpoint)' if ai_loaded else 'Chưa tìm thấy checkpoint (Dùng fallback)'}")


def categorize_error(original: str, corrected: str) -> Dict[str, str]:
    """Phân loại lỗi chính tả tiếng Việt theo đặc trưng ngôn ngữ học."""
    orig_clean = original.strip(".,;:?!\"'()[]{}«»-–").lower()
    corr_clean = corrected.strip(".,;:?!\"'()[]{}«»-–").lower()

    # Bảng phụ âm đầu phương ngữ
    phonetics = [
        ("ch", "tr"), ("tr", "ch"), ("s", "x"), ("x", "s"),
        ("d", "gi"), ("gi", "d"), ("d", "r"), ("r", "d"), ("gi", "r"), ("r", "gi"),
        ("l", "n"), ("n", "l"), ("c", "k"), ("k", "c"),
        ("g", "gh"), ("gh", "g"), ("ng", "ngh"), ("ngh", "ng")
    ]
    for p1, p2 in phonetics:
        if orig_clean.startswith(p1) and corr_clean.startswith(p2):
            return {"type": "phonetic_initial", "label": "Lỗi âm đầu phương ngữ", "badge": "badge-phonetic", "icon": "🗣️"}

    # Bảng phụ âm cuối phương ngữ (t/c, n/ng)
    final_cons = [("t", "c"), ("c", "t"), ("n", "ng"), ("ng", "n")]
    for f1, f2 in final_cons:
        if orig_clean.endswith(f1) and corr_clean.endswith(f2):
            return {"type": "phonetic_final", "label": "Lỗi âm cuối phương ngữ", "badge": "badge-phonetic", "icon": "🗣️"}

    # Bảng hỏi / ngã
    hoi_nga_pairs = [
        ("ả", "ã"), ("ã", "ả"), ("ẳ", "ẵ"), ("ẵ", "ẳ"), ("ẩ", "ẫ"), ("ẫ", "ẩ"),
        ("ẻ", "ẽ"), ("ẽ", "ẻ"), ("ể", "ễ"), ("ễ", "ể"), ("ỉ", "ĩ"), ("ĩ", "ỉ"),
        ("ỏ", "õ"), ("õ", "ỏ"), ("ổ", "ỗ"), ("ỗ", "ổ"), ("ở", "ỡ"), ("ỡ", "ở"),
        ("ủ", "ũ"), ("ũ", "ủ"), ("ử", "ữ"), ("ữ", "ử"), ("ỷ", "ỹ"), ("ỹ", "ỷ")
    ]
    for h, n in hoi_nga_pairs:
        if h in orig_clean and n in corr_clean:
            return {"type": "tone", "label": "Lỗi dấu Hỏi / Ngã", "badge": "badge-tone", "icon": "❓"}

    # Lỗi mất dấu (rụng dấu)
    def strip_acc(s):
        norm = unicodedata.normalize("NFD", s)
        return "".join(c for c in norm if unicodedata.category(c) != "Mn")
    if strip_acc(orig_clean) == strip_acc(corr_clean) and orig_clean != corr_clean:
        return {"type": "tone_drop", "label": "Lỗi rụng dấu / Dấu thanh", "badge": "badge-tone", "icon": "🔤"}

    # Lỗi gõ Telex / VNI (dính s, f, r, x, j, w, số 1-9)
    if re.search(r'[sfrxjwaeod1-9]$', orig_clean) or re.search(r'(aa|oo|ee|dd|uw|ow|as|af|ar|ax|aj)', orig_clean):
        return {"type": "telex", "label": "Lỗi gõ phím Telex / VNI", "badge": "badge-telex", "icon": "⌨️"}

    # Lỗi đảo ký tự (hoán vị)
    if sorted(orig_clean) == sorted(corr_clean):
        return {"type": "swap", "label": "Lỗi đảo vị trí ký tự", "badge": "badge-other", "icon": "🔄"}

    return {"type": "vocab", "label": "Lỗi từ vựng & Ngữ cảnh", "badge": "badge-context", "icon": "🧠"}


HTML_CONTENT = """<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ Thống Tự Động Sửa Lỗi Chính Tả Tiếng Việt - ViT5 Deep Learning</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --surface: #111726;
            --surface-hover: #172033;
            --border: #1e293b;
            --border-highlight: #3b82f6;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --primary-glow: rgba(79, 70, 229, 0.35);
            --accent-red: #ef4444;
            --accent-red-bg: rgba(239, 68, 68, 0.12);
            --accent-green: #10b981;
            --accent-green-bg: rgba(16, 185, 129, 0.12);
            --accent-amber: #f59e0b;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }

        [data-theme="light"] {
            --bg: #f8fafc;
            --surface: #ffffff;
            --surface-hover: #f1f5f9;
            --border: #e2e8f0;
            --border-highlight: #4f46e5;
            --text: #0f172a;
            --text-muted: #64748b;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --primary-glow: rgba(79, 70, 229, 0.2);
            --accent-red: #dc2626;
            --accent-red-bg: #fee2e2;
            --accent-green: #059669;
            --accent-green-bg: #d1fae5;
            --accent-amber: #d97706;
            --accent-purple: #7c3aed;
            --accent-cyan: #0891b2;
            --shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
        body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 1.5rem 1rem; display: flex; flex-direction: column; align-items: center; transition: background 0.25s, color 0.25s; }
        .container { width: 100%; max-width: 1140px; }

        /* Top Bar */
        .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
        .brand-badge {
            display: inline-flex; align-items: center; gap: 0.5rem;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(14, 165, 233, 0.15));
            border: 1px solid rgba(99, 102, 241, 0.35); color: #818cf8;
            padding: 0.35rem 0.9rem; border-radius: 9999px; font-size: 0.82rem; font-weight: 700;
        }
        .theme-toggle {
            background: var(--surface); border: 1px solid var(--border); color: var(--text);
            padding: 0.4rem 0.85rem; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
            display: flex; align-items: center; gap: 0.4rem; transition: all 0.2s;
        }
        .theme-toggle:hover { background: var(--surface-hover); }

        /* Header */
        header { text-align: center; margin-bottom: 2rem; position: relative; }
        h1 { font-size: 2.25rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.6rem; }
        .subtitle { color: var(--text-muted); font-size: 1.02rem; max-width: 720px; margin: 0 auto; line-height: 1.6; }

        .meta-badges { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.1rem; }
        .tag-pill { background: var(--surface); border: 1px solid var(--border); color: #38bdf8; font-size: 0.78rem; padding: 0.25rem 0.7rem; border-radius: 6px; font-weight: 600; }

        /* Tabs */
        .tab-nav { display: flex; gap: 0.4rem; background: var(--surface); padding: 0.35rem; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 1.5rem; overflow-x: auto; }
        .tab-btn {
            flex: 1; min-width: 170px; padding: 0.8rem 1rem; border: none; background: transparent; color: var(--text-muted);
            border-radius: 9px; font-weight: 600; font-size: 0.92rem; cursor: pointer; transition: all 0.2s ease;
            display: flex; align-items: center; justify-content: center; gap: 0.5rem; white-space: nowrap;
        }
        .tab-btn:hover { color: var(--text); background: var(--surface-hover); }
        .tab-btn.active { background: var(--primary); color: #ffffff; box-shadow: 0 4px 14px var(--primary-glow); }

        /* Cards */
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: var(--shadow); }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem; }
        .card-title { font-size: 1.15rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; }

        /* Textarea Container */
        .editor-wrap { position: relative; margin-bottom: 1rem; }
        textarea {
            width: 100%; min-height: 140px; background: rgba(0,0,0,0.15); border: 1px solid var(--border); border-radius: 10px;
            padding: 1.1rem; color: var(--text); font-size: 1.05rem; line-height: 1.7; resize: vertical; transition: border-color 0.2s;
        }
        [data-theme="light"] textarea { background: #ffffff; }
        textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }

        .editor-footer { display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: var(--text-muted); margin-top: 0.5rem; flex-wrap: wrap; gap: 0.5rem; }
        .char-counter { font-family: 'JetBrains Mono', monospace; font-weight: 500; }

        /* Samples Row */
        .sample-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; margin-bottom: 1.25rem; }
        .sample-label { font-size: 0.82rem; font-weight: 700; color: var(--text-muted); margin-right: 0.3rem; }
        .sample-pill {
            background: rgba(255, 255, 255, 0.04); border: 1px solid var(--border); color: var(--text-muted); font-size: 0.8rem; padding: 0.35rem 0.75rem;
            border-radius: 9999px; cursor: pointer; transition: all 0.15s ease;
        }
        .sample-pill:hover { background: var(--surface-hover); border-color: var(--border-highlight); color: var(--text); transform: translateY(-1px); }

        /* Engine Controls */
        .engine-picker {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.75rem;
            background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 10px; border: 1px solid var(--border); margin-bottom: 1.25rem;
        }
        [data-theme="light"] .engine-picker { background: #f8fafc; }
        .engine-option {
            display: flex; align-items: flex-start; gap: 0.65rem; padding: 0.65rem 0.85rem; border-radius: 8px;
            cursor: pointer; border: 1px solid transparent; transition: all 0.2s ease;
        }
        .engine-option:hover { background: var(--surface-hover); }
        .engine-option input[type="radio"] { margin-top: 0.25rem; accent-color: var(--primary); }
        .engine-title { font-weight: 700; font-size: 0.92rem; display: block; }
        .engine-desc { font-size: 0.76rem; color: var(--text-muted); line-height: 1.35; margin-top: 0.2rem; }

        /* Action Buttons */
        .action-row { display: flex; gap: 0.75rem; flex-wrap: wrap; }
        .btn-primary {
            flex: 2; min-width: 200px; display: flex; align-items: center; justify-content: center; gap: 0.5rem;
            background: linear-gradient(135deg, var(--primary) 0%, #6366f1 100%); color: #ffffff;
            border: none; padding: 0.9rem 1.6rem; border-radius: 10px; font-weight: 700; font-size: 1rem;
            cursor: pointer; transition: all 0.2s ease; box-shadow: 0 4px 18px var(--primary-glow);
        }
        .btn-primary:hover { opacity: 0.95; transform: translateY(-1px); box-shadow: 0 6px 22px rgba(79, 70, 229, 0.4); }
        .btn-secondary {
            display: flex; align-items: center; justify-content: center; gap: 0.4rem;
            background: var(--surface-hover); color: var(--text); border: 1px solid var(--border); padding: 0.85rem 1.2rem;
            border-radius: 10px; font-weight: 600; font-size: 0.9rem; cursor: pointer; transition: all 0.2s ease;
        }
        .btn-secondary:hover { background: var(--border); }

        /* Results Display */
        .results-section { display: none; margin-top: 1.75rem; }
        .stats-bar {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .stat-card {
            background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 10px; padding: 0.85rem; text-align: center;
        }
        [data-theme="light"] .stat-card { background: #f8fafc; }
        .stat-value { font-size: 1.4rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
        .stat-value.errors { color: var(--accent-red); }
        .stat-value.fixed { color: var(--accent-green); }
        .stat-value.latency { color: #38bdf8; }
        .stat-label { font-size: 0.76rem; color: var(--text-muted); margin-top: 0.25rem; }

        /* Error breakdown progress bar */
        .error-breakdown { background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; margin-bottom: 1.25rem; }
        .breakdown-bar { height: 10px; border-radius: 5px; background: #1e293b; display: flex; overflow: hidden; margin: 0.6rem 0; }
        .breakdown-seg { height: 100%; transition: width 0.4s ease; }
        .breakdown-labels { display: flex; flex-wrap: wrap; gap: 1rem; font-size: 0.78rem; }
        .breakdown-item { display: flex; align-items: center; gap: 0.4rem; }
        .breakdown-dot { width: 8px; height: 8px; border-radius: 50%; }

        /* Dual View */
        .dual-view { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem; }
        @media (max-width: 768px) { .dual-view { grid-template-columns: 1fr; } }

        .view-panel { background: rgba(0,0,0,0.15); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; }
        [data-theme="light"] .view-panel { background: #f8fafc; }
        .view-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
        .view-header.red { color: #f87171; }
        .view-header.green { color: #34d399; }
        .text-display { font-size: 1.08rem; line-height: 1.85; word-break: break-word; min-height: 90px; }

        /* Interactive Tokens */
        .error-token {
            background: var(--accent-red-bg); color: var(--accent-red); padding: 0.15rem 0.4rem; border-radius: 4px;
            border-bottom: 2px wavy var(--accent-red); font-weight: 700; cursor: pointer; transition: all 0.15s;
            position: relative; display: inline-block;
        }
        .error-token:hover { background: rgba(239, 68, 68, 0.25); transform: translateY(-1px); }
        .fixed-token {
            background: var(--accent-green-bg); color: var(--accent-green); padding: 0.15rem 0.4rem; border-radius: 4px;
            border-bottom: 2px solid var(--accent-green); font-weight: 700; display: inline-block;
        }

        /* Popover Quick Card */
        .popover-card {
            background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-top: 1rem; display: none;
            animation: fadeIn 0.2s ease;
        }
        .popover-card.active { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }

        /* Error Details Table */
        .table-responsive { overflow-x: auto; margin-top: 1rem; }
        .details-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; text-align: left; }
        .details-table th { background: rgba(0,0,0,0.2); color: var(--text-muted); padding: 0.75rem 0.9rem; font-size: 0.78rem; text-transform: uppercase; border-bottom: 1px solid var(--border); }
        .details-table td { padding: 0.85rem 0.9rem; border-bottom: 1px solid var(--border); }
        .details-table tr:hover { background: var(--surface-hover); }

        .badge-pill { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
        .badge-phonetic { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-tone { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
        .badge-telex { background: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }
        .badge-context { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-other { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }

        /* Toast */
        .toast {
            position: fixed; bottom: 2rem; right: 2rem; background: var(--surface); color: var(--text);
            padding: 0.85rem 1.4rem; border-radius: 10px; border-left: 4px solid var(--accent-green);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5); font-size: 0.92rem; display: none; z-index: 999;
            border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
        }

        /* Tab Content */
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.2s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

        /* Compare Grid */
        .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
        @media (max-width: 768px) { .compare-grid { grid-template-columns: 1fr; } }
        .compare-box { background: rgba(0,0,0,0.15); border: 1px solid var(--border); border-radius: 10px; padding: 1.1rem; }
        [data-theme="light"] .compare-box { background: #f8fafc; }
        .compare-title { font-size: 0.95rem; font-weight: 700; margin-bottom: 0.6rem; display: flex; align-items: center; justify-content: space-between; }

        /* Chart SVG */
        .chart-container { background: rgba(0,0,0,0.25); border-radius: 10px; padding: 1.5rem; margin-top: 1rem; border: 1px solid var(--border); }
        [data-theme="light"] .chart-container { background: #f8fafc; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Bar -->
        <div class="top-bar">
            <div class="brand-badge">🇻🇳 ĐỒ ÁN HỌC SÂU (DEEP LEARNING) - VIETNAMESE NLP</div>
            <button class="theme-toggle" onclick="toggleTheme()">
                <span id="theme-icon">☀️</span> <span id="theme-label">Giao diện Sáng</span>
            </button>
        </div>

        <!-- Header -->
        <header>
            <h1>Phát Hiện & Tự Động Sửa Lỗi Chính Tả Tiếng Việt</h1>
            <p class="subtitle">Ứng dụng kiến trúc <strong>Transformer Seq2Seq (ViT5)</strong> huấn luyện chuyên sâu trên ngữ liệu chuẩn <strong>VSEC Corpus</strong> kết hợp cơ chế kiểm soát ngữ cảnh từ vựng</p>
            <div class="meta-badges">
                <span class="tag-pill">🧠 Model: VietAI/vit5-base (~226M params)</span>
                <span class="tag-pill">📚 Dữ liệu: VSEC Corpus 9.341 câu (11.202 lỗi thực tế)</span>
                <span class="tag-pill">🎯 Độ đo: EM, Accuracy, BLEU-4, CER, WER, F1-Score</span>
            </div>
        </header>

        <!-- Navigation Tabs -->
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('corrector')">
                ✨ Trình Sửa Lỗi AI (Grammarly VN)
            </button>
            <button class="tab-btn" onclick="switchTab('compare')">
                ⚡ Đối Sánh (ViT5 vs Rule Engine)
            </button>
            <button class="tab-btn" onclick="switchTab('noise')">
                ⚙️ Bộ Sinh Lỗi Mở Rộng Dữ Liệu
            </button>
            <button class="tab-btn" onclick="switchTab('report')">
                📊 Báo Cáo Thực Nghiệm & Loss
            </button>
        </div>

        <!-- ================= TAB 1: TRÌNH SỬA LỖI AI ================= -->
        <div id="tab-corrector" class="tab-content active">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📝 Soạn Thảo Hoặc Dán Văn Bản Cần Sửa Lỗi</span>
                    <div style="display:flex; gap:0.5rem;">
                        <button class="btn-secondary" style="padding:0.35rem 0.75rem; font-size:0.8rem;" onclick="clearEditor()">
                            🗑️ Xóa trắng
                        </button>
                    </div>
                </div>

                <div class="editor-wrap">
                    <textarea id="input-text" placeholder="Nhập hoặc dán văn bản tiếng Việt có chứa lỗi chính tả vào đây...">Hôm nai chời mưa to qúa, tôi quến mang rù đi họt. Tối nay tôi tự nấu ăn ở nhà để tiet kiệm fiền.</textarea>
                    <div class="editor-footer">
                        <span>💡 Phím tắt: <strong>Ctrl + Enter</strong> để sửa ngay. Hỗ trợ lỗi âm đầu, hỏi ngã, rụng dấu, telex, ngữ cảnh.</span>
                        <span class="char-counter" id="char-counter">0 từ | 0 ký tự</span>
                    </div>
                </div>

                <!-- Thử câu mẫu thực tế -->
                <div class="sample-row">
                    <span class="sample-label">Thử câu mẫu:</span>
                    <span class="sample-pill" onclick="setSample(1)">🌧️ Quên mang dù & Đi học</span>
                    <span class="sample-pill" onclick="setSample(2)">🍲 Tự nấu ăn & Tiết kiệm tiền</span>
                    <span class="sample-pill" onclick="setSample(3)">🏛️ ĐH Bách Khoa (Câu sạch không lỗi)</span>
                    <span class="sample-pill" onclick="setSample(4)">❄️ Mùa đông se lạnh & Mưa phùn</span>
                    <span class="sample-pill" onclick="setSample(5)">🤖 Học sâu & Mạng nơ-ron</span>
                    <span class="sample-pill" onclick="setSample(6)">📚 Lỗi hỏi / ngã & Lẫn lộn</span>
                </div>

                <!-- Lựa chọn động cơ -->
                <div class="engine-picker">
                    <label class="engine-option">
                        <input type="radio" name="engine" value="ai" checked>
                        <div>
                            <span class="engine-title">🧠 ViT5 Deep Learning (Khuyên Dùng)</span>
                            <span class="engine-desc">Mô hình Transformer Seq2Seq vừa huấn luyện trên VSEC, bắt lỗi chuẩn xác theo ngữ cảnh.</span>
                        </div>
                    </label>
                    <label class="engine-option">
                        <input type="radio" name="engine" value="hybrid">
                        <div>
                            <span class="engine-title">🚀 Hybrid (ViT5 + Từ Điển Bảo Toàn)</span>
                            <span class="engine-desc">Kết hợp sức mạnh Transformer và từ điển 74K từ tiếng Việt để bảo vệ tên riêng và thực thể.</span>
                        </div>
                    </label>
                    <label class="engine-option">
                        <input type="radio" name="engine" value="rule">
                        <div>
                            <span class="engine-title">⚡ Rule Engine & N-gram</span>
                            <span class="engine-desc">Động cơ quy luật & N-gram truyền thống (xử lý siêu tốc 0.01s).</span>
                        </div>
                    </label>
                </div>

                <!-- Nút chạy -->
                <div class="action-row">
                    <button class="btn-primary" id="btn-correct" onclick="runCorrection()">
                        <span id="btn-text">🔍 Kiểm Tra & Sửa Lỗi Ngay</span>
                        <span id="btn-spinner" style="display:none;">⏳ Đang giải mã Transformer...</span>
                    </button>
                </div>

                <!-- Khu vực hiển thị kết quả -->
                <div class="results-section" id="results-box">
                    <!-- Thẻ thống kê định lượng -->
                    <div class="stats-bar">
                        <div class="stat-card">
                            <div class="stat-value errors" id="stat-error-count">0</div>
                            <div class="stat-label">Lỗi phát hiện</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value fixed" id="stat-fixed-count">0</div>
                            <div class="stat-label">Vị trí đã chuẩn hóa</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value latency" id="stat-latency">0 ms</div>
                            <div class="stat-label">Thời gian phản hồi</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-engine" style="font-size:0.92rem; color:#818cf8; margin-top:0.35rem;">ViT5</div>
                            <div class="stat-label">Động cơ chạy</div>
                        </div>
                    </div>

                    <!-- Thanh phân bổ tỷ lệ lỗi -->
                    <div class="error-breakdown" id="error-breakdown-box" style="display:none;">
                        <div style="font-size:0.82rem; font-weight:700; color:var(--text-muted); display:flex; justify-content:space-between;">
                            <span>📊 PHÂN BỔ LOẠI LỖI CHÍNH TẢ PHÁT HIỆN:</span>
                            <span id="total-error-text">0 lỗi</span>
                        </div>
                        <div class="breakdown-bar" id="breakdown-bar"></div>
                        <div class="breakdown-labels" id="breakdown-labels"></div>
                    </div>

                    <!-- Popover tương tác khi bấm vào từ lỗi -->
                    <div class="popover-card" id="popover-card">
                        <div style="display:flex; align-items:center; gap:0.75rem;">
                            <span style="font-size:1.2rem;">💡</span>
                            <div>
                                <div style="font-size:0.92rem;">
                                    Thay thế: <del id="pop-orig" style="color:var(--accent-red); font-weight:700;"></del> ➔ 
                                    <strong id="pop-corr" style="color:var(--accent-green); font-size:1.05rem;"></strong>
                                </div>
                                <div id="pop-category" style="margin-top:0.2rem;"></div>
                            </div>
                        </div>
                        <div style="display:flex; gap:0.5rem;">
                            <button class="btn-primary" style="padding:0.4rem 0.9rem; font-size:0.82rem;" onclick="applySingleFix()">
                                ✓ Áp dụng thay thế
                            </button>
                            <button class="btn-secondary" style="padding:0.4rem 0.75rem; font-size:0.82rem;" onclick="closePopover()">
                                ✕ Bỏ qua
                            </button>
                        </div>
                    </div>

                    <!-- Hiển thị đối chiếu song song -->
                    <div class="dual-view">
                        <div class="view-panel">
                            <div class="view-header red">
                                <span>❌ Văn bản gốc (Nhấp vào từ gạch chân để sửa)</span>
                                <span id="error-badge-count" style="font-size:0.75rem;">0 lỗi</span>
                            </div>
                            <div class="text-display" id="annotated-original"></div>
                        </div>
                        <div class="view-panel">
                            <div class="view-header green">
                                <span>✅ Văn bản đã sửa chuẩn xác</span>
                                <div style="display:flex; gap:0.4rem;">
                                    <button class="btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="applyAllFixes()">
                                        ✨ Dán lên ô nhập
                                    </button>
                                    <button class="btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="copyCorrected()">
                                        📋 Sao chép
                                    </button>
                                    <button class="btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="downloadTxt()">
                                        💾 Tải TXT
                                    </button>
                                </div>
                            </div>
                            <div class="text-display" id="annotated-corrected"></div>
                        </div>
                    </div>

                    <!-- Bảng chi tiết từng lỗi -->
                    <div id="details-table-wrap" style="display:none;">
                        <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.6rem; color:var(--text);">📋 Bảng Chi Tiết Từng Vị Trí Sai & Đề Xuất Sửa Đổi:</h4>
                        <div class="table-responsive">
                            <table class="details-table">
                                <thead>
                                    <tr>
                                        <th>Vị trí</th>
                                        <th>Từ viết sai (Input)</th>
                                        <th>Từ chuẩn hóa (ViT5 Target)</th>
                                        <th>Phân loại lỗi</th>
                                        <th>Thao tác</th>
                                    </tr>
                                </thead>
                                <tbody id="details-tbody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================= TAB 2: SO SÁNH SONG SONG ================= -->
        <div id="tab-compare" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">⚡ So Sánh Trực Tiếp: ViT5 Deep Learning vs Rule Engine Truyền Thống</span>
                </div>
                <p style="color:var(--text-muted); font-size:0.92rem; margin-bottom:1rem; line-height:1.5;">
                    Đối chiếu khả năng xử lý ngữ cảnh sâu giữa mô hình Học sâu <strong>ViT5 Transformer</strong> và phương pháp truyền thống <strong>Rule-based & N-gram</strong>.
                </p>
                <div class="editor-wrap">
                    <textarea id="compare-input" placeholder="Nhập câu để đối chiếu cả 2 mô hình...">Nhà trường ngày từ đầu năm học đã có hướng dẫn kế hoạch năm học của Sở giáo dục.</textarea>
                </div>
                <div class="action-row">
                    <button class="btn-primary" onclick="runComparison()">
                        ⚡ So Sánh Cả 2 Động Cơ Cùng Lúc
                    </button>
                </div>

                <div class="compare-grid" id="compare-results" style="display:none;">
                    <div class="compare-box">
                        <div class="compare-title" style="color:#818cf8;">
                            <span>🧠 1. ViT5 Deep Learning (Transformer)</span>
                            <span id="cmp-ai-time" style="font-size:0.75rem; color:var(--text-muted);"></span>
                        </div>
                        <div id="cmp-ai-text" style="font-size:1.05rem; line-height:1.8;"></div>
                        <p style="font-size:0.76rem; color:var(--text-muted); margin-top:0.6rem;">
                            ✓ Tự động phân tích quan hệ phụ thuộc cú pháp toàn câu bằng cơ chế Self-Attention.
                        </p>
                    </div>
                    <div class="compare-box">
                        <div class="compare-title" style="color:#38bdf8;">
                            <span>⚙️ 2. Rule Engine (Từ Điển 74K & N-gram)</span>
                            <span id="cmp-rule-time" style="font-size:0.75rem; color:var(--text-muted);"></span>
                        </div>
                        <div id="cmp-rule-text" style="font-size:1.05rem; line-height:1.8;"></div>
                        <p style="font-size:0.76rem; color:var(--text-muted); margin-top:0.6rem;">
                            ✓ Tra cứu từ điển và khoảng cách ký tự Levenshtein nhanh nhưng khó bắt các từ đồng âm khác nghĩa.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================= TAB 3: BỘ SINH NHIỄU MỞ RỘNG DỮ LIỆU ================= -->
        <div id="tab-noise" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">⚙️ Bộ Sinh Lỗi Nhân Tạo (Data Augmentation & Stress Testing)</span>
                </div>
                <p style="color:var(--text-muted); font-size:0.92rem; margin-bottom:1rem; line-height:1.5;">
                    Công cụ mô phỏng 6 cơ chế lỗi chính tả phổ biến của người Việt (âm đầu địa phương, dấu thanh, gõ bàn phím Telex, bàn phím QWERTY lân cận) để kiểm thử độ bền bỉ (robustness) của mô hình.
                </p>
                <textarea id="noise-clean-text" placeholder="Nhập một câu tiếng Việt sạch...">Hôm nay trời rất trong xanh, tôi ngồi đọc sách tại thư viện trường Đại học Bách Khoa.</textarea>
                
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:0.6rem; margin:1rem 0;">
                    <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; color:var(--text);">
                        <input type="checkbox" id="chk-telex" checked style="accent-color:var(--primary);"> Lỗi Telex / VNI (quến, sasch)
                    </label>
                    <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; color:var(--text);">
                        <input type="checkbox" id="chk-regional" checked style="accent-color:var(--primary);"> Lỗi âm đầu (ch/tr, s/x, d/gi/r)
                    </label>
                    <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; color:var(--text);">
                        <input type="checkbox" id="chk-tone" checked style="accent-color:var(--primary);"> Lỗi hỏi / ngã & rụng dấu
                    </label>
                    <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; color:var(--text);">
                        <input type="checkbox" id="chk-qwerty" checked style="accent-color:var(--primary);"> Phím QWERTY lân cận (a->s)
                    </label>
                </div>

                <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.25rem;">
                    <label style="font-size:0.85rem; color:var(--text-muted);">Tỷ lệ gây lỗi:</label>
                    <input type="range" id="noise-prob" min="10" max="60" value="30" style="flex:1; accent-color:var(--primary);">
                    <span id="noise-prob-val" style="font-family:'JetBrains Mono'; font-weight:700; color:#38bdf8;">30%</span>
                </div>

                <div class="action-row">
                    <button class="btn-primary" onclick="runGenerateNoise()">
                        🎲 Sinh Câu Lỗi Nhân Tạo
                    </button>
                </div>

                <div id="noise-output-box" style="display:none; margin-top:1.25rem;">
                    <div class="view-panel">
                        <div class="view-header red">
                            <span>Câu sau khi tạo lỗi nhân tạo:</span>
                            <button class="btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="sendToCorrector()">
                                🚀 Gửi sang Tab 1 để ViT5 sửa thử
                            </button>
                        </div>
                        <div class="text-display" id="noise-output-text"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================= TAB 4: BÁO CÁO THỰC NGHIỆM & LOSS ================= -->
        <div id="tab-report" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📊 Báo Cáo Thực Nghiệm & Đồ Thị Hội Tụ Loss</span>
                    <span class="tag-pill">Checkpoint: vit5_spelling_corrector/best_model</span>
                </div>
                <p style="color:var(--text-muted); font-size:0.92rem; margin-bottom:1rem; line-height:1.55;">
                    Kết quả dưới đây được đánh giá trên <strong>192 câu test "unseen"</strong> — các câu không nằm trong
                    tập huấn luyện của notebook. Toàn bộ 935 câu chỉ dùng để tham chiếu vì 743 câu bị trùng tập train.
                </p>

                <div class="stats-bar">
                    <div class="stat-card">
                        <div class="stat-value">58.33%</div>
                        <div class="stat-label">Exact Match (EM)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value fixed">95.05%</div>
                        <div class="stat-label">Token Accuracy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value latency">96.02</div>
                        <div class="stat-label">BLEU-4</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value fixed">1.53%</div>
                        <div class="stat-label">CER (càng thấp càng tốt)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value fixed">2.43%</div>
                        <div class="stat-label">WER (càng thấp càng tốt)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value latency">65.14%</div>
                        <div class="stat-label">Detection F1</div>
                    </div>
                </div>

                <div class="chart-container">
                    <div style="font-weight:700; margin-bottom:0.5rem;">Đường cong huấn luyện (Train loss theo global step)</div>
                    <svg viewBox="0 0 640 230" width="100%" height="auto" role="img" aria-label="Đồ thị train loss">
                        <line x1="40" y1="190" x2="620" y2="190" stroke="var(--border)" stroke-width="1"/>
                        <line x1="40" y1="40" x2="40" y2="190" stroke="var(--border)" stroke-width="1"/>
                        <line x1="40" y1="115" x2="620" y2="115" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="4 4"/>
                        <line x1="40" y1="40" x2="620" y2="40" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="4 4"/>
                        <text x="34" y="194" fill="var(--text-muted)" font-size="10" text-anchor="end">0</text>
                        <text x="34" y="119" fill="var(--text-muted)" font-size="10" text-anchor="end">0.375</text>
                        <text x="34" y="44" fill="var(--text-muted)" font-size="10" text-anchor="end">0.75</text>
                        <polyline fill="none" stroke="#38bdf8" stroke-width="2.5"
                            points="138.9,53.5 237.9,167 336.8,172.2 435.8,175.5 534.7,176.7 600,180.9"/>
                        <circle cx="138.9" cy="53.5" r="3.5" fill="#38bdf8"/>
                        <circle cx="237.9" cy="167" r="3.5" fill="#38bdf8"/>
                        <circle cx="336.8" cy="172.2" r="3.5" fill="#38bdf8"/>
                        <circle cx="435.8" cy="175.5" r="3.5" fill="#38bdf8"/>
                        <circle cx="534.7" cy="176.7" r="3.5" fill="#38bdf8"/>
                        <circle cx="600" cy="180.9" r="4.5" fill="#22c55e"/>
                        <text x="138.9" y="45" fill="var(--text-muted)" font-size="9" text-anchor="middle">0.683</text>
                        <text x="237.9" y="159" fill="var(--text-muted)" font-size="9" text-anchor="middle">0.115</text>
                        <text x="336.8" y="164" fill="var(--text-muted)" font-size="9" text-anchor="middle">0.089</text>
                        <text x="435.8" y="167" fill="var(--text-muted)" font-size="9" text-anchor="middle">0.073</text>
                        <text x="534.7" y="168" fill="var(--text-muted)" font-size="9" text-anchor="middle">0.067</text>
                        <text x="600" y="172" fill="#22c55e" font-size="9" text-anchor="middle">eval 0.046</text>
                        <text x="40" y="212" fill="var(--text-muted)" font-size="9" text-anchor="middle">0</text>
                        <text x="237.9" y="212" fill="var(--text-muted)" font-size="9" text-anchor="middle">100</text>
                        <text x="435.8" y="212" fill="var(--text-muted)" font-size="9" text-anchor="middle">200</text>
                        <text x="600" y="212" fill="var(--text-muted)" font-size="9" text-anchor="middle">283</text>
                        <text x="330" y="228" fill="var(--text-muted)" font-size="9" text-anchor="middle">Global step</text>
                    </svg>
                </div>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem;">
                    <div class="compare-box">
                        <div class="compare-title" style="color:#818cf8;">Cấu hình huấn luyện</div>
                        <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Backbone</td><td style="text-align:right;">VietAI/vit5-base</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Epochs</td><td style="text-align:right;">1</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Global step</td><td style="text-align:right;">283</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Batch × GradAccum</td><td style="text-align:right;">16 × 2</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Learning rate</td><td style="text-align:right;">2e-4</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Max length</td><td style="text-align:right;">96 token</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Train loss cuối</td><td style="text-align:right;">0.0666</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Eval loss cuối</td><td style="text-align:right;">0.0456</td></tr>
                            <tr><td style="padding:0.25rem 0; color:var(--text-muted);">Thời gian (CPU)</td><td style="text-align:right;">~99 phút</td></tr>
                        </table>
                    </div>
                    <div class="compare-box">
                        <div class="compare-title" style="color:#38bdf8;">ViT5 vs Rule Engine (192 câu unseen)</div>
                        <table style="width:100%; font-size:0.82rem; border-collapse:collapse;">
                            <tr style="color:var(--text-muted);"><th style="text-align:left; padding:0.3rem 0;">Chỉ số</th><th style="text-align:right;">ViT5</th><th style="text-align:right;">Rule</th></tr>
                            <tr><td style="padding:0.3rem 0;">Exact Match</td><td style="text-align:right; color:#22c55e; font-weight:700;">58.33%</td><td style="text-align:right;">6.77%</td></tr>
                            <tr><td style="padding:0.3rem 0;">Token Accuracy</td><td style="text-align:right; color:#22c55e; font-weight:700;">95.05%</td><td style="text-align:right;">92.27%</td></tr>
                            <tr><td style="padding:0.3rem 0;">BLEU-4</td><td style="text-align:right; color:#22c55e; font-weight:700;">96.02</td><td style="text-align:right;">85.72</td></tr>
                            <tr><td style="padding:0.3rem 0;">CER</td><td style="text-align:right; color:#22c55e; font-weight:700;">1.53%</td><td style="text-align:right;">1.97%</td></tr>
                            <tr><td style="padding:0.3rem 0;">WER</td><td style="text-align:right; color:#22c55e; font-weight:700;">2.43%</td><td style="text-align:right;">6.47%</td></tr>
                            <tr><td style="padding:0.3rem 0;">Detection F1</td><td style="text-align:right; color:#22c55e; font-weight:700;">65.14%</td><td style="text-align:right;">25.89%</td></tr>
                        </table>
                    </div>
                </div>

                <div style="margin-top:1rem; font-size:0.82rem; color:var(--text-muted); line-height:1.55; border-left:3px solid #f59e0b; padding-left:0.75rem;">
                    <strong>Lưu ý về dữ liệu:</strong> notebook huấn luyện trộn 2.000 câu sạch với 7.472 câu VSEC trước khi chia split,
                    khiến 743/935 câu test trùng id với tập train. Vì vậy báo cáo chính dùng 192 câu unseen; kết quả trên 935 câu
                    (EM 73.48%, BLEU 97.21) chỉ mang tính tham chiếu và không nên trích dẫn như kết quả sạch.
                </div>
            </div>
        </div>

    <!-- Toast Notification -->
    <div class="toast" id="toast">📋 Đã sao chép vào bộ nhớ tạm!</div>

    <script>
        // Theme toggle
        function toggleTheme() {
            const html = document.documentElement;
            const current = html.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            document.getElementById('theme-icon').innerText = next === 'dark' ? '☀️' : '🌙';
            document.getElementById('theme-label').innerText = next === 'dark' ? 'Giao diện Sáng' : 'Giao diện Tối';
        }

        // Tab switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
            if (targetBtn) targetBtn.classList.add('active');

            const content = document.getElementById('tab-' + tabId);
            if (content) content.classList.add('active');
        }

        // Deep-link tabs: /#corrector | /#compare | /#noise | /#report
        document.addEventListener('DOMContentLoaded', function () {
            const hash = (location.hash || '').replace('#', '');
            if (hash) switchTab(hash);
        });

        // Char / Word counter
        const textarea = document.getElementById('input-text');
        const counter = document.getElementById('char-counter');
        function updateCounter() {
            const text = textarea.value.trim();
            const words = text ? text.split(/\\s+/).length : 0;
            const chars = textarea.value.length;
            counter.innerText = `${words} từ | ${chars} ký tự`;
        }
        textarea.addEventListener('input', updateCounter);
        textarea.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                runCorrection();
            }
        });
        updateCounter();

        // Sample picker
        const samples = {
            1: "Hôm nai chời mưa to qúa, tôi quến mang rù đi họt.",
            2: "Tối nay tôi tự nấu ăn ở nhà để tiet kiệm fiền.",
            3: "Trường Đại học Bách Khoa có bề dày lịch sử và truyền thống vẻ vang.",
            4: "Thời tiếc miền bắc mùa đông xe lạnh và có mưa fùn nhẹ.",
            5: "Mô hình uọc sâu cà ma5ng nơ-ronj biên đổi cách fon người giải quyết bài toán.",
            6: "Chuẩn bị kỹ lưởng trước khi bươc vào kỳ thi đại hoc quan trọng."
        };
        function setSample(id) {
            textarea.value = samples[id] || "";
            updateCounter();
            runCorrection();
        }

        function clearEditor() {
            textarea.value = "";
            updateCounter();
            document.getElementById('results-box').style.display = 'none';
            closePopover();
        }

        let currentResult = null;
        let selectedError = null;

        // Sửa lỗi chính
        async function runCorrection() {
            const text = textarea.value.trim();
            if (!text) return alert("Vui lòng nhập văn bản!");

            const mode = document.querySelector('input[name="engine"]:checked').value;
            const btn = document.getElementById('btn-correct');
            const btnText = document.getElementById('btn-text');
            const btnSpinner = document.getElementById('btn-spinner');

            btn.disabled = true;
            btnText.style.display = 'none';
            btnSpinner.style.display = 'inline';
            closePopover();

            const startTime = performance.now();

            try {
                const res = await fetch('/api/correct_spelling', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text, mode })
                });
                const data = await res.json();
                const latency = Math.round(performance.now() - startTime);

                currentResult = data;
                displayResults(data, latency, mode);
            } catch (err) {
                alert("Lỗi khi kết nối máy chủ: " + err);
            } finally {
                btn.disabled = false;
                btnText.style.display = 'inline';
                btnSpinner.style.display = 'none';
            }
        }

        function displayResults(data, latency, mode) {
            const box = document.getElementById('results-box');
            box.style.display = 'block';

            const errors = data.errors_found || [];
            document.getElementById('stat-error-count').innerText = errors.length;
            document.getElementById('stat-fixed-count').innerText = errors.length;
            document.getElementById('stat-latency').innerText = latency + ' ms';
            document.getElementById('stat-engine').innerText = (mode === 'ai' ? 'ViT5 Deep Learning' : (mode === 'hybrid' ? 'ViT5 Hybrid' : 'Rule Engine'));
            document.getElementById('error-badge-count').innerText = errors.length + ' lỗi';

            // Error breakdown bar
            renderBreakdown(errors);

            // Annotated Original Text
            const origTokens = (data.original_text || "").split(' ');
            const corrTokens = (data.corrected_text || "").split(' ');

            let origHtml = "";
            let errorTokenMap = new Map();
            errors.forEach((e, errIdx) => {
                if (e.original_indices) {
                    for (let idx = e.original_indices[0]; idx < e.original_indices[1]; idx++) {
                        errorTokenMap.set(idx, { err: e, errIdx: errIdx });
                    }
                }
            });

            origTokens.forEach((tok, idx) => {
                if (errorTokenMap.has(idx)) {
                    const item = errorTokenMap.get(idx);
                    origHtml += `<span class="error-token" onclick="openPopover(${item.errIdx})" title="Nhấp để xem gợi ý: ${tok}">${tok}</span> `;
                } else {
                    origHtml += tok + " ";
                }
            });
            document.getElementById('annotated-original').innerHTML = origHtml.trim();

            // Annotated Corrected Text
            let corrHtml = "";
            let fixedSet = new Set();
            errors.forEach(e => {
                if (e.corrected_indices) {
                    for (let idx = e.corrected_indices[0]; idx < e.corrected_indices[1]; idx++) {
                        fixedSet.add(idx);
                    }
                }
            });

            corrTokens.forEach((tok, idx) => {
                if (fixedSet.has(idx)) {
                    corrHtml += `<span class="fixed-token">${tok}</span> `;
                } else {
                    corrHtml += tok + " ";
                }
            });
            document.getElementById('annotated-corrected').innerHTML = corrHtml.trim();

            // Table details
            const tableWrap = document.getElementById('details-table-wrap');
            const tbody = document.getElementById('details-tbody');
            if (errors.length > 0) {
                tableWrap.style.display = 'block';
                tbody.innerHTML = errors.map((err, i) => {
                    const cat = err.category || { label: "Lỗi chính tả", badge: "badge-other", icon: "✏️" };
                    return `
                        <tr>
                            <td style="font-family:'JetBrains Mono'; font-weight:600; color:var(--text-muted);">#${i + 1}</td>
                            <td><span style="color:var(--accent-red); font-weight:700; text-decoration:line-through;">${err.original_segment || ''}</span></td>
                            <td><span style="color:var(--accent-green); font-weight:700;">${err.corrected_segment || ''}</span></td>
                            <td><span class="badge-pill ${cat.badge}">${cat.icon} ${cat.label}</span></td>
                            <td>
                                <button class="btn-secondary" style="padding:0.2rem 0.6rem; font-size:0.75rem;" onclick="openPopover(${i})">
                                    🔍 Xem & Áp dụng
                                </button>
                            </td>
                        </tr>
                    `;
                }).join('');
            } else {
                tableWrap.style.display = 'none';
            }

            box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function renderBreakdown(errors) {
            const breakdownBox = document.getElementById('error-breakdown-box');
            if (!errors || errors.length === 0) {
                breakdownBox.style.display = 'none';
                return;
            }
            breakdownBox.style.display = 'block';
            document.getElementById('total-error-text').innerText = `${errors.length} lỗi tìm thấy`;

            const counts = {};
            errors.forEach(e => {
                const type = (e.category && e.category.label) || 'Khác';
                counts[type] = (counts[type] || 0) + 1;
            });

            const colorMap = {
                "Lỗi âm đầu phương ngữ": "#f59e0b",
                "Lỗi âm cuối phương ngữ": "#fbbf24",
                "Lỗi dấu Hỏi / Ngã": "#3b82f6",
                "Lỗi rụng dấu / Dấu thanh": "#60a5fa",
                "Lỗi gõ phím Telex / VNI": "#8b5cf6",
                "Lỗi đảo vị trí ký tự": "#ec4899",
                "Lỗi từ vựng & Ngữ cảnh": "#10b981"
            };

            let barHtml = "";
            let labelHtml = "";
            for (const [label, count] of Object.entries(counts)) {
                const pct = Math.round((count / errors.length) * 100);
                const color = colorMap[label] || "#94a3b8";
                barHtml += `<div class="breakdown-seg" style="width:${pct}%; background:${color};" title="${label}: ${count} (${pct}%)"></div>`;
                labelHtml += `
                    <div class="breakdown-item">
                        <span class="breakdown-dot" style="background:${color};"></span>
                        <span>${label}: <strong>${count}</strong> (${pct}%)</span>
                    </div>
                `;
            }
            document.getElementById('breakdown-bar').innerHTML = barHtml;
            document.getElementById('breakdown-labels').innerHTML = labelHtml;
        }

        // Popover handlers
        function openPopover(errIdx) {
            if (!currentResult || !currentResult.errors_found) return;
            const err = currentResult.errors_found[errIdx];
            if (!err) return;

            selectedError = err;
            const pop = document.getElementById('popover-card');
            document.getElementById('pop-orig').innerText = err.original_segment;
            document.getElementById('pop-corr').innerText = err.corrected_segment;

            const cat = err.category || { label: "Lỗi chính tả", badge: "badge-other", icon: "✏️" };
            document.getElementById('pop-category').innerHTML = `<span class="badge-pill ${cat.badge}">${cat.icon} ${cat.label}</span>`;

            pop.classList.add('active');
            pop.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function closePopover() {
            document.getElementById('popover-card').classList.remove('active');
            selectedError = null;
        }

        function applySingleFix() {
            if (!selectedError) return;
            const orig = selectedError.original_segment;
            const corr = selectedError.corrected_segment;
            textarea.value = textarea.value.replace(orig, corr);
            updateCounter();
            closePopover();
            runCorrection();
            showToast(`✓ Đã áp dụng thay thế: ${orig} ➔ ${corr}`);
        }

        function applyAllFixes() {
            if (!currentResult || !currentResult.corrected_text) return;
            textarea.value = currentResult.corrected_text;
            updateCounter();
            closePopover();
            showToast("✨ Đã cập nhật văn bản chuẩn lên ô soạn thảo!");
            runCorrection();
        }

        function copyCorrected() {
            const text = document.getElementById('annotated-corrected').innerText;
            navigator.clipboard.writeText(text).then(() => {
                showToast("📋 Đã sao chép văn bản đã sửa vào Clipboard!");
            });
        }

        function downloadTxt() {
            const text = document.getElementById('annotated-corrected').innerText;
            if (!text) return alert("Không có nội dung để tải về!");
            const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "van_ban_da_sua_chinh_ta.txt";
            a.click();
            URL.revokeObjectURL(url);
            showToast("💾 Đã tải file van_ban_da_sua_chinh_ta.txt!");
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.style.display = 'block';
            setTimeout(() => { toast.style.display = 'none'; }, 2600);
        }

        // So sánh song song
        async function runComparison() {
            const text = document.getElementById('compare-input').value.trim();
            if (!text) return alert("Vui lòng nhập câu cần so sánh!");

            const cmpBox = document.getElementById('compare-results');
            cmpBox.style.display = 'grid';

            document.getElementById('cmp-ai-text').innerHTML = "⏳ Đang giải mã ViT5 Deep Learning...";
            document.getElementById('cmp-rule-text').innerHTML = "⏳ Đang chạy Rule Engine...";

            try {
                const res = await fetch('/api/compare_engines', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text })
                });
                const data = await res.json();

                document.getElementById('cmp-ai-text').innerText = data.ai.corrected_text;
                document.getElementById('cmp-ai-time').innerText = `(${data.ai.latency_ms} ms)`;

                document.getElementById('cmp-rule-text').innerText = data.rule.corrected_text;
                document.getElementById('cmp-rule-time').innerText = `(${data.rule.latency_ms} ms)`;
            } catch (err) {
                alert("Lỗi khi so sánh: " + err);
            }
        }

        // Sinh lỗi nhân tạo
        const noiseSlider = document.getElementById('noise-prob');
        const noiseVal = document.getElementById('noise-prob-val');
        noiseSlider.addEventListener('input', () => { noiseVal.innerText = noiseSlider.value + '%'; });

        async function runGenerateNoise() {
            const text = document.getElementById('noise-clean-text').value.trim();
            if (!text) return alert("Vui lòng nhập câu sạch!");

            const prob = parseFloat(noiseSlider.value) / 100.0;
            const res = await fetch('/api/generate_noise', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text, noise_prob: prob })
            });
            const data = await res.json();

            document.getElementById('noise-output-box').style.display = 'block';
            document.getElementById('noise-output-text').innerText = data.noisy_text;
        }

        function sendToCorrector() {
            const noisyText = document.getElementById('noise-output-text').innerText;
            textarea.value = noisyText;
            updateCounter();
            switchTab('corrector');
            runCorrection();
        }
    </script>
</body>
</html>
"""


class SpellCheckerRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Xử lý các HTTP REST API và phục vụ giao diện Web chuyên nghiệp."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self._set_headers(200, "text/html")
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/api/status":
            self._set_headers(200)
            status_data = {
                "status": "ready",
                "ai_loaded": smart_corrector.use_ai,
                "model_name": "VietAI/vit5-base",
                "checkpoint": "checkpoints/vit5_spelling_corrector/best_model",
                "architecture": "Transformer Seq2Seq (Encoder-Decoder)"
            }
            self.wfile.write(json.dumps(status_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            req = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            req = {}

        if parsed.path == "/api/correct_spelling":
            text = req.get("text", "")
            mode = req.get("mode", "ai")

            start_t = time.time()
            result = smart_corrector.check_and_correct(text, mode=mode)
            latency_ms = round((time.time() - start_t) * 1000)

            # Phân loại chuyên sâu các vị trí lỗi
            if result.get("errors_found"):
                for err in result["errors_found"]:
                    err["category"] = categorize_error(
                        err.get("original_segment", ""),
                        err.get("corrected_segment", "")
                    )

            result["latency_ms"] = latency_ms
            self._set_headers(200)
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

        elif parsed.path == "/api/compare_engines":
            text = req.get("text", "")

            # 1. ViT5 AI
            t0 = time.time()
            res_ai = smart_corrector.check_and_correct(text, mode="ai")
            ai_time = round((time.time() - t0) * 1000)
            res_ai["latency_ms"] = ai_time

            # 2. Rule
            t1 = time.time()
            res_rule = smart_corrector.check_and_correct(text, mode="rule")
            rule_time = round((time.time() - t1) * 1000)
            res_rule["latency_ms"] = rule_time

            response_data = {
                "ai": res_ai,
                "rule": res_rule
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

        elif parsed.path == "/api/generate_noise":
            text = req.get("text", "")
            prob = float(req.get("noise_prob", 0.3))

            noisy = noise_gen.add_noise(text, noise_prob=prob)
            diffs = noise_gen.find_differences(text, noisy)

            response_data = {
                "clean_text": text,
                "noisy_text": noisy,
                "noise_prob": prob,
                "errors_count": len(diffs),
                "differences": diffs
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Ẩn bớt HTTP log để terminal luôn sạch đẹp
        pass


def run_app(port: int = 7860):
    handler = SpellCheckerRequestHandler
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}"
            print("=" * 65)
            print("   🚀 HỆ THỐNG WEB SỬA LỖI CHÍNH TẢ TIẾNG VIỆT (ViT5 TRANSFORMER)")
            print(f"   🌐 Đang mở tại: {url}")
            print("   (Nhấn Ctrl + C để dừng máy chủ)")
            print("=" * 65)
            try:
                webbrowser.open(url)
            except Exception:
                pass
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e) or getattr(e, "errno", None) in [98, 10048]:
            alt_port = port + 1
            print(f"[!] Cổng {port} đang bận, thử đổi sang cổng {alt_port}...")
            run_app(alt_port)
        else:
            raise e


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chạy giao diện Web Sửa lỗi chính tả Tiếng Việt.")
    parser.add_argument("--port", type=int, default=7860, help="Cổng chạy server (mặc định 7860)")
    args = parser.parse_args()

    run_app(port=args.port)
