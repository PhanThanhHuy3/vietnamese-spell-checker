"""Giao diện Web tương tác cho Hệ thống Sửa lỗi chính tả & Sinh lỗi Tiếng Việt.
Chạy độc lập bằng thư viện chuẩn Python (không yêu cầu cài thêm thư viện để mở UI).
Tự động tích hợp mô hình Deep Learning (ViT5/PhoBERT) khi có PyTorch.
"""

import http.server
import json
import os
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Dict

# Đảm bảo import được module từ src/
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.noise_generator import VietnameseNoiseGenerator
from src.smart_corrector import SmartVietnameseCorrector

# Khởi tạo bộ sửa lỗi thông minh và bộ sinh lỗi
smart_corrector = SmartVietnameseCorrector()
noise_gen = VietnameseNoiseGenerator(seed=42)
AI_MODEL_AVAILABLE = smart_corrector.use_ai

# ==============================================================================
# HTML / CSS / JS GIAO DIỆN HIỆN ĐẠI
# ==============================================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ Thống Sửa Lỗi Chính Tả Tiếng Việt - AI NLP</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg: #0b0f19;
            --card-bg: #151c2e;
            --card-border: #263352;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-red: #ef4444;
            --accent-green: #10b981;
            --accent-orange: #f59e0b;
            --accent-purple: #8b5cf6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 1050px;
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .badge-ai {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(79, 70, 229, 0.15);
            border: 1px solid rgba(79, 70, 229, 0.4);
            color: #a5b4fc;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        /* Navigation Tabs */
        .tab-nav {
            display: flex;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.35rem;
            margin-bottom: 1.5rem;
            gap: 0.35rem;
        }

        .tab-btn {
            flex: 1;
            padding: 0.75rem 1.25rem;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.95rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .tab-btn.active {
            background: var(--primary);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
        }

        /* Cards */
        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .card-title {
            font-size: 1.2rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Form Controls */
        label {
            display: block;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        textarea {
            width: 100%;
            height: 120px;
            background: #0d121f;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            color: var(--text-main);
            font-size: 1rem;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s ease;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
        }

        /* Quick Samples */
        .samples {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0 1.25rem 0;
            align-items: center;
        }

        .sample-label {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .sample-pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: #cbd5e1;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .sample-pill:hover {
            background: rgba(79, 70, 229, 0.2);
            border-color: var(--primary);
            color: #fff;
        }

        /* Config Grid */
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            background: #0d121f;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid var(--card-border);
            margin-bottom: 1.25rem;
        }

        .config-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            cursor: pointer;
        }

        .config-item input[type="checkbox"] {
            accent-color: var(--primary);
            width: 17px;
            height: 17px;
            cursor: pointer;
        }

        .slider-box {
            grid-column: 1 / -1;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .slider-box input[type="range"] {
            flex: 1;
            accent-color: var(--primary);
        }

        /* Actions */
        .action-row {
            display: flex;
            justify-content: flex-end;
            gap: 0.75rem;
        }

        button.btn-primary {
            background: var(--primary);
            color: #fff;
            border: none;
            padding: 0.85rem 1.75rem;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 10px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }

        button.btn-primary:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        }

        button.btn-secondary {
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
            border: 1px solid var(--card-border);
            padding: 0.85rem 1.25rem;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }

        button.btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        /* Output Box */
        .output-box {
            margin-top: 1.5rem;
            background: #0d121f;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
        }

        .output-text {
            font-size: 1.15rem;
            line-height: 1.7;
            margin-bottom: 1rem;
            word-break: break-word;
        }

        .highlight-err {
            background: rgba(239, 68, 68, 0.25);
            color: #fca5a5;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            border-bottom: 2px solid var(--accent-red);
            font-weight: 600;
        }

        .highlight-fixed {
            background: rgba(16, 185, 129, 0.2);
            color: #6ee7b7;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            border-bottom: 2px solid var(--accent-green);
            font-weight: 600;
        }

        .diff-tag {
            text-decoration: line-through;
            color: #f87171;
            margin-right: 0.25rem;
        }

        /* Table */
        table.detail-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 1rem;
        }

        table.detail-table th, table.detail-table td {
            text-align: left;
            padding: 0.65rem 0.85rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }

        table.detail-table th {
            color: var(--text-muted);
            font-weight: 600;
            background: rgba(0, 0, 0, 0.2);
        }

        .badge-rule {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(139, 92, 246, 0.2);
            color: #c4b5fd;
        }

        .alert-info {
            background: rgba(79, 70, 229, 0.1);
            border: 1px solid rgba(79, 70, 229, 0.3);
            border-radius: 10px;
            padding: 1rem;
            color: #c7d2fe;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            line-height: 1.5;
        }

        .code-box {
            font-family: 'JetBrains Mono', monospace;
            background: #060911;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            color: #38bdf8;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge-ai">🇻🇳 NLP Deep Learning Project</div>
            <h1>Phát Hiện & Sửa Lỗi Chính Tả Tiếng Việt</h1>
            <p class="subtitle">Ứng dụng mô hình Transformers (ViT5) & Bộ sinh nhiễu lỗi đa quy luật</p>
        </header>

        <!-- Navigation -->
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('noise')">
                ⚙️ Tab 1: Bộ Sinh Nhiễu (Noise Generator)
            </button>
            <button class="tab-btn" onclick="switchTab('corrector')">
                ✨ Tab 2: Sửa Lỗi Chính Tả (AI Corrector)
            </button>
        </div>

        <!-- ================= TAB 1: NOISE GENERATOR ================= -->
        <div id="tab-noise" class="tab-content active">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📝 Nhập câu tiếng Việt sạch (Ground Truth)</span>
                </div>

                <textarea id="clean-text" placeholder="Nhập một câu tiếng Việt chuẩn để tạo lỗi nhân tạo...">Hôm nay trời rất trong xanh, tôi ngồi đọc sách nghiên cứu về tiếng Việt tại thư viện.</textarea>

                <div class="samples">
                    <span class="sample-label">Gợi ý mẫu:</span>
                    <span class="sample-pill" onclick="setCleanSample(1)">Bách Khoa</span>
                    <span class="sample-pill" onclick="setCleanSample(2)">Học sâu AI</span>
                    <span class="sample-pill" onclick="setCleanSample(3)">Thời tiết mùa đông</span>
                    <span class="sample-pill" onclick="setCleanSample(4)">Phương ngữ & Dấu</span>
                </div>

                <label>Tuỳ chọn quy luật sinh lỗi:</label>
                <div class="config-grid">
                    <label class="config-item">
                        <input type="checkbox" id="rule-telex" checked>
                        <span>⌨️ Lỗi Telex/VNI (sasch, hoangf, sa1ch)</span>
                    </label>
                    <label class="config-item">
                        <input type="checkbox" id="rule-keyboard" checked>
                        <span>🔤 Phím QWERTY lân cận (n -> b, a -> s)</span>
                    </label>
                    <label class="config-item">
                        <input type="checkbox" id="rule-regional" checked>
                        <span>🗣️ Lỗi vùng miền (ch/tr, s/x, d/gi/r, ?/~)</span>
                    </label>
                    <label class="config-item">
                        <input type="checkbox" id="rule-diacritics" checked>
                        <span>🚫 Lỗi bỏ dấu (tieng viet)</span>
                    </label>
                    <div class="slider-box">
                        <label style="margin:0; white-space:nowrap;">Tỷ lệ lỗi: <strong id="rate-val" style="color:#a5b4fc;">35%</strong></label>
                        <input type="range" id="error-rate" min="10" max="80" value="35" oninput="document.getElementById('rate-val').innerText = this.value + '%'">
                    </div>
                </div>

                <div class="action-row">
                    <button class="btn-primary" onclick="generateNoise()">
                        ⚡ Tạo Lỗi Ngẫu Nhiên
                    </button>
                </div>

                <!-- Kết quả sinh lỗi -->
                <div id="noise-result" class="output-box" style="display:none;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <label style="margin:0;">Kết quả văn bản bị chèn lỗi:</label>
                        <button class="btn-secondary" style="padding:0.4rem 0.8rem; font-size:0.85rem;" onclick="copyToCorrector()">
                            ➡️ Chuyển sang Tab Sửa Lỗi
                        </button>
                    </div>
                    <div class="output-text" id="noisy-sentence-display"></div>
                    <div id="noise-details"></div>
                </div>
            </div>
        </div>

        <!-- ================= TAB 2: SPELL CORRECTOR ================= -->
        <div id="tab-corrector" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">🔍 Nhập câu tiếng Việt cần phát hiện & sửa lỗi</span>
                </div>

                <div id="ai-status-banner"></div>

                <textarea id="corrupt-text" placeholder="Nhập câu có chứa lỗi chính tả...">Hôm nai chời mưa to qúa, tôi quến mang ô đi hok.</textarea>

                <div class="samples">
                    <span class="sample-label">Câu lỗi mẫu:</span>
                    <span class="sample-pill" onclick="setCorruptSample(1)">Lỗi telex dính đuôi</span>
                    <span class="sample-pill" onclick="setCorruptSample(2)">Lỗi vùng miền ch/tr, s/x</span>
                    <span class="sample-pill" onclick="setCorruptSample(3)">Lỗi phím lân cận & mất dấu</span>
                </div>

                <div style="background:#0d121f; border:1px solid var(--card-border); border-radius:10px; padding:0.85rem 1rem; margin-bottom:1.25rem;">
                    <label style="margin-bottom:0.5rem; display:block;">Chọn động cơ sửa lỗi:</label>
                    <div style="display:flex; flex-wrap:wrap; gap:1.5rem;">
                        <label style="display:flex; align-items:center; gap:0.5rem; cursor:pointer; color:#a5b4fc; font-weight:600;">
                            <input type="radio" name="engine_mode" value="rule" checked style="accent-color:var(--primary);">
                            <span>⚡ Bộ Luật & Ngữ Âm Thông Minh (Khuyên dùng: Siêu tốc 0.01s, Chuẩn xác)</span>
                        </label>
                        <label style="display:flex; align-items:center; gap:0.5rem; cursor:pointer; color:#cbd5e1;">
                            <input type="radio" name="engine_mode" value="ai" style="accent-color:var(--primary);">
                            <span>🧠 Mô hình Deep Learning ViT5 (Cần GPU / Train đủ epochs)</span>
                        </label>
                    </div>
                </div>

                <div class="action-row">
                    <button id="btn-correct" class="btn-primary" onclick="correctSpelling()">
                        🚀 Kiểm Tra & Sửa Lỗi Ngay
                    </button>
                </div>

                <!-- Kết quả sửa lỗi -->
                <div id="correct-result" class="output-box" style="display:none;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <label style="margin:0;">Câu chuẩn sau khi sửa:</label>
                        <span id="engine-used-badge" class="badge-rule" style="background:rgba(16,185,129,0.2); color:#6ee7b7;"></span>
                    </div>
                    <div class="output-text" id="corrected-sentence-display"></div>
                    <div id="correct-details"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (tab === 'noise') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('tab-noise').classList.add('active');
            } else {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('tab-corrector').classList.add('active');
            }
        }

        const cleanSamples = {
            1: "Trường đại học Bách Khoa có bề dày lịch sử và truyền thống vẻ vang.",
            2: "Mô hình học sâu và mạng nơ-ron biến đổi cách con người giải quyết bài toán phức tạp.",
            3: "Thời tiết miền Bắc đang bước vào những ngày đầu đông se lạnh và có mưa phùn nhẹ.",
            4: "Tôi rất thích đọc sách nghiên cứu ngôn ngữ, giữ gìn sự trong sáng của tiếng Việt."
        };

        const corruptSamples = {
            1: "Hôm nay tôi ngồii đocj sasch tại thư viện thành phố.",
            2: "Chời hôm nay dất trong xanh, xạch xẽ và thoáng mát.",
            3: "truong dai hoc bach khoa ha noi dao tao ki su gioi."
        };

        function setCleanSample(idx) {
            document.getElementById('clean-text').value = cleanSamples[idx];
        }

        function setCorruptSample(idx) {
            document.getElementById('corrupt-text').value = corruptSamples[idx];
        }

        async function generateNoise() {
            const sentence = document.getElementById('clean-text').value.trim();
            if (!sentence) return;

            const errorTypes = [];
            if (document.getElementById('rule-telex').checked) errorTypes.push('telex_vni');
            if (document.getElementById('rule-keyboard').checked) errorTypes.push('keyboard');
            if (document.getElementById('rule-regional').checked) errorTypes.push('regional');
            if (document.getElementById('rule-diacritics').checked) errorTypes.push('remove_diacritics');

            const rate = parseInt(document.getElementById('error-rate').value) / 100.0;

            const res = await fetch('/api/generate_noise', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ sentence, error_rate: rate, error_types: errorTypes })
            });

            const data = await res.json();
            displayNoiseResult(data);
        }

        function displayNoiseResult(data) {
            const resultBox = document.getElementById('noise-result');
            const displayEl = document.getElementById('noisy-sentence-display');
            const detailsEl = document.getElementById('noise-details');

            resultBox.style.display = 'block';

            // Highlight các từ bị lỗi
            const corruptedWords = data.corrupted_sentence.split(' ');
            const modifiedIndices = new Set(data.modifications.map(m => m.token_index));

            let htmlSentence = corruptedWords.map((w, idx) => {
                if (modifiedIndices.has(idx)) {
                    return `<span class="highlight-err">${w}</span>`;
                }
                return w;
            }).join(' ');

            displayEl.innerHTML = htmlSentence;

            if (data.modifications.length > 0) {
                let tableHtml = `
                    <table class="detail-table">
                        <thead>
                            <tr>
                                <th>STT</th>
                                <th>Từ gốc chuẩn</th>
                                <th>Từ bị làm lỗi</th>
                                <th>Quy luật mô phỏng</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                data.modifications.forEach((m, i) => {
                    tableHtml += `
                        <tr>
                            <td>${i + 1}</td>
                            <td><strong style="color:#6ee7b7;">${m.original}</strong></td>
                            <td><span style="color:#fca5a5;">${m.corrupted}</span></td>
                            <td><span class="badge-rule">${m.rule}</span></td>
                        </tr>
                    `;
                });
                tableHtml += '</tbody></table>';
                detailsEl.innerHTML = tableHtml;
            } else {
                detailsEl.innerHTML = '<p style="color:#94a3b8; font-size:0.9rem;">Không có từ nào bị đổi (thử tăng tỷ lệ lỗi).</p>';
            }
        }

        function copyToCorrector() {
            const rawText = document.getElementById('noisy-sentence-display').innerText;
            document.getElementById('corrupt-text').value = rawText;
            switchTab('corrector');
        }

        async function correctSpelling() {
            const text = document.getElementById('corrupt-text').value.trim();
            if (!text) return;

            const btn = document.getElementById('btn-correct');
            const originalBtnHtml = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '⏳ Đang kiểm tra & sửa lỗi...';

            const selectedMode = document.querySelector('input[name="engine_mode"]:checked').value;

            try {
                const res = await fetch('/api/correct_spelling', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text, mode: selectedMode })
                });

                const data = await res.json();
                displayCorrectResult(data);
            } catch (err) {
                alert('Có lỗi xảy ra khi gọi server: ' + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalBtnHtml;
            }
        }

        function displayCorrectResult(data) {
            const resultBox = document.getElementById('correct-result');
            const displayEl = document.getElementById('corrected-sentence-display');
            const detailsEl = document.getElementById('correct-details');
            const engineBadge = document.getElementById('engine-used-badge');

            resultBox.style.display = 'block';
            if (engineBadge && data.engine) {
                engineBadge.innerText = '⚙️ ' + data.engine;
            }
            displayEl.innerHTML = `<span class="highlight-fixed">${data.corrected_text}</span>`;

            if (data.has_error && data.errors_found && data.errors_found.length > 0) {
                let tableHtml = `
                    <table class="detail-table">
                        <thead>
                            <tr>
                                <th>Vị trí sai</th>
                                <th>Từ lỗi (Original)</th>
                                <th>Gợi ý sửa đúng (Corrected)</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                data.errors_found.forEach(err => {
                    tableHtml += `
                        <tr>
                            <td>Từ ${err.original_indices[0] + 1}</td>
                            <td><span class="diff-tag">${err.original_segment || '(thiếu)'}</span></td>
                            <td><strong style="color:#10b981;">${err.corrected_segment || '(xóa)'}</strong></td>
                        </tr>
                    `;
                });
                tableHtml += '</tbody></table>';
                detailsEl.innerHTML = tableHtml;
            } else {
                detailsEl.innerHTML = '<p style="color:#10b981; font-weight:600; font-size:0.95rem; margin-top:0.5rem;">✓ Câu hoàn toàn chuẩn xác, không phát hiện lỗi chính tả.</p>';
            }
        }

        // Kiểm tra trạng thái AI Banner khi tải trang
        fetch('/api/status').then(r => r.json()).then(st => {
            const banner = document.getElementById('ai-status-banner');
            if (st.ai_available) {
                banner.innerHTML = `<div class="alert-info" style="border-color:#10b981; background:rgba(16,185,129,0.1); color:#6ee7b7;">
                    ✓ <strong>Mô hình Deep Learning AI (${st.model_name}) đã kích hoạt sẵn sàng!</strong> Đang sử dụng thiết bị: <span class="code-box">${st.device}</span>
                </div>`;
            } else {
                banner.innerHTML = `<div class="alert-info">
                    ℹ️ <strong>Chế độ Rules / Heuristic đang chạy:</strong> Để kích hoạt toàn diện mô hình học sâu ViT5 Transformer, bạn chỉ cần cài đặt thư viện bằng lệnh:
                    <br><span class="code-box">pip install -r requirements.txt</span> rồi huấn luyện <span class="code-box">python train.py</span>.
                </div>`;
            }
        });
    </script>
</body>
</html>
"""

# ==============================================================================
# HTTP REQUEST HANDLER
# ==============================================================================
class SpellCheckerRequestHandler(http.server.BaseHTTPRequestHandler):
    """Bộ xử lý HTTP request phục vụ UI và API REST."""

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
        if parsed.path == "/" or parsed.path == "/index.html":
            self._set_headers(200, "text/html")
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/api/status":
            self._set_headers(200, "application/json")
            status_data = {
                "ai_available": smart_corrector.use_ai,
                "engine": "ViT5 Deep Learning" if smart_corrector.use_ai else "Smart Vietnamese Rule & Phonetic Engine",
                "device": "CUDA/CPU" if smart_corrector.use_ai else "N/A",
            }
            self.wfile.write(json.dumps(status_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            req = json.loads(post_data.decode("utf-8"))
        except Exception:
            req = {}

        # 1. API SINH NHIỄU (NOISE GENERATOR)
        if parsed.path == "/api/generate_noise":
            sentence = req.get("sentence", "")
            error_rate = float(req.get("error_rate", 0.3))
            error_types = req.get("error_types", None)

            details = noise_gen.generate_with_details(
                sentence=sentence,
                error_rate=error_rate,
                error_types=error_types,
            )
            self._set_headers(200)
            self.wfile.write(json.dumps(details, ensure_ascii=False).encode("utf-8"))

        # 2. API SỬA LỖI (SPELL CHECKER)
        elif parsed.path == "/api/correct_spelling":
            text = req.get("text", "")
            mode = req.get("mode", "rule")
            if mode == "rule":
                try:
                    import importlib
                    import src.smart_corrector
                    importlib.reload(src.smart_corrector)
                    rule_corrector = src.smart_corrector.SmartVietnameseCorrector()
                    result = rule_corrector.check_and_correct(text, mode=mode)
                except Exception:
                    result = smart_corrector.check_and_correct(text, mode=mode)
            else:
                result = smart_corrector.check_and_correct(text, mode=mode)
            self._set_headers(200)
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Ẩn bớt log request để console sạch sẽ
        pass


# ==============================================================================
# HÀM CHÍNH KHỞI CHẠY SERVER
# ==============================================================================
def run_app(port: int = 7860):
    handler = SpellCheckerRequestHandler
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}"
            print("=" * 65)
            print("   🚀 GIAO DIỆN WEB SỬA LỖI CHÍNH TẢ & SINH NHIỄU TIẾNG VIỆT")
            print("=" * 65)
            print(f"[*] Server đang chạy tại: {url}")
            print(f"[*] Mở trình duyệt và truy cập: {url}")
            print("[*] Nhấn Ctrl + C trong terminal này để dừng server.")
            print("=" * 65)

            # Tự động mở trình duyệt
            try:
                webbrowser.open(url)
            except Exception:
                pass

            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 98 or e.errno == 10048:
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
