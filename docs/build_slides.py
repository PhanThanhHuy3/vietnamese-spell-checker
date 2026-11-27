# -*- coding: utf-8 -*-
"""Sinh slide báo cáo DevOps cho dự án Sửa lỗi chính tả tiếng Việt (ViT5).

Bố cục/màu/font bám theo mẫu Slide_DevOps_FindWork.pptx:
  - 16:9 (13.33 x 7.5 in), nền body F8FAFC, cover 0F172A
  - pill nhãn EFF6FF/BFDBFE + chữ 18 đậm 1D4ED8
  - tiêu đề 32 Segoe UI 0F172A, phụ đề 13 Arial 64748B
  - card trắng viền E2E8F0 bo góc nhẹ, dải header F1F5F9
  - số trang "NN / 15" góc phải dưới
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(os.path.dirname(HERE), "Slide_DevOps_VietnameseSpellChecker.pptx")

NAVY = RGBColor(0x0F, 0x17, 0x2A)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
TEAL = RGBColor(0x0D, 0x94, 0x88)
PURPLE = RGBColor(0x7E, 0x22, 0xCE)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
AMBER = RGBColor(0xD9, 0x77, 0x06)
SLATE = RGBColor(0x33, 0x41, 0x55)
MUTED = RGBColor(0x64, 0x74, 0x8B)
BORDER = RGBColor(0xE2, 0xE8, 0xF0)
SOFT = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG = RGBColor(0xF8, 0xFA, 0xFC)
PILLBG = RGBColor(0xEF, 0xF6, 0xFF)
PILLLINE = RGBColor(0xBF, 0xDB, 0xFE)
CYAN = RGBColor(0x38, 0xBD, 0xF8)
SOFTBLUE = RGBColor(0x93, 0xC5, 0xFD)
TXT_LIGHT = RGBColor(0xE2, 0xE8, 0xF0)

TOTAL = 16
prs = Presentation()
prs.slide_width = Emu(12192000)
prs.slide_height = Emu(6858000)
BLANK = prs.slide_layouts[6]


# ------------------------------------------------------------------ helpers
def set_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def no_line(shape):
    shape.line.fill.background()


def dash(shape, color=BORDER, width=1.25):
    shape.line.color.rgb = color
    shape.line.width = Pt(width)
    ln = shape.line._get_or_add_ln()
    for old in ln.findall(qn("a:prstDash")):
        ln.remove(old)
    etree.SubElement(ln, qn("a:prstDash")).set("val", "dash")


def rrect(slide, x, y, w, h, fill=WHITE, line=BORDER, lw=1.0, radius=0.035):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
    s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        no_line(s)
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    s.text_frame.word_wrap = True
    return s


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf


def para(tf, first, runs, size=12.2, color=SLATE, font="Arial", bold=False,
         align=PP_ALIGN.LEFT, space_after=5, space_before=0, line=1.28):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    p.line_spacing = line
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, opt in runs:
        r = p.add_run()
        r.text = text
        r.font.size = Pt(opt.get("size", size))
        r.font.name = opt.get("font", font)
        r.font.bold = opt.get("bold", bold)
        r.font.color.rgb = opt.get("color", color)
    return p


def fill_text(tf, paras):
    for i, spec in enumerate(paras):
        runs = spec.get("runs")
        if runs is None:
            runs = spec["text"]
        para(tf, i == 0, runs, **{k: v for k, v in spec.items()
                                  if k in ("size", "color", "font", "bold", "align",
                                           "space_after", "space_before", "line")})


def pill(slide, text, w=None, x=0.80, y=0.42, h=0.32, color=BLUE):
    w = w or max(1.6, len(text) * 0.116 + 0.46)
    s = rrect(slide, x, y, w, h, fill=PILLBG, line=PILLLINE, lw=1.0, radius=0.10)
    tf = s.text_frame
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, True, text, size=11.5, color=color, bold=True, align=PP_ALIGN.CENTER,
         space_after=0, line=1.0)
    return s


def header(slide, kicker, title, subtitle, page, kicker_w=None):
    pill(slide, kicker, w=kicker_w)
    tf = textbox(slide, 0.80, 0.78, 11.70, 0.56)
    para(tf, True, title, size=27, color=NAVY, font="Segoe UI", bold=True,
         space_after=0, line=1.0)
    tf2 = textbox(slide, 0.80, 1.34, 11.70, 0.36)
    para(tf2, True, subtitle, size=12.5, color=MUTED, font="Arial",
         space_after=0, line=1.0)
    footer(slide, page)


def footer(slide, page):
    tf = textbox(slide, 11.45, 7.06, 1.35, 0.28)
    para(tf, True, f"{page:02d} / {TOTAL}", size=10, color=MUTED,
         align=PP_ALIGN.RIGHT, space_after=0, line=1.0)


def picture(slide, name, x, y, w, h=None):
    path = os.path.join(ASSETS, name)
    kw = {"width": Inches(w)}
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def body_card(slide, x, y, w, h, title, bullets, accent=BLUE, fs_title=15.5,
              fs_body=12.2, top_strip=True):
    rrect(slide, x, y, w, h, fill=WHITE, line=BORDER, lw=1.0)
    if top_strip:
        rrect(slide, x, y, w, 0.62, fill=SOFT, line=BORDER, lw=1.0)
    tf = textbox(slide, x + 0.15, y + 0.14, w - 0.30, 0.40)
    para(tf, True, title, size=fs_title, color=accent, bold=True, space_after=0, line=1.05)
    tb = textbox(slide, x + 0.15, y + 0.78, w - 0.32, h - 0.90)
    specs = []
    for b in bullets:
        if b.startswith("➡"):
            specs.append({"runs": [("➡ ", {"color": accent, "bold": True}),
                                   (b[2:], {"color": NAVY, "bold": True})],
                          "size": fs_body, "space_after": 5})
        else:
            specs.append({"text": ("• " + b) if not b.startswith("•") else b,
                          "size": fs_body, "color": SLATE, "space_after": 6})
    fill_text(tb, specs)


def three_cards(slide, items, y=2.00, h=4.85, fs_title=15.5, fs_body=12.2):
    for i, (title, bullets, accent) in enumerate(items):
        body_card(slide, 0.80 + i * 4.02, y, 3.78, h, title, bullets,
                  accent=accent, fs_title=fs_title, fs_body=fs_body)


def new_slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    set_bg(s, NAVY if dark else BG)
    return s


def dashes_to_placeholder(slide, x, y, w, h, label):
    s = rrect(slide, x, y, w, h, fill=WHITE, line=None, lw=1.0)
    dash(s, RGBColor(0x94, 0xA3, 0xB8), 1.5)
    tf = s.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    para(tf, True, label, size=10.5, color=MUTED, align=PP_ALIGN.CENTER,
         space_after=0, line=1.4)


def table(slide, x, y, w, rows, col_w, header_fill=BLUE):
    n_r, n_c = len(rows), len(rows[0])
    row_h = 0.33
    shape = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * n_r))
    tbl = shape.table
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = WHITE if ri % 2 else RGBColor(0xF8, 0xFA, 0xFC)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(9.8)
            r.font.name = "Arial"
            r.font.bold = (ri == 0)
            r.font.color.rgb = WHITE if ri == 0 else SLATE
    return tbl


# ================================================================== 1. cover
s = new_slide(dark=True)
picture(s, "cover.png", 0.17, 1.08, 6.49, 4.99)
tf = textbox(s, 6.85, 0.58, 6.20, 0.32)
para(tf, True, "TRƯỜNG ĐẠI HỌC ĐÔNG Á  •  KHOA CÔNG NGHỆ THÔNG TIN",
     size=10.5, color=SOFTBLUE, bold=True, space_after=0)
tf = textbox(s, 6.85, 0.96, 6.20, 0.32)
para(tf, True, "BÁO CÁO KẾT THÚC MÔN DEVOPS", size=12, color=AMBER,
     bold=True, space_after=0)
tf = textbox(s, 6.85, 1.34, 6.10, 2.90)
para(tf, True, "Triển khai Hệ thống", size=41, color=WHITE, font="Segoe UI",
     bold=True, space_after=0, line=1.06)
para(tf, False, "Sửa lỗi Chính tả", size=41, color=WHITE, font="Segoe UI",
     bold=True, space_after=0, line=1.06)
para(tf, False, "Tiếng Việt (ViT5)", size=41, color=WHITE, font="Segoe UI",
     bold=True, space_after=0, line=1.06)
tf = textbox(s, 6.85, 4.30, 6.20, 0.40)
para(tf, True, "Dự án: Transformer Seq2Seq – ViT5 trên VSEC Corpus, đóng gói Docker",
     size=13, color=CYAN, space_after=0, line=1.2)
tf = textbox(s, 6.85, 4.86, 6.15, 1.85)
para(tf, True, [("GVHD: ", {"color": SOFTBLUE, "bold": True}),
                ("ThS. Nguyễn Hoàng Anh Vũ", {"color": WHITE})],
     size=12, space_after=4)
para(tf, False, [("Họ và tên: ", {"color": SOFTBLUE, "bold": True}),
                 ("[Họ tên 1] – MSSV: [......]", {"color": WHITE})],
     size=12, space_after=2)
para(tf, False, "[Họ tên 2] – MSSV: [......]", size=12, color=WHITE, space_after=2)
para(tf, False, "[Họ tên 3] – MSSV: [......]", size=12, color=WHITE, space_after=6)
para(tf, False, [("Lớp: ", {"color": SOFTBLUE, "bold": True}),
                 ("ST23D  •  Đà Nẵng, tháng 11 năm 2026", {"color": WHITE})],
     size=12, space_after=0)

# ================================================================== 2. bối cảnh
s = new_slide()
header(s, "BỐI CẢNH & ĐỘNG LỰC",
       "Từ mô hình AI đến dịch vụ chạy thật",
       "Vì sao một mô hình Deep Learning 903,8 MB cần được đóng gói và vận hành theo chuẩn DevOps", 2)
three_cards(s, [
    ("1. Môi trường mong manh", [
        "PyTorch 2.x, Transformers 4.45 và SentencePiece rất nhạy phiên bản, lệch một bản là sai kết quả.",
        "Cùng mã nguồn nhưng chạy khác nhau giữa máy cá nhân, Google Colab và máy chủ.",
        "Không thể “pip install thủ công” cho từng lần triển khai.",
        "➡ Docker đóng băng môi trường, loại bỏ xung đột thư viện.",
    ], BLUE),
    ("2. Suy luận tốn tài nguyên", [
        "Checkpoint thực tế nặng 903,8 MB với ~250 triệu tham số.",
        "Suy luận CPU, beam search 4: trung bình ~2,3 giây/câu trên 935 câu test.",
        "Mỗi request chiếm RAM và CPU đáng kể, cần giới hạn tài nguyên.",
        "➡ Đặt API sau Nginx, thêm healthcheck và memory limit.",
    ], TEAL),
    ("3. Cập nhật mô hình liên tục", [
        "Mỗi lần finetune sinh ra một checkpoint mới cần thay bản cũ.",
        "Thay model thủ công dễ gây downtime và lẫn phiên bản.",
        "Cần khả năng quay lui khi bản mới suy giảm chất lượng.",
        "➡ CI/CD gắn nhãn commit SHA: đổi model chỉ bằng một lần push.",
    ], PURPLE),
], h=4.40)

# ================================================================== 3. lý thuyết
s = new_slide()
header(s, "CƠ SỞ LÝ THUYẾT", "DEVOPS là gì?",
       "Khái niệm cốt lõi, 4 chỉ số DORA và giá trị chuyển đổi thực tế cho dự án", 3)
cards = [
    ("Khái niệm DevOps", "devops_loop.png", 1.70, [
        "Kết hợp Development và Operations, xoá bỏ rào cản giữa hai nhóm.",
        "Vòng lặp 8 giai đoạn khép kín: Plan → Code → Build → Test → Release → Deploy → Operate → Monitor.",
        "Cả nhóm cùng chịu trách nhiệm về tốc độ và độ tin cậy.",
    ], BLUE),
    ("Mục tiêu đo lường DORA", "dora.png", 1.53, [
        "Lead Time: thời gian từ commit đến khi chạy thật.",
        "Deployment Frequency: tần suất phát hành bản mới.",
        "Change Failure Rate: tỷ lệ bản phát hành gây lỗi.",
        "MTTR: thời gian phục hồi dịch vụ sau sự cố.",
    ], TEAL),
    ("Trước vs. Sau DevOps", "before_after.png", 1.80, [
        "Cài thư viện thủ công → Docker image bất biến.",
        "Deploy copy file qua SSH → pipeline tự động.",
        "Model nằm rải rác → checkpoint được version hoá.",
        "Sửa lỗi bằng tay → rollback theo image SHA.",
    ], PURPLE),
]
for i, (title, img, ih, bullets, accent) in enumerate(cards):
    x = 0.80 + i * 4.02
    rrect(s, x, 1.90, 3.78, 4.95, fill=WHITE, line=BORDER)
    tf = textbox(s, x + 0.15, 2.02, 3.48, 0.38)
    para(tf, True, title, size=16, color=accent, bold=True, space_after=0, line=1.05)
    rrect(s, x + 0.20, 2.52, 3.38, 2.05, fill=WHITE, line=BORDER)
    picture(s, img, x + 0.25, 2.68, 3.28, ih)
    tb = textbox(s, x + 0.15, 4.68, 3.48, 2.05)
    fill_text(tb, [{"text": "• " + b, "size": 10.8, "color": SLATE, "space_after": 5}
                   for b in bullets])

# ================================================================== 4. linux
s = new_slide()
footer(s, 4)
rrect(s, 0.60, 0.50, 4.60, 6.50, fill=BLUE, line=None, radius=0.05)
tf = textbox(s, 0.90, 0.95, 4.05, 1.75)
para(tf, True, "Hệ thống Linux &", size=32, color=WHITE, font="Segoe UI", bold=True,
     space_after=0, line=1.1)
para(tf, False, "Ubuntu Server", size=32, color=WHITE, font="Segoe UI", bold=True,
     space_after=0, line=1.1)
picture(s, "terminal.png", 1.36, 2.78, 3.07, 3.20)
rrect(s, 5.40, 0.50, 7.33, 6.50, fill=WHITE, line=BORDER)
tf = textbox(s, 5.62, 0.80, 6.85, 0.70)
para(tf, True, "Vì sao chọn Ubuntu 24.04 LTS cho hạ tầng dự án?", size=18,
     color=NAVY, font="Segoe UI", bold=True, space_after=0, line=1.1)
tf = textbox(s, 5.70, 1.62, 6.75, 2.95)
fill_text(tf, [
    {"text": "• Ổn định 24/7: quản lý tiến trình và bộ nhớ tốt, phù hợp máy chủ suy luận chạy liên tục.",
     "color": SLATE, "size": 12.6, "space_after": 8},
    {"text": "• Nền tảng của container: Docker dựa trực tiếp vào cgroups (giới hạn CPU/RAM) và namespaces (cô lập tiến trình, mạng).",
     "color": SLATE, "size": 12.6, "space_after": 8},
    {"text": "• Bảo mật đa lớp: phân quyền chmod/chown, tường lửa UFW và đăng nhập bằng khoá SSH thay mật khẩu.",
     "color": SLATE, "size": 12.6, "space_after": 8},
    {"text": "• Miễn phí bản quyền, hệ sinh thái tài liệu phong phú, tương thích mọi VPS/EC2.",
     "color": SLATE, "size": 12.6, "space_after": 0},
])
rrect(s, 5.70, 4.78, 6.75, 1.95, fill=SOFT, line=BORDER)
tf = textbox(s, 5.90, 4.92, 6.35, 0.35)
para(tf, True, "Nhóm lệnh Linux dùng trong quá trình vận hành:", size=12.5,
     color=BLUE, bold=True, space_after=0)
tf = textbox(s, 5.90, 5.30, 6.40, 1.30)
fill_text(tf, [
    {"text": "• Tệp & phân quyền: ls -la, cp, mv, chmod 400, chown", "size": 11.2,
     "color": SLATE, "space_after": 4},
    {"text": "• Tiến trình & tài nguyên: ps aux, top, df -h, free -m", "size": 11.2,
     "color": SLATE, "space_after": 4},
    {"text": "• Mạng & dịch vụ: ss -tulpn, ufw status, curl /api/status, docker compose ps",
     "size": 11.2, "color": SLATE, "space_after": 0},
])

# ================================================================== 5. docker
s = new_slide()
header(s, "CÔNG NGHỆ CONTAINER", "Docker và Containerization",
       "Đóng gói ứng dụng suy luận ViT5 (Python 3.10 + PyTorch CPU) thành image bất biến", 5,
       kicker_w=3.55)
rrect(s, 0.80, 1.80, 11.70, 2.60, fill=WHITE, line=BORDER)
picture(s, "docker_stages.png", 3.45, 1.88, 6.40, 2.44)
body_card(s, 0.80, 4.60, 5.70, 2.35, "Kỹ thuật Multi-stage Build", [
    "Stage builder cài wheel CPU của PyTorch, Transformers, SentencePiece.",
    "Stage runtime chỉ copy site-packages + app.py + src/, chạy user non-root.",
    "Checkpoint 903,8 MB mount từ volume, không bake vào image.",
], accent=BLUE, fs_title=14.5, fs_body=11.4)
body_card(s, 6.80, 4.60, 5.70, 2.35, "Ưu điểm vượt trội so với Máy ảo (VM)", [
    "Dùng chung nhân Linux, không tốn hàng GB cho Guest OS.",
    "Khởi động vài giây thay vì vài phút như máy ảo.",
    "3 container (nginx, api, certbot) chạy gọn trên một VPS nhỏ.",
], accent=TEAL, fs_title=14.5, fs_body=11.4)

# ================================================================== 6. compose
s = new_slide()
header(s, "ĐIỀU PHỐI DỊCH VỤ", "Cấu hình Docker Compose",
       "Khai báo toàn bộ hệ thống trong một tệp YAML duy nhất và khởi chạy bằng một lệnh", 6,
       kicker_w=3.75)
rrect(s, 0.80, 1.75, 5.00, 5.20, fill=NAVY, line=None, radius=0.04)
rrect(s, 0.80, 1.75, 5.00, 0.45, fill=RGBColor(0x11, 0x1C, 0x31), line=None, radius=0.10)
tf = textbox(s, 1.00, 1.83, 4.60, 0.32)
para(tf, True, [("● ● ●  ", {"color": RGBColor(0xFB, 0xBF, 0x24)}),
                ("docker-compose.prod.yml (trích đoạn)", {"color": WHITE})],
     size=10.5, font="Consolas", space_after=0)
code = (
    "services:\n"
    "  api:\n"
    "    build: .\n"
    "    image: ${API_IMAGE}\n"
    "    container_name: spellcheck_api\n"
    "    restart: unless-stopped\n"
    "    env_file: [.env]\n"
    "    volumes:\n"
    "      - ./checkpoints:/app/checkpoints:ro\n"
    "    networks: [spellcheck-net]\n"
    "    mem_limit: 3g\n"
    "    healthcheck:\n"
    "      test: [\"CMD-SHELL\",\n"
    "        \"curl -fs localhost:8000/api/status\"]\n"
    "      interval: 20s\n"
    "      start_period: 90s\n"
    "\n"
    "  nginx:\n"
    "    image: nginx:1.27-alpine\n"
    "    ports: [\"80:80\", \"443:443\"]\n"
    "    depends_on:\n"
    "      api: {condition: service_healthy}\n"
)
tf = textbox(s, 0.98, 2.35, 4.65, 4.45)
tf.paragraphs[0].line_spacing = 1.0
r = tf.paragraphs[0].add_run()
r.text = code
r.font.size = Pt(9.2)
r.font.name = "Consolas"
r.font.color.rgb = RGBColor(0x38, 0xBD, 0xF8)
rrect(s, 6.05, 1.75, 6.50, 5.20, fill=WHITE, line=BORDER)
tf = textbox(s, 6.28, 1.92, 6.05, 0.45)
para(tf, True, "Các thành phần chính của Docker", size=20, color=NAVY,
     font="Segoe UI", bold=True, space_after=0)
tf = textbox(s, 6.28, 2.45, 6.05, 2.10)
fill_text(tf, [
    {"text": "• Docker Image: bản mẫu chỉ đọc, build một lần và tái sử dụng ở mọi môi trường.",
     "size": 12.2, "color": SLATE, "space_after": 7},
    {"text": "• Docker Container: thực thể đang chạy của image (api, nginx, certbot).",
     "size": 12.2, "color": SLATE, "space_after": 7},
    {"text": "• Docker Volume: vùng nhớ bền vững, giữ checkpoint và log khi container bị tạo lại.",
     "size": 12.2, "color": SLATE, "space_after": 7},
    {"text": "• Docker Network: mạng bridge riêng, API không mở cổng ra Internet.",
     "size": 12.2, "color": SLATE, "space_after": 0},
])
tf = textbox(s, 6.28, 4.62, 6.05, 0.45)
para(tf, True, "Điều phối với Docker Compose", size=20, color=BLUE,
     font="Segoe UI", bold=True, space_after=0)
tf = textbox(s, 6.28, 5.18, 6.05, 1.62)
fill_text(tf, [
    {"text": "• Định nghĩa đa dịch vụ trong một tệp YAML thay vì nhiều lệnh docker run.",
     "size": 12.2, "color": SLATE, "space_after": 7},
    {"text": "• Khởi chạy toàn hệ thống: docker compose -f docker-compose.prod.yml up -d",
     "size": 11.4, "color": SLATE, "font": "Consolas", "space_after": 7},
    {"text": "• depends_on + healthcheck: Nginx chỉ nhận traffic khi API đã nạp xong model.",
     "size": 12.2, "color": SLATE, "space_after": 0},
])

# ================================================================== 7. nginx
s = new_slide()
header(s, "HẠ TẦNG WEB & ĐỊNH TUYẾN", "Nginx và Reverse Proxy",
       "Cổng tiếp nhận duy nhất, tối ưu hiệu năng và bảo mật SSL/TLS cho API suy luận", 7,
       kicker_w=4.35)
rrect(s, 0.80, 1.80, 11.70, 0.95, fill=WHITE, line=BORDER)
tf = textbox(s, 1.00, 1.92, 11.30, 0.70)
para(tf, True, "Nginx là điểm vào duy nhất của hệ thống: phục vụ giao diện web, chuyển tiếp "
               "/api/* tới tiến trình suy luận ở cổng nội bộ 8000, xử lý chứng chỉ SSL và "
               "giới hạn tần suất request để bảo vệ CPU.", size=12.6, color=SLATE,
     space_after=0, line=1.35)
three_cards(s, [
    ("1. Phục vụ giao diện & nén", [
        "Trả trực tiếp HTML/CSS/JS của app.py cho trình duyệt.",
        "Bật Gzip giảm dung lượng truyền tải trang.",
        "Cache tài nguyên tĩnh, giảm tải cho tiến trình Python.",
    ], BLUE),
    ("2. Reverse Proxy & Routing", [
        "Request /api/* được chuyển tới http://api:8000/api/.",
        "Gắn header X-Real-IP, X-Forwarded-For, X-Forwarded-Proto.",
        "Frontend và API cùng một domain nên triệt tiêu lỗi CORS.",
    ], TEAL),
    ("3. SSL & Rate Limiting", [
        "Chứng chỉ Let's Encrypt tự động cấp và gia hạn qua certbot.",
        "Chuyển hướng toàn bộ HTTP (80) sang HTTPS (443).",
        "Giới hạn tần suất: mỗi request tốn ~2,3 s CPU nên rất cần chống spam.",
    ], PURPLE),
], y=2.95, h=3.45, fs_title=15, fs_body=11.6)
rrect(s, 0.80, 6.52, 11.70, 0.42, fill=WHITE, line=BORDER)
tf = textbox(s, 1.00, 6.59, 11.30, 0.30)
para(tf, True, "location /api/  {  proxy_pass http://api:8000/api/;  proxy_read_timeout 120s;  }   # đoạn cấu hình then chốt",
     size=9.8, color=BLUE, font="Consolas", space_after=0)

# ================================================================== 8. cloud
s = new_slide()
header(s, "HẠ TẦNG ĐIỆN TOÁN ĐÁM MÂY", "Máy chủ Cloud VPS / AWS EC2",
       "Máy chủ linh hoạt, bảo mật đa tầng và tối ưu chi phí cho một mô hình 250M tham số", 8,
       kicker_w=4.60)
rrect(s, 7.80, 0.55, 4.70, 2.75, fill=WHITE, line=BORDER)
picture(s, "topology.png", 7.90, 0.65, 4.50, 2.55)
three_cards(s, [
    ("Virtual Server (Ubuntu 24.04)", [
        "Cloud VPS 2 vCPU / 4 GB RAM / 40 GB SSD, tương đương EC2 t3.medium.",
        "Toàn quyền kernel, Docker và ổ đĩa; IP công khai [điền IP máy chủ].",
        "Giới hạn RAM container: API 3 GB, Nginx 128 MB.",
    ], BLUE),
    ("Security Group & Firewall", [
        "Chỉ mở 22 (SSH), 80 (HTTP), 443 (HTTPS) từ Internet.",
        "Cổng 8000 của API chỉ tồn tại trong mạng nội bộ Docker.",
        "UFW đồng bộ với Security Group để tránh cấu hình lệch.",
    ], TEAL),
    ("Key Pair (xác thực RSA)", [
        "Bỏ đăng nhập bằng mật khẩu, dùng cặp khoá RSA bất đối xứng.",
        "Phân quyền tệp khoá: chmod 400 hoặc icacls trên Windows.",
        "Khoá lưu trong GitHub Secrets phục vụ deploy tự động.",
    ], AMBER),
], y=3.50, h=3.15, fs_title=13.8, fs_body=11.2)

# ================================================================== 9. cicd
s = new_slide()
header(s, "TỰ ĐỘNG HÓA TRIỂN KHAI", "CI/CD với GitHub Actions",
       "Tự động kiểm thử, build image, đẩy lên registry và triển khai qua SSH", 9,
       kicker_w=3.95)
rrect(s, 0.80, 1.85, 11.70, 2.55, fill=WHITE, line=BORDER)
picture(s, "pipeline.png", 4.14, 1.95, 5.03, 2.35)
three_cards(s, [
    ("1. Tích hợp liên tục (ci.yml)", [
        "Kích hoạt khi push hoặc mở Pull Request.",
        "Kiểm tra import, nạp tokenizer và chạy bộ câu hồi quy.",
        "Chặn merge nếu chất lượng giảm dưới ngưỡng EM.",
    ], BLUE),
    ("2. Đóng gói & Registry", [
        "Build image từ Dockerfile multi-stage, cache layer.",
        "Gắn nhãn commit SHA để truy vết đúng phiên bản.",
        "Đẩy image lên GitHub Container Registry (GHCR).",
    ], TEAL),
    ("3. Deploy SSH & Rollback", [
        "SSH vào VPS bằng action appleboy/ssh-action.",
        "Kéo image mới, chạy docker compose up -d.",
        "Gọi /api/status; thất bại thì rollback image SHA cũ.",
    ], PURPLE),
], y=4.60, h=2.55, fs_title=13.8, fs_body=11.0)

# ================================================================== 10. hiện thực hóa
s = new_slide()
header(s, "HIỆN THỰC HÓA HỆ THỐNG", "Ba thách thức kỹ thuật đặc thù",
       "Những vấn đề chỉ xuất hiện khi đưa mô hình Deep Learning lên môi trường thật", 10,
       kicker_w=4.55)
rrect(s, 0.80, 1.78, 11.70, 2.10, fill=WHITE, line=BORDER)
picture(s, "inference.png", 3.20, 1.92, 6.90, 1.95)
three_cards(s, [
    ("Checkpoint lớn & cold start", [
        "903,8 MB trọng số khiến việc nạp model mất hàng chục giây.",
        "Giải pháp: mount read-only, warm-up 1 câu khi khởi động.",
        "healthcheck start_period 90s để không restart oan.",
    ], BLUE),
    ("Tương thích Tokenizer T5", [
        "Transformers mới đổi cấu trúc vocab Unigram gây lỗi khi nạp.",
        "Đã vá T5Tokenizer.__init__ trong src/models/spell_corrector.py.",
        "Pin transformers==4.45.2 trong image để tái lập lỗi đã sửa.",
    ], TEAL),
    ("Hợp đồng API & fallback", [
        "3 endpoint: correct_spelling, compare_engines, generate_noise.",
        "Model lỗi vẫn còn rule engine N-gram làm phương án dự phòng.",
        "Giới hạn max_length và timeout để tránh treo worker.",
    ], PURPLE),
], y=4.08, h=2.82, fs_title=13.8, fs_body=10.8)

# ================================================================== 11. kiến trúc
s = new_slide()
header(s, "THIẾT KẾ KIẾN TRÚC", "Kiến trúc ba tầng của hệ thống",
       "Mô hình 3 tầng chuẩn mực, container hoá toàn bộ và không phụ thuộc API bên ngoài", 11,
       kicker_w=3.55)
rrect(s, 0.80, 1.85, 11.70, 2.45, fill=WHITE, line=BORDER)
picture(s, "architecture.png", 3.90, 1.88, 5.50, 2.40)
three_cards(s, [
    ("1. Tầng Trình bày", [
        "Giao diện HTML/CSS/JS nhúng trong app.py (http.server, cổng 7860).",
        "Nginx đứng trước, phục vụ TLS và nén nội dung.",
        "Hiển thị diff: từ sai màu đỏ, từ đã sửa màu xanh.",
    ], BLUE),
    ("2. Tầng Suy luận", [
        "VietnameseSpellCorrector: ViT5-base, beam 4, prefix “spelling: ”.",
        "SmartVietnameseCorrector: từ điển âm tiết, bigram, Levenshtein.",
        "SequenceMatcher tạo danh sách lỗi và đề xuất sửa.",
    ], TEAL),
    ("3. Tầng Dữ liệu", [
        "checkpoints/best_model: model.safetensors 903,8 MB (read-only).",
        "data/processed: train/val/test.jsonl từ VSEC Corpus.",
        "logs/: kết quả đánh giá, log vận hành ghi ra volume.",
    ], PURPLE),
], y=4.50, h=2.50, fs_title=13.4, fs_body=10.8)

# ================================================================== 12. quy trình
s = new_slide()
header(s, "QUY TRÌNH VẬN HÀNH", "Deployment Steps",
       "Quy trình triển khai chuẩn hoá trên máy chủ đám mây, lặp lại được nhiều lần", 12,
       kicker_w=3.65)
rrect(s, 7.80, 0.55, 4.70, 2.75, fill=WHITE, line=BORDER)
picture(s, "terminal.png", 8.31, 0.65, 3.68, 2.55)
three_cards(s, [
    ("1. Cấu hình VPS & tên miền", [
        "Tạo máy chủ Ubuntu 24.04, mở cổng 22/80/443.",
        "Phân quyền khoá SSH và kiểm tra kết nối.",
        "Trỏ bản ghi DNS về IP máy chủ.",
    ], BLUE),
    ("2. Cài Docker & môi trường", [
        "Cài Docker Engine và Compose Plugin từ kho chính thức.",
        "Clone mã nguồn, tạo .env và mount thư mục checkpoints.",
        "Tạo network và volume trước khi khởi chạy.",
    ], TEAL),
    ("3. Khởi chạy & kích hoạt CD", [
        "Chạy docker compose -f docker-compose.prod.yml up -d.",
        "Kiểm tra /api/status trả model_loaded: true.",
        "Khai báo GitHub Secrets để pipeline tự deploy.",
    ], GREEN),
], y=3.50, h=3.40, fs_title=13.8, fs_body=11.4)

# ================================================================== 13. kết quả
s = new_slide()
header(s, "KẾT QUẢ THỰC NGHIỆM", "Chất lượng mô hình trên tập test VSEC",
       "935 câu test độc lập, suy luận CPU với beam search 4 — đánh giá bằng 8 chỉ số định lượng",
       13, kicker_w=3.95)
rows = [
    ["Chỉ số", "ViT5 finetuned", "Baseline rule"],
    ["Exact Match (EM)", "73,48 %", "6,84 %"],
    ["Token Accuracy", "95,59 %", "93,57 %"],
    ["BLEU-4", "97,21", "0,85"],
    ["CER (↓)", "1,25 %", "2,04 %"],
    ["WER (↓)", "5,63 %", "6,52 %"],
    ["Detection Precision", "62,20 %", "27,18 %"],
    ["Detection Recall", "89,26 %", "30,49 %"],
    ["Detection F1", "73,31 %", "28,74 %"],
]
table(s, 0.80, 2.00, 6.60, rows, [2.90, 1.95, 1.75])
tf = textbox(s, 0.80, 5.15, 6.60, 0.60)
para(tf, True, "Nguồn: _tmp_eval_full.log — 935 câu, max_length 96, beam 4, thời gian 2.140 giây (CPU).",
     size=9.5, color=MUTED, space_after=0, line=1.25)
rrect(s, 7.75, 2.00, 4.75, 4.85, fill=WHITE, line=BORDER)
tf = textbox(s, 7.95, 2.12, 4.40, 0.35)
para(tf, True, "Đọc kết quả", size=15, color=BLUE, bold=True, space_after=0)
tf = textbox(s, 7.95, 2.60, 4.40, 4.10)
fill_text(tf, [
    {"text": "• EM 73,48 %: gần 3/4 câu được sửa đúng hoàn toàn, gấp hơn 10 lần baseline.",
     "size": 11.2, "color": SLATE, "space_after": 7},
    {"text": "• BLEU-4 97,21 và CER 1,25 %: câu sửa rất sát câu chuẩn.",
     "size": 11.2, "color": SLATE, "space_after": 7},
    {"text": "• Detection Recall 89,26 %: bắt được phần lớn lỗi thật.",
     "size": 11.2, "color": SLATE, "space_after": 7},
    {"text": "• Detection Precision 62,20 %: còn 930 vị trí bị sửa dù không sai (over-correction).",
     "size": 11.2, "color": SLATE, "space_after": 7},
    {"text": "• WER 5,63 %: khoảng 1 từ sai trên 18 từ; Token Accuracy 95,59 %.",
     "size": 11.2, "color": SLATE, "space_after": 7},
    {"text": "• Toàn bộ 935 câu chạy CPU hết 2.140 giây (~2,3 giây/câu).",
     "size": 11.2, "color": SLATE, "space_after": 7},
    {"runs": [("➡ ", {"color": BLUE, "bold": True}),
              ("Vượt baseline ở cả 8 chỉ số; hạn chế còn lại là độ chính xác phát hiện.",
               {"color": NAVY, "bold": True})], "size": 11.0, "space_after": 0},
])

# ================================================================== 14. giao diện
s = new_slide()
header(s, "KẾT QUẢ THỰC NGHIỆM", "Giao diện và API của hệ thống đang chạy",
       "Ảnh chụp trực tiếp từ dịch vụ tại localhost:7860 sau khi nạp checkpoint ViT5", 14,
       kicker_w=3.95)
rrect(s, 0.80, 1.72, 6.55, 4.92, fill=WHITE, line=BORDER)
picture(s, "ui_result_ai.png", 0.95, 1.86, 6.25, 4.79)
tf = textbox(s, 0.95, 6.70, 6.25, 0.30)
para(tf, True, "Tab sửa lỗi: phát hiện 5 lỗi, sửa 5 vị trí, phản hồi 5.258 ms trên CPU",
     size=9.5, color=MUTED, space_after=0)
rrect(s, 7.55, 1.72, 4.95, 4.92, fill=WHITE, line=BORDER)
tf = textbox(s, 7.72, 1.84, 4.60, 0.32)
para(tf, True, "Đối sánh hai động cơ", size=14, color=BLUE, bold=True, space_after=0)
picture(s, "ui_compare.png", 7.75, 2.24, 4.55, 3.49)
tf = textbox(s, 7.72, 5.82, 4.62, 0.72)
fill_text(tf, [
    {"text": "• ViT5: 1.213 ms — Rule Engine: 0 ms, cùng trả về câu đúng.",
     "size": 9.6, "color": SLATE, "space_after": 3},
    {"text": '• GET /api/status → {"status": "ready", "ai_loaded": true}',
     "size": 8.8, "color": SLATE, "font": "Consolas", "space_after": 0},
])

# ================================================================== 15. tổng kết
s = new_slide()
header(s, "ĐÁNH GIÁ & TỔNG KẾT", "Kết quả đạt được và điểm mạnh",
       "Tổng kết giá trị kỹ thuật của việc đóng gói – tự động hoá – vận hành hệ thống", 15,
       kicker_w=3.75)
rrect(s, 0.80, 1.85, 6.10, 2.60, fill=WHITE, line=BORDER)
picture(s, "ui_home_top.png", 1.19, 1.94, 5.32, 2.42)
rrect(s, 7.10, 1.85, 5.40, 2.60, fill=WHITE, line=BORDER)
tf = textbox(s, 7.30, 1.98, 5.00, 0.35)
para(tf, True, "Kết quả nổi bật", size=15, color=GREEN, bold=True, space_after=0)
tf = textbox(s, 7.30, 2.42, 5.00, 1.95)
fill_text(tf, [
    {"text": "• EM 73,48 % so với 6,84 % của rule engine.", "size": 11.2,
     "color": SLATE, "space_after": 5},
    {"text": "• BLEU-4 97,21 so với 0,85; CER 1,25 % so với 2,04 %.", "size": 11.2,
     "color": SLATE, "space_after": 5},
    {"text": "• Detection F1 73,31 % so với 28,74 %.", "size": 11.2,
     "color": SLATE, "space_after": 5},
    {"runs": [("➡ ", {"color": GREEN, "bold": True}),
              ("Chất lượng mô hình và tính ổn định vận hành cùng được bảo đảm.",
               {"color": NAVY, "bold": True})], "size": 10.8, "space_after": 0},
])
three_cards(s, [
    ("Tự động hoá", [
        "Một lần push: test, build image, deploy, healthcheck.",
        "Rollback về image SHA cũ khi bản mới không đạt.",
    ], BLUE),
    ("Môi trường đồng nhất", [
        "Docker khoá phiên bản Python, PyTorch, Transformers.",
        "Loại bỏ lỗi “chạy được trên máy tôi”.",
    ], TEAL),
    ("Dễ bảo trì & mở rộng", [
        "Cấu hình dưới dạng mã, tái lập trên máy chủ mới.",
        "Hướng mở rộng: GPU, queue, monitoring, logging tập trung.",
    ], PURPLE),
], y=4.60, h=2.35, fs_title=15.5, fs_body=11.0)

# ================================================================== 16. cảm ơn
s = new_slide(dark=True)
picture(s, "cover.png", 0.80, 0.50, 4.21, 6.50)
tf = textbox(s, 6.24, 0.90, 6.30, 1.30)
para(tf, True, "Cảm ơn", size=64, color=WHITE, font="Segoe UI", bold=True,
     space_after=0, line=1.0)
tf = textbox(s, 6.05, 2.40, 6.20, 0.75)
para(tf, True, "Cảm ơn Thầy và các bạn đã chú ý lắng nghe!\nNhóm rất mong nhận được ý kiến đóng góp.",
     size=15, color=CYAN, space_after=0, line=1.35)
tf = textbox(s, 6.05, 3.35, 6.10, 2.40)
fill_text(tf, [
    {"text": "• Triển khai hệ thống sửa lỗi chính tả tiếng Việt (ViT5) bằng Docker và CI/CD",
     "size": 12.5, "color": TXT_LIGHT, "space_after": 6},
    {"text": "• GVHD: ThS. Nguyễn Hoàng Anh Vũ", "size": 12.5, "color": TXT_LIGHT,
     "space_after": 6},
    {"text": "• Nhóm: [Họ tên 1] – [Họ tên 2] – [Họ tên 3]", "size": 12.5,
     "color": TXT_LIGHT, "space_after": 6},
    {"text": "• ST23D – Khoa Công nghệ Thông tin, Trường Đại học Đông Á", "size": 12.5,
     "color": TXT_LIGHT, "space_after": 6},
    {"text": "• Nguồn dữ liệu: VSEC Corpus 9.341 câu – 11.202 lỗi thực tế", "size": 12.5,
     "color": SOFTBLUE, "space_after": 0},
])

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
