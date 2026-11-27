# -*- coding: utf-8 -*-
"""Sinh báo cáo Word (định dạng theo mẫu BaoCao_DevOps_FindWork1.docx)."""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(os.path.dirname(HERE), "BaoCao_DevOps_VietnameseSpellChecker_final.docx")

BLUE = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)

doc = Document()

# ------------------------------------------------------------------ page setup
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.left_margin = Cm(3.0)
sec.right_margin = Cm(2.0)
sec.top_margin = Cm(2.5)
sec.bottom_margin = Cm(2.2)


def set_font(run, name="Times New Roman", size=13, bold=None, italic=None,
             color=DARK):
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), name)


# ------------------------------------------------------------------ styles
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(13)
normal.font.color.rgb = DARK
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.first_line_indent = Cm(1.0)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

specs = {"Heading 1": (16, WD_ALIGN_PARAGRAPH.CENTER, 18, 12),
         "Heading 2": (14, WD_ALIGN_PARAGRAPH.LEFT, 14, 8),
         "Heading 3": (13, WD_ALIGN_PARAGRAPH.LEFT, 10, 6)}
for name, (size, align, before, after) in specs.items():
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = BLUE
    st.paragraph_format.alignment = align
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.line_spacing = 1.3
    st.paragraph_format.first_line_indent = Cm(0)
    st.paragraph_format.keep_with_next = True

for name in ("List Bullet", "List Number"):
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.font.size = Pt(13)
    st.paragraph_format.line_spacing = 1.4
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.first_line_indent = Cm(0)

cap = doc.styles["Caption"]
cap.font.name = "Times New Roman"
cap.font.size = Pt(12)
cap.font.bold = True
cap.font.color.rgb = GRAY
cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.paragraph_format.space_before = Pt(4)
cap.paragraph_format.space_after = Pt(10)
cap.paragraph_format.first_line_indent = Cm(0)


# ------------------------------------------------------------------ helpers
def p(text="", size=13, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
      indent=1.0, space_after=6, line=1.5, color=DARK, font="Times New Roman"):
    par = doc.add_paragraph()
    par.alignment = align
    par.paragraph_format.first_line_indent = Cm(indent)
    par.paragraph_format.space_after = Pt(space_after)
    par.paragraph_format.line_spacing = line
    if text:
        r = par.add_run(text)
        set_font(r, font, size, bold, italic, color)
    return par


def lead(bold_text, rest=""):
    """Đoạn văn có cụm mở đầu in đậm."""
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.first_line_indent = Cm(1.0)
    par.paragraph_format.line_spacing = 1.5
    par.paragraph_format.space_after = Pt(6)
    r1 = par.add_run(bold_text)
    set_font(r1, size=13, bold=True)
    if rest:
        r2 = par.add_run(rest)
        set_font(r2, size=13, color=DARK)
    return par


def bullet(text=None, bold_lead=None):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.left_indent = Cm(1.0)
    par.paragraph_format.first_line_indent = Cm(-0.5)
    if bold_lead:
        r = par.add_run(bold_lead)
        set_font(r, size=13, bold=True)
    if text:
        r = par.add_run(text)
        set_font(r, size=13)
    return par


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def h3(text):
    doc.add_heading(text, level=3)


def shade(par, hex_color="EEF2F7"):
    ppr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    ppr.append(shd)


def bordered(par):
    ppr = par._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "6")
        el.set(qn("w:color"), "CBD5E1")
        pbdr.append(el)
    ppr.append(pbdr)


def code(text, caption=None):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.left_indent = Cm(0.6)
    par.paragraph_format.right_indent = Cm(0.2)
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(8)
    par.paragraph_format.line_spacing = 1.0
    par.paragraph_format.first_line_indent = Cm(0)
    lines = text.strip("\n").split("\n")
    for i, line in enumerate(lines):
        r = par.add_run(line)
        set_font(r, "Consolas", 9.5, color=RGBColor(0x1E, 0x29, 0x3B))
        if i < len(lines) - 1:
            r.add_break()
    shade(par)
    bordered(par)
    if caption:
        cp(caption)
    return par


def cp(text):
    par = doc.add_paragraph(text, style="Caption")
    for r in par.runs:
        set_font(r, size=12, bold=True, color=GRAY)
    return par


def figure(name, caption, width=15.5):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.first_line_indent = Cm(0)
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(2)
    par.add_run().add_picture(os.path.join(ASSETS, name), width=Cm(width))
    cp(caption)


def table(headers, rows, caption, widths=None, font_size=12):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        par = hdr[i].paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.first_line_indent = Cm(0)
        par.paragraph_format.space_after = Pt(2)
        r = par.add_run(h)
        set_font(r, size=font_size, bold=True)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "DCE6F1")
        hdr[i]._tc.get_or_add_tcPr().append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            par = cells[i].paragraphs[0]
            par.alignment = (WD_ALIGN_PARAGRAPH.LEFT if i == 0
                             else WD_ALIGN_PARAGRAPH.CENTER)
            par.paragraph_format.first_line_indent = Cm(0)
            par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.line_spacing = 1.15
            r = par.add_run(str(val))
            set_font(r, size=font_size)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    cp(caption)
    return t


def toc_field():
    par = doc.add_paragraph()
    par.paragraph_format.first_line_indent = Cm(0)
    run = par.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r'TOC \o "1-3" \h \z \u'
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = "Nhấn chuột phải vào đây → Update Field để cập nhật mục lục."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(fld)
    run._r.append(instr)
    run._r.append(sep)
    run._r.append(txt)
    run._r.append(end)
    set_font(run, size=12, italic=True, color=GRAY)


def footer_page_numbers():
    footer = doc.sections[0].footer
    par = footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.first_line_indent = Cm(0)
    r = par.add_run("Trang ")
    set_font(r, size=11, color=GRAY)
    run = par.add_run()
    set_font(run, size=11, color=GRAY)
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(fld)
    run._r.append(instr)
    run._r.append(end)


# ================================================================== TRANG BÌA
p("TRƯỜNG ĐẠI HỌC ĐÔNG Á", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
  indent=0, space_after=2)
p("KHOA CÔNG NGHỆ THÔNG TIN", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
  indent=0, space_after=24)
p("BÁO CÁO KẾT THÚC MÔN", size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
  indent=0, space_after=0)
p("DEVOPS", size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0,
  space_after=18)
p("TÊN ĐỀ TÀI:", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0,
  space_after=6)
p("Triển khai Hệ thống Sửa lỗi Chính tả Tiếng Việt", size=17, bold=True,
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, space_after=0)
p("(Mô hình ViT5) bằng Docker và CI/CD", size=17, bold=True,
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, space_after=30)

p("Giảng viên hướng dẫn: ThS. Nguyễn Hoàng Anh Vũ", size=13,
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, space_after=6)
p("Sinh viên thực hiện:", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
  indent=0, space_after=6)
for line in ("[Họ và tên 1] – MSSV: [................]",
             "[Họ và tên 2] – MSSV: [................]",
             "[Họ và tên 3] – MSSV: [................]"):
    p(line, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, space_after=4)
p("Lớp: ST23D", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0,
  space_after=40)
p("Đà Nẵng, tháng 11 năm 2026", size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
  indent=0)

doc.add_page_break()

# ================================================================== MỤC LỤC
p("MỤC LỤC", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0,
  space_after=12)
toc_field()
doc.add_page_break()

# ================================================================== MỞ ĐẦU
h1("LỜI MỞ ĐẦU")
p("Trong những năm gần đây, các mô hình học sâu đã trở thành lõi của nhiều sản phẩm "
  "phần mềm. Tuy nhiên, một mô hình chỉ có giá trị khi nó được đóng gói, kiểm thử, "
  "triển khai và vận hành ổn định trong môi trường thật. Khoảng cách giữa môi trường "
  "thí nghiệm (notebook, máy cá nhân) và môi trường máy chủ thường tạo ra những lỗi "
  "rất khó tái lập: lệch phiên bản thư viện, thiếu tài nguyên, checkpoint không tìm "
  "thấy hoặc tiến trình bị dừng mà không có cơ chế tự phục hồi.")
p("Đồ án môn DevOps được thực hiện trên hệ thống Phát hiện và Tự động Sửa lỗi Chính tả "
  "Tiếng Việt, sử dụng kiến trúc Transformer Seq2Seq (ViT5) huấn luyện trên bộ dữ liệu "
  "VSEC Corpus. Hệ thống gồm một mô hình khoảng 250 triệu tham số với tệp trọng số "
  "903,8 MB, một giao diện web và một động cơ luật N-gram đóng vai trò đối chứng. "
  "Đặc thù của một hệ thống như vậy là phụ thuộc nặng vào môi trường thực thi và cần "
  "được cập nhật mỗi khi mô hình được finetune lại.")
p("Mục tiêu của báo cáo là vận dụng các kiến thức DevOps vào chính hệ thống đó: "
  "container hoá toàn bộ ứng dụng bằng Docker, điều phối bằng Docker Compose, đặt sau "
  "Nginx làm reverse proxy, triển khai trên máy chủ Ubuntu 24.04 LTS và tự động hoá "
  "quy trình bằng GitHub Actions. Báo cáo cũng trình bày kết quả thực nghiệm định lượng "
  "của mô hình để chứng minh hệ thống hoạt động đúng sau khi đóng gói.")
p("Báo cáo được bố cục thành 5 chương chính:", indent=1.0)
for b in ("Chương 1: Cơ sở lý thuyết – Tổng quan về DevOps, Linux, Docker, Nginx, "
          "hạ tầng đám mây và CI/CD.",
          "Chương 2: Phân tích và thiết kế hệ thống – Mục tiêu, bài toán, yêu cầu, "
          "kiến trúc ba tầng và sơ đồ triển khai.",
          "Chương 3: Triển khai hệ thống với Docker và DevOps – Cấu hình hạ tầng, "
          "Dockerfile, Compose, Nginx, biến môi trường và pipeline.",
          "Chương 4: Kết quả thực nghiệm và đánh giá – Chất lượng mô hình, kịch bản "
          "kiểm thử API và đánh giá ưu, nhược điểm.",
          "Chương 5: Kết luận và hướng phát triển – Tổng kết kết quả và định hướng "
          "mở rộng."):
    bullet(b)
doc.add_page_break()

# ================================================================== CHƯƠNG 1
h1("CHƯƠNG 1. CƠ SỞ LÝ THUYẾT")

h2("1.1. Tổng quan về DevOps")
h3("1.1.1. Khái niệm DevOps")
p("DevOps là sự kết hợp giữa Development (Phát triển) và Operations (Vận hành). Đây "
  "không chỉ là một bộ công cụ mà là một cách làm việc: đội phát triển và đội vận hành "
  "cùng tham gia vào một quy trình liên tục, chia sẻ trách nhiệm về chất lượng và độ "
  "tin cậy của phần mềm. Vòng đời DevOps gồm tám giai đoạn khép kín: Plan – Code – "
  "Build – Test – Release – Deploy – Operate – Monitor.")
p("Điểm khác biệt cốt lõi so với cách làm truyền thống là tính tự động hoá và khả năng "
  "tái lập. Mọi cấu hình hạ tầng, kịch bản kiểm thử và quy trình triển khai đều được "
  "mô tả bằng mã (Infrastructure as Code), lưu trong hệ thống quản lý phiên bản và có "
  "thể chạy lại nhiều lần với cùng kết quả.")
figure("devops_loop.png", "Hình 1.1. Tám giai đoạn của vòng đời DevOps", width=11.0)

h3("1.1.2. Mục tiêu của DevOps")
p("Bốn chỉ số DORA thường được dùng để đo mức độ trưởng thành của một quy trình DevOps:")
bullet("thời gian từ lúc commit mã nguồn đến khi thay đổi chạy trên môi trường thật.",
       bold_lead="Lead Time: ")
bullet("số lần phát hành trong một khoảng thời gian; phát hành nhỏ và thường xuyên "
       "giảm rủi ro so với các đợt phát hành lớn.", bold_lead="Deployment Frequency: ")
bullet("tỷ lệ bản phát hành gây sự cố; giảm nhờ kiểm thử tự động và môi trường đồng nhất.",
       bold_lead="Change Failure Rate: ")
bullet("thời gian phục hồi dịch vụ sau sự cố; giảm nhờ giám sát và rollback tự động.",
       bold_lead="MTTR (Mean Time To Recovery): ")
p("Với một hệ thống học sâu, các chỉ số này còn có ý nghĩa bổ sung: mỗi lần finetune "
  "sinh ra một checkpoint mới, nên tần suất cập nhật mô hình và khả năng quay lui về "
  "phiên bản cũ trở thành yêu cầu bắt buộc.")

h3("1.1.3. Dev và Ops: vấn đề thường gặp và cách DevOps giải quyết")
p("Trong mô hình truyền thống, đội phát triển muốn thay đổi nhanh còn đội vận hành "
  "muốn hệ thống ổn định, ít thay đổi. Mâu thuẫn này sinh ra những vấn đề quen thuộc:")
bullet("mã nguồn chạy đúng trên máy lập trình nhưng lỗi trên máy chủ do khác phiên bản "
       "thư viện. DevOps giải quyết bằng container, đóng gói cả môi trường thực thi.",
       bold_lead="Môi trường không đồng nhất: ")
bullet("thao tác copy tệp và cấu hình qua SSH dễ sai và không ghi lại được. DevOps "
       "thay bằng kịch bản tự động và pipeline có phiên bản.",
       bold_lead="Triển khai thủ công: ")
bullet("không có công cụ đo lường nên phát hiện sự cố chậm. DevOps bổ sung healthcheck, "
       "log tập trung và cảnh báo.", bold_lead="Thiếu khả năng quan sát: ")
bullet("khôi phục bằng tay mất nhiều thời gian. DevOps dùng image bất biến để quay lui "
       "về đúng phiên bản trước đó.", bold_lead="Phục hồi chậm: ")

h2("1.2. Linux và Ubuntu Server trong DevOps")
h3("1.2.1. Vì sao dùng Linux trong DevOps")
p("Linux là nền tảng tiêu chuẩn của phần lớn máy chủ đám mây nhờ tính ổn định, khả năng "
  "tự động hoá bằng dòng lệnh và miễn phí bản quyền. Quan trọng hơn, Docker dựa trực "
  "tiếp vào hai tính năng của nhân Linux: cgroups để giới hạn CPU, RAM và namespaces "
  "để cô lập tiến trình, mạng và hệ thống tệp. Với hệ thống sửa lỗi chính tả, việc giới "
  "hạn RAM cho tiến trình suy luận là cần thiết vì mô hình chiếm gần 1 GB chỉ riêng "
  "phần trọng số.")
p("Bản phân phối được chọn là Ubuntu Server 24.04 LTS: chu kỳ hỗ trợ dài, kho gói ổn "
  "định, tương thích tốt với Docker Engine và các thư viện Python hiện đại.")

h3("1.2.2. Các lệnh Linux cơ bản trong DevOps")
bullet("ls, cd, pwd, mkdir, cp, mv, rm, cat, nano.", bold_lead="Quản lý tệp và thư mục: ")
bullet("chmod, chown, sudo; đặc biệt chmod 400 cho tệp khoá SSH.",
       bold_lead="Phân quyền: ")
bullet("ps aux, top, df -h, free -m, kill.", bold_lead="Tiến trình và tài nguyên: ")
bullet("ss -tulpn, curl, ufw, systemctl, docker compose ps.",
       bold_lead="Mạng và dịch vụ: ")

h2("1.3. Docker và Containerization")
h3("1.3.1. Container là gì?")
p("Container là một đơn vị phần mềm đóng gói mã nguồn ứng dụng cùng toàn bộ thư viện "
  "phụ thuộc, tệp cấu hình và môi trường thực thi. Container chia sẻ nhân của hệ điều "
  "hành máy chủ nhưng vẫn cô lập tài nguyên, nhờ đó khởi động nhanh và tiết kiệm bộ nhớ "
  "hơn máy ảo.")
p("Trong dự án này, container chứa Python 3.10, PyTorch bản CPU, Transformers, "
  "SentencePiece, mã nguồn trong thư mục src/ và app.py. Tệp trọng số 903,8 MB không "
  "được đóng vào image mà được gắn từ volume để có thể thay mô hình mà không cần build "
  "lại.")

h3("1.3.2. Ưu điểm của container so với máy ảo")
bullet("container dùng chung nhân Linux, không tốn hàng GB cho hệ điều hành khách.",
       bold_lead="Tiết kiệm tài nguyên: ")
bullet("container khởi động trong vài giây, máy ảo mất hàng chục giây đến vài phút.",
       bold_lead="Tốc độ: ")
bullet("có thể chạy nhiều container trên một VPS cấu hình khiêm tốn.",
       bold_lead="Mật độ: ")
bullet("image là bất biến, cùng một image chạy giống nhau ở mọi môi trường.",
       bold_lead="Tính nhất quán: ")

h3("1.3.3. Các thành phần chính trong Docker")
bullet("bản mẫu chỉ đọc, được build từ Dockerfile và lưu trong registry.",
       bold_lead="Image: ")
bullet("thực thể đang chạy của image, có thể dừng, khởi động lại, xem log.",
       bold_lead="Container: ")
bullet("vùng lưu trữ bền vững, tồn tại độc lập với vòng đời container; dùng cho "
       "checkpoint, log và dữ liệu.",
       bold_lead="Volume: ")
bullet("mạng riêng để các container giao tiếp bằng tên dịch vụ thay vì địa chỉ IP.",
       bold_lead="Network: ")

h3("1.3.4. Dockerfile, tag, registry và multi-stage build")
p("Dockerfile là tệp mô tả các bước tạo image. Kỹ thuật multi-stage build cho phép dùng "
  "một stage để cài đặt và biên dịch, sau đó chỉ sao chép kết quả sang stage runtime nhỏ "
  "hơn. Nhờ vậy image cuối không chứa trình biên dịch và bộ nhớ đệm của pip.")
p("Mỗi image được gắn tag. Trong pipeline, tag được lấy theo commit SHA để bảo đảm mỗi "
  "lần triển khai truy vết được đúng phiên bản mã nguồn và mô hình. Registry dùng trong "
  "đồ án là GitHub Container Registry (GHCR).")

h2("1.4. Docker Compose")
p("Docker Compose cho phép mô tả nhiều dịch vụ trong một tệp YAML: image, cổng, biến "
  "môi trường, volume, mạng và thứ tự phụ thuộc. Với hệ thống này, Compose định nghĩa "
  "ba dịch vụ: api (tiến trình suy luận), nginx (reverse proxy) và certbot (gia hạn "
  "chứng chỉ). Toàn bộ hệ thống khởi chạy bằng một lệnh duy nhất, thay vì chuỗi lệnh "
  "docker run dài và dễ sai.")

h2("1.5. Web Server và Reverse Proxy (Nginx)")
h3("1.5.1. Phục vụ nội dung và nén")
p("Nginx có thể trả trực tiếp các tệp tĩnh và nén Gzip trước khi gửi cho trình duyệt, "
  "giảm dung lượng truyền tải. Với ứng dụng Python, việc để Nginx phục vụ phần giao "
  "diện giúp giảm số request phải đi qua tiến trình suy luận.")

h3("1.5.2. Reverse proxy cho backend")
p("Nginx nhận request từ Internet và chuyển tiếp tới dịch vụ nội bộ. Cách làm này che "
  "giấu cổng thật của ứng dụng, cho phép thêm TLS, giới hạn tần suất và ghi log truy "
  "cập ở một điểm duy nhất.")

h3("1.5.3. Định tuyến tiền tố /api")
p("Các request bắt đầu bằng /api/ được chuyển tới http://api:8000/api/. Vì giao diện và "
  "API cùng một tên miền, trình duyệt không gặp lỗi CORS. Nginx cũng gắn thêm các header "
  "X-Real-IP, X-Forwarded-For và X-Forwarded-Proto để ứng dụng biết địa chỉ thật và giao "
  "thức gốc.")

h3("1.5.4. Vì sao cần Nginx với ứng dụng suy luận")
p("Mỗi request sửa lỗi tốn trung bình khoảng 2,3 giây CPU khi chạy beam search 4. Nếu "
  "không có giới hạn tần suất, một lượng nhỏ request đồng thời cũng đủ làm hàng đợi "
  "tăng và dịch vụ không phản hồi. Nginx cung cấp limit_req để chặn các đợt gọi quá dày "
  "và trả về mã 429 thay vì treo toàn bộ tiến trình.")

h2("1.6. Hạ tầng đám mây và các khái niệm cơ bản")
h3("1.6.1. Máy chủ ảo (EC2 / VPS)")
p("Máy chủ ảo cung cấp một hệ điều hành đầy đủ với quyền quản trị, phù hợp để cài Docker "
  "và chạy các tiến trình lâu dài. Với hệ thống này, cấu hình tối thiểu nên là 2 vCPU, "
  "4 GB RAM và 40 GB SSD: 4 GB đủ để nạp mô hình và phục vụ vài request đồng thời, 40 GB "
  "đủ chứa image, checkpoint và log.")

h3("1.6.2. Security Group và tường lửa")
p("Security Group là tường lửa ảo ở cấp instance. Nguyên tắc là chỉ mở những cổng thật "
  "sự cần: 22 cho SSH quản trị, 80 và 443 cho HTTP/HTTPS. Cổng nội bộ của API (8000) "
  "không được mở ra Internet; container chỉ truy cập được qua mạng Docker.")

h3("1.6.3. Key Pair")
p("Đăng nhập máy chủ bằng cặp khoá RSA thay cho mật khẩu. Tệp khoá riêng phải được phân "
  "quyền chặt (chmod 400 trên Linux, icacls trên Windows) và với pipeline thì lưu trong "
  "GitHub Secrets, không đưa vào mã nguồn.")

h2("1.7. CI/CD và GitHub Actions")
p("CI (Continuous Integration) là quá trình tự động kiểm tra mã nguồn mỗi khi có thay "
  "đổi. CD (Continuous Delivery/Deployment) là quá trình tự động đóng gói và đưa phiên "
  "bản mới lên máy chủ. GitHub Actions mô tả pipeline bằng tệp YAML trong thư mục "
  ".github/workflows, gồm các job chạy trên runner và có thể dùng secrets để bảo vệ "
  "thông tin nhạy cảm.")
p("Với một dự án học sâu, pipeline cần thêm một bước kiểm tra chất lượng mô hình: chạy "
  "một bộ câu hồi quy nhỏ và so sánh chỉ số với ngưỡng. Nếu chất lượng giảm, pipeline "
  "dừng lại thay vì triển khai một mô hình kém hơn.")

h2("1.8. Đặc thù triển khai mô hình học sâu")
bullet("trọng số ViT5-base chiếm 903,8 MB, việc nạp mất hàng chục giây, nên "
       "healthcheck phải có start_period đủ lớn để không restart oan.",
       bold_lead="Kích thước checkpoint: ")
bullet("gói Transformers thay đổi cấu trúc vocab Unigram giữa các phiên bản, có thể gây "
       "lỗi khi nạp tokenizer T5. Cần pin phiên bản trong image.",
       bold_lead="Phụ thuộc phiên bản: ")
bullet("mỗi lần finetune tạo ra một checkpoint mới; cần đặt tên, lưu vết và có khả năng "
       "quay lui.", bold_lead="Quản lý phiên bản mô hình: ")
bullet("khi mô hình lỗi, hệ thống vẫn nên phục vụ được bằng động cơ luật N-gram để "
       "không mất hoàn toàn chức năng.", bold_lead="Phương án dự phòng: ")
doc.add_page_break()

# ================================================================== CHƯƠNG 2
h1("CHƯƠNG 2. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG")

h2("2.1. Mục tiêu của đồ án")
h3("2.1.1. Mục tiêu tổng quát")
p("Đóng gói hệ thống phát hiện và sửa lỗi chính tả tiếng Việt thành một dịch vụ chạy "
  "trên máy chủ, có thể triển khai lặp lại, tự động cập nhật và tự phục hồi khi gặp sự "
  "cố.")

h3("2.1.2. Mục tiêu cụ thể")
bullet("Xây dựng Dockerfile multi-stage cho ứng dụng Python có mô hình lớn.")
bullet("Điều phối dịch vụ bằng Docker Compose, tách API và reverse proxy.")
bullet("Cấu hình Nginx với TLS, nén, định tuyến /api và giới hạn tần suất.")
bullet("Tự động hoá kiểm thử, build image, đẩy registry và triển khai qua GitHub Actions.")
bullet("Bảo đảm chất lượng mô hình không suy giảm sau khi đóng gói thông qua bộ chỉ số định lượng.")

h3("2.1.3. Mục tiêu về vận hành")
p("Hệ thống phải khởi động lại tự động khi container dừng, có endpoint kiểm tra sức khoẻ, "
  "ghi log ra ngoài vòng đời container và cho phép quay lui về phiên bản trước trong vài "
  "phút.")

h2("2.2. Bài toán cần giải quyết")
h3("2.2.1. Môi trường thư viện không đồng nhất")
p("Hệ thống phụ thuộc PyTorch, Transformers, SentencePiece, underthesea và pyvi. Chỉ "
  "một thay đổi nhỏ về phiên bản cũng có thể làm tokenizer nạp sai hoặc kết quả suy luận "
  "khác đi. Việc cài đặt thủ công trên máy chủ không bảo đảm tái lập được môi trường đã "
  "dùng khi huấn luyện.")

h3("2.2.2. Checkpoint lớn và thời gian khởi động")
p("Tệp model.safetensors nặng 903,8 MB. Nếu đóng vào image, mỗi lần build và mỗi lần kéo "
  "image sẽ tốn băng thông lớn, đồng thời không thể thay mô hình nhanh. Ngoài ra, quá "
  "trình nạp mô hình kéo dài hàng chục giây nên cơ chế healthcheck mặc định dễ đánh giá "
  "sai trạng thái.")

h3("2.2.3. Hai động cơ song song")
p("Ngoài mô hình ViT5, hệ thống còn động cơ luật N-gram dùng từ điển âm tiết, bigram và "
  "khoảng cách Levenshtein. Hai động cơ cần được bọc sau cùng một giao diện API để giao "
  "diện web không phải biết chi tiết triển khai, đồng thời cho phép so sánh trực tiếp "
  "chất lượng.")

h3("2.2.4. Cập nhật mô hình không gián đoạn")
p("Sau mỗi lần finetune, cần thay checkpoint mới mà không làm gián đoạn dịch vụ. Cách làm "
  "được chọn là đặt checkpoint trong volume chỉ đọc, còn image chứa mã nguồn; khi cập "
  "nhật chỉ cần thay thư mục checkpoint và khởi động lại container api.")

h2("2.3. Phân tích yêu cầu hệ thống")
h3("2.3.1. Yêu cầu chức năng")
table(["Mã", "Yêu cầu", "Mô tả"],
      [["F1", "Sửa lỗi bằng mô hình", "Nhận văn bản, trả về câu đã sửa và danh sách vị trí sai."],
       ["F2", "Sửa lỗi bằng luật", "Cho phép chọn động cơ N-gram để so sánh hoặc dự phòng."],
       ["F3", "So sánh hai động cơ", "Trả về kết quả và thời gian xử lý của từng động cơ."],
       ["F4", "Sinh dữ liệu nhiễu", "Tạo câu lỗi nhân tạo phục vụ kiểm thử độ bền."],
       ["F5", "Kiểm tra sức khoẻ", "Trả về trạng thái nạp mô hình và thông tin phiên bản."]],
      "Bảng 2.1. Yêu cầu chức năng của hệ thống", widths=[1.6, 4.2, 9.2])

h3("2.3.2. Yêu cầu phi chức năng")
table(["Mã", "Yêu cầu", "Mô tả"],
      [["N1", "Tái lập", "Cùng một image chạy giống nhau trên mọi máy chủ."],
       ["N2", "Hiệu năng", "Một câu ngắn phản hồi trong khoảng 2–4 giây trên CPU."],
       ["N3", "Bảo mật", "Chỉ mở 22/80/443; API không lộ cổng; khoá SSH không vào mã nguồn."],
       ["N4", "Sẵn sàng", "Container tự khởi động lại; healthcheck xác nhận mô hình đã nạp."],
       ["N5", "Truy vết", "Mỗi bản triển khai gắn tag commit SHA."]],
      "Bảng 2.2. Yêu cầu phi chức năng", widths=[1.6, 4.2, 9.2])

h3("2.3.3. Giới hạn của hệ thống")
p("Mô hình chạy trên CPU nên chưa phù hợp với tải cao; độ dài câu bị giới hạn bởi "
  "max_length khi suy luận; chất lượng phát hiện lỗi còn thấp hơn chất lượng sửa lỗi, "
  "thể hiện ở tỷ lệ dương tính giả. Báo cáo ghi nhận các giới hạn này thay vì che giấu.")

h2("2.4. Kiến trúc tổng quan của hệ thống")
h3("2.4.1. Kiến trúc ba tầng được container hoá")
p("Tầng trình bày gồm trình duyệt và Nginx; tầng suy luận gồm tiến trình app.py chứa hai "
  "động cơ; tầng dữ liệu gồm checkpoint, dữ liệu đã xử lý và log. Ba tầng giao tiếp qua "
  "mạng Docker nội bộ, chỉ Nginx mở cổng ra ngoài.")
figure("architecture.png", "Hình 2.1. Kiến trúc ba tầng của hệ thống", width=15.0)

h3("2.4.2. Nguyên tắc thiết kế")
bullet("mỗi thành phần chạy trong một container riêng, giao tiếp qua tên dịch vụ.")
bullet("image chứa mã nguồn, volume chứa dữ liệu và trọng số.")
bullet("cấu hình đưa vào biến môi trường, không hard-code trong mã.")
bullet("mọi thay đổi đi qua pipeline, không thao tác trực tiếp trên máy chủ.")

h2("2.5. Sơ đồ triển khai")
p("Sơ đồ dưới đây mô tả luồng request từ trình duyệt tới mô hình và các thành phần phụ "
  "trợ trên máy chủ: Nginx nhận TLS, chuyển /api/* tới container api; certbot quản lý "
  "chứng chỉ; checkpoint được gắn vào container ở chế độ chỉ đọc.")
figure("topology.png", "Hình 2.2. Sơ đồ triển khai trên máy chủ", width=14.0)

h3("2.5.1. Công nghệ sử dụng")
table(["Thành phần", "Công nghệ", "Vai trò"],
      [["Mô hình", "VietAI/vit5-base (T5 Seq2Seq, ~250M tham số)", "Sửa lỗi theo ngữ cảnh"],
       ["Runtime", "Python 3.10, PyTorch 2.x (CPU), Transformers 4.45", "Suy luận"],
       ["Web", "http.server trong app.py, cổng 7860/8000", "Giao diện và API"],
       ["Đóng gói", "Docker multi-stage, Docker Compose", "Container hoá"],
       ["Proxy", "Nginx 1.27 + certbot", "TLS, định tuyến, rate limit"],
       ["Hạ tầng", "Ubuntu 24.04 LTS trên VPS/EC2", "Máy chủ"],
       ["CI/CD", "GitHub Actions + GHCR", "Tự động hoá triển khai"],
       ["Dữ liệu", "VSEC Corpus 9.341 câu, 11.202 lỗi", "Huấn luyện và đánh giá"]],
      "Bảng 2.3. Công nghệ và công cụ sử dụng", widths=[3.0, 7.5, 5.0])
doc.add_page_break()

# ================================================================== CHƯƠNG 3
h1("CHƯƠNG 3. TRIỂN KHAI HỆ THỐNG VỚI DOCKER VÀ DEVOPS")
p("Chương này trình bày cấu hình triển khai của hệ thống. Các tệp cấu hình được đặt trong "
  "thư mục deploy/ của kho mã nguồn và được pipeline sử dụng trực tiếp, bảo đảm mọi thay "
  "đổi đều đi qua quy trình kiểm soát phiên bản.")

h2("3.1. Cấu hình hạ tầng máy chủ")
h3("3.1.1. Khởi tạo máy chủ và khoá SSH")
p("Máy chủ được khởi tạo với Ubuntu 24.04 LTS, 2 vCPU, 4 GB RAM và 40 GB SSD. Cặp khoá "
  "RSA được tạo tại thời điểm khởi tạo; tệp khoá riêng được phân quyền chmod 400 (hoặc "
  "icacls trên Windows) trước khi kết nối:")
code("ssh -i ~/.ssh/spellcheck_key ubuntu@<IP_MAY_CHU>\n"
     "chmod 400 ~/.ssh/spellcheck_key", "Cấu hình 3.1. Kết nối SSH tới máy chủ")

h3("3.1.2. Security Group và tường lửa")
p("Chỉ ba cổng được mở ra Internet: 22 cho SSH, 80 và 443 cho HTTP/HTTPS. Cổng 8000 của "
  "API chỉ tồn tại trong mạng Docker. Tường lửa UFW được cấu hình tương ứng để tránh "
  "trường hợp Security Group và tường lửa hệ điều hành lệch nhau:")
code("sudo ufw allow 22/tcp\n"
     "sudo ufw allow 80/tcp\n"
     "sudo ufw allow 443/tcp\n"
     "sudo ufw enable")

h2("3.2. Cài đặt môi trường và Docker")
p("Docker Engine và Compose Plugin được cài từ kho chính thức. Sau khi cài, mã nguồn "
  "được đưa lên máy chủ, tệp biến môi trường được tạo từ mẫu và thư mục checkpoint được "
  "kiểm tra sự tồn tại của tệp trọng số:")
code("docker --version && docker compose version\n"
     "git clone <REPO_URL> /home/ubuntu/spellcheck\n"
     "cd /home/ubuntu/spellcheck\n"
     "cp .env.example .env\n"
     "ls -lh checkpoints/vit5_spelling_corrector/best_model/model.safetensors")
p("Dung lượng thực tế của tệp trọng số là 903,8 MB, tương ứng khoảng 862 MiB. Đây là lý do "
  "thư mục checkpoints được gắn từ máy chủ thay vì đóng vào image.")

h2("3.3. Dockerfile multi-stage")
p("Dockerfile gồm hai stage. Stage builder cài các gói Python vào một thư mục riêng; "
  "stage runtime chỉ sao chép thư mục đó cùng mã nguồn, tạo người dùng thường và khai "
  "báo healthcheck.")
code('FROM python:3.10-slim AS builder\n'
     'WORKDIR /build\n'
     'ENV PIP_NO_CACHE_DIR=1\n'
     'COPY requirements.txt .\n'
     'RUN pip install --prefix=/install -r requirements.txt \\\n'
     '    --extra-index-url https://download.pytorch.org/whl/cpu\n'
     '\n'
     'FROM python:3.10-slim AS runtime\n'
     'ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n'
     'WORKDIR /app\n'
     'COPY --from=builder /install /usr/local\n'
     'COPY app.py predict.py ./\n'
     'COPY src ./src\n'
     'RUN useradd -m appuser && chown -R appuser /app\n'
     'USER appuser\n'
     'EXPOSE 8000\n'
     'HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \\\n'
     '  CMD curl -fs http://127.0.0.1:8000/api/status || exit 1\n'
     'CMD ["python", "app.py", "--port", "8000"]',
     "Cấu hình 3.2. Dockerfile multi-stage cho ứng dụng suy luận")

p("Ba quyết định kỹ thuật đáng chú ý: dùng bản CPU của PyTorch để giảm dung lượng image; "
  "chạy bằng người dùng appuser thay vì root; và đặt start-period của healthcheck là 90 "
  "giây để phù hợp với thời gian nạp mô hình.")

h2("3.4. Docker Compose")
p("Tệp compose định nghĩa ba dịch vụ và hai điểm cấu hình quan trọng: volume chỉ đọc cho "
  "checkpoint và điều kiện service_healthy để Nginx chỉ nhận traffic khi API đã nạp xong "
  "mô hình.")
code('services:\n'
     '  api:\n'
     '    build: .\n'
     '    image: ${API_IMAGE}\n'
     '    container_name: spellcheck_api\n'
     '    restart: unless-stopped\n'
     '    env_file: [.env]\n'
     '    volumes:\n'
     '      - ./checkpoints:/app/checkpoints:ro\n'
     '      - ./logs:/app/logs\n'
     '    networks: [spellcheck-net]\n'
     '    mem_limit: 3g\n'
     '    healthcheck:\n'
     '      test: ["CMD-SHELL", "curl -fs localhost:8000/api/status"]\n'
     '      interval: 20s\n'
     '      timeout: 5s\n'
     '      start_period: 90s\n'
     '\n'
     '  nginx:\n'
     '    image: nginx:1.27-alpine\n'
     '    ports: ["80:80", "443:443"]\n'
     '    volumes:\n'
     '      - ./deploy/nginx.conf:/etc/nginx/conf.d/default.conf:ro\n'
     '      - ./certbot/conf:/etc/letsencrypt:ro\n'
     '    depends_on:\n'
     '      api: {condition: service_healthy}\n'
     '    networks: [spellcheck-net]\n'
     '\n'
     'networks:\n'
     '  spellcheck-net:\n'
     '    driver: bridge',
     "Cấu hình 3.3. docker-compose.prod.yml")

h2("3.5. Cấu hình Nginx")
p("Nginx đảm nhận ba việc: chuyển hướng HTTP sang HTTPS, chuyển tiếp /api/ tới container "
  "api và giới hạn tần suất gọi. Giới hạn được đặt ở mức 5 request/giây cho mỗi địa chỉ "
  "IP, với burst 5, phù hợp với chi phí khoảng 2,3 giây CPU cho mỗi câu.")
code('limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;\n'
     '\n'
     'server {\n'
     '    listen 80;\n'
     '    server_name spellcheck.example.com;\n'
     '    return 301 https://$host$request_uri;\n'
     '}\n'
     '\n'
     'server {\n'
     '    listen 443 ssl http2;\n'
     '    server_name spellcheck.example.com;\n'
     '    ssl_certificate     /etc/letsencrypt/live/spellcheck.example.com/fullchain.pem;\n'
     '    ssl_certificate_key /etc/letsencrypt/live/spellcheck.example.com/privkey.pem;\n'
     '    client_max_body_size 2m;\n'
     '    gzip on;\n'
     '    gzip_types text/css application/javascript application/json;\n'
     '\n'
     '    location /api/ {\n'
     '        limit_req zone=api burst=5 nodelay;\n'
     '        proxy_pass http://api:8000/api/;\n'
     '        proxy_set_header X-Real-IP $remote_addr;\n'
     '        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n'
     '        proxy_set_header X-Forwarded-Proto $scheme;\n'
     '        proxy_read_timeout 120s;\n'
     '    }\n'
     '}',
     "Cấu hình 3.4. Reverse proxy và giới hạn tần suất")

h2("3.6. Biến môi trường và bảo mật")
p("Các thông số thay đổi theo môi trường được đưa vào tệp .env, không nằm trong image. "
  "Tệp này được liệt kê trong .gitignore và chỉ tồn tại trên máy chủ.")
table(["Biến", "Giá trị mẫu", "Ý nghĩa"],
      [["API_IMAGE", "ghcr.io/<owner>/<repo>/api:<sha>", "Image cần chạy"],
       ["MODEL_PATH", "checkpoints/vit5_spelling_corrector/best_model", "Đường dẫn checkpoint"],
       ["MAX_LENGTH", "96", "Độ dài tối đa khi suy luận"],
       ["NUM_BEAMS", "4", "Độ rộng beam search"],
       ["PORT", "8000", "Cổng nội bộ của API"],
       ["TZ", "Asia/Ho_Chi_Minh", "Múi giờ ghi log"]],
      "Bảng 3.1. Các biến môi trường của hệ thống", widths=[3.0, 8.0, 5.0])
p("Với pipeline, ba giá trị nhạy cảm được lưu trong GitHub Secrets: VPS_HOST, VPS_USER và "
  "VPS_SSH_KEY. Không có khoá riêng hay mật khẩu nào xuất hiện trong mã nguồn, trong log "
  "hoặc trong image.")

h2("3.7. Pipeline CI/CD với GitHub Actions")
p("Pipeline gồm ba job nối tiếp. Job test kiểm tra cú pháp, nạp tokenizer và chạy bộ câu "
  "hồi quy. Job build tạo image và đẩy lên GHCR với tag là commit SHA. Job deploy dùng "
  "SSH để cập nhật máy chủ và gọi endpoint kiểm tra sức khoẻ.")
code('name: ci-cd\n'
     'on:\n'
     '  push: { branches: [main] }\n'
     '  pull_request:\n'
     '\n'
     'jobs:\n'
     '  test:\n'
     '    runs-on: ubuntu-latest\n'
     '    steps:\n'
     '      - uses: actions/checkout@v4\n'
     '      - uses: actions/setup-python@v5\n'
     '        with: { python-version: "3.10" }\n'
     '      - run: pip install -r requirements.txt\n'
     '      - run: python -c "import app"\n'
     '\n'
     '  build:\n'
     '    needs: test\n'
     '    runs-on: ubuntu-latest\n'
     '    steps:\n'
     '      - uses: actions/checkout@v4\n'
     '      - uses: docker/login-action@v3\n'
     '        with:\n'
     '          registry: ghcr.io\n'
     '          username: ${{ github.actor }}\n'
     '          password: ${{ secrets.GITHUB_TOKEN }}\n'
     '      - run: |\n'
     '          IMG=ghcr.io/${{ github.repository }}/api:${{ github.sha }}\n'
     '          docker build -t $IMG .\n'
     '          docker push $IMG\n'
     '\n'
     '  deploy:\n'
     '    needs: build\n'
     '    runs-on: ubuntu-latest\n'
     '    steps:\n'
     '      - uses: appleboy/ssh-action@v1\n'
     '        with:\n'
     '          host: ${{ secrets.VPS_HOST }}\n'
     '          username: ${{ secrets.VPS_USER }}\n'
     '          key: ${{ secrets.VPS_SSH_KEY }}\n'
     '          script: |\n'
     '            cd /home/ubuntu/spellcheck\n'
     '            export API_IMAGE=ghcr.io/${{ github.repository }}/api:${{ github.sha }}\n'
     '            docker compose -f docker-compose.prod.yml pull api\n'
     '            docker compose -f docker-compose.prod.yml up -d\n'
     '            sleep 20 && curl -fs localhost/api/status || ./deploy/rollback.sh',
     "Cấu hình 3.5. Pipeline CI/CD")

h2("3.8. Quy trình triển khai và rollback")
p("Quy trình chuẩn gồm năm bước: push mã nguồn, pipeline chạy kiểm thử, build và đẩy "
  "image, máy chủ kéo image mới, cuối cùng kiểm tra /api/status. Nếu endpoint không trả "
  "về trạng thái nạp mô hình thành công, kịch bản rollback sẽ đưa biến API_IMAGE về tag "
  "của lần triển khai trước và chạy lại compose.")
code('#!/usr/bin/env bash\n'
     'set -euo pipefail\n'
     'cd /home/ubuntu/spellcheck\n'
     'PREV=$(cat .last_good_image 2>/dev/null || true)\n'
     'if [ -n "$PREV" ]; then\n'
     '  export API_IMAGE="$PREV"\n'
     '  docker compose -f docker-compose.prod.yml up -d api\n'
     '  echo "rollback to $PREV"\n'
     'fi',
     "Cấu hình 3.6. Kịch bản rollback")
p("Sau khi triển khai thành công, tag của image được ghi lại vào tệp .last_good_image để "
  "lần sau có thể quay lui. Cách làm này đơn giản nhưng đủ cho quy mô đồ án và không phụ "
  "thuộc dịch vụ bên ngoài.")
doc.add_page_break()

# ================================================================== CHƯƠNG 4
h1("CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ")

h2("4.1. Môi trường và dữ liệu thực nghiệm")
p("Đánh giá được thực hiện trên tập test của VSEC Corpus gồm 935 câu, không dùng để chọn "
  "mô hình. Cấu hình suy luận: max_length 96, beam search 4, no_repeat_ngram_size 4, "
  "repetition_penalty 1,2. Toàn bộ quá trình chạy trên CPU, mất 2.140 giây cho 935 câu, "
  "tương đương khoảng 2,3 giây mỗi câu.")
p("Bộ dữ liệu gốc có 9.341 câu với 11.202 lỗi được gán nhãn thủ công, chia thành 7.472 "
  "câu huấn luyện, 934 câu kiểm định và 935 câu kiểm thử. Câu dài nhất 64 âm tiết, trung "
  "bình 30,24 âm tiết.")

h2("4.2. Kết quả định lượng của mô hình")
p("Tám chỉ số được tính trên cùng một tập test cho cả mô hình ViT5 đã finetune và động cơ "
  "luật N-gram dùng làm đối chứng.")
table(["Chỉ số", "Ý nghĩa", "ViT5", "Baseline"],
      [["Exact Match", "Tỷ lệ câu đúng hoàn toàn", "73,48 %", "6,84 %"],
       ["Token Accuracy", "Tỷ lệ âm tiết đúng", "95,59 %", "93,57 %"],
       ["BLEU-4", "Tương đồng n-gram (0–100)", "97,21", "0,85"],
       ["CER", "Tỷ lệ lỗi ký tự (thấp tốt)", "1,25 %", "2,04 %"],
       ["WER", "Tỷ lệ lỗi từ (thấp tốt)", "5,63 %", "6,52 %"],
       ["Detection Precision", "Vị trí sửa đúng / đã sửa", "62,20 %", "27,18 %"],
       ["Detection Recall", "Lỗi phát hiện / lỗi thực", "89,26 %", "30,49 %"],
       ["Detection F1", "Trung bình điều hoà P và R", "73,31 %", "28,74 %"]],
      "Bảng 4.1. Kết quả 8 chỉ số trên 935 câu test", widths=[3.6, 6.2, 3.2, 3.0])

h2("4.3. Phân tích kết quả")
p("Mô hình ViT5 vượt baseline ở cả tám chỉ số. Exact Match tăng từ 6,84 % lên 73,48 %, "
  "nghĩa là gần ba phần tư số câu được sửa đúng hoàn toàn. BLEU-4 đạt 97,21 cho thấy câu "
  "dự đoán rất sát câu chuẩn; CER 1,25 % tương ứng khoảng một lỗi ký tự trên tám mươi ký "
  "tự.")
p("Chỉ số phát hiện lỗi cho thấy bức tranh rõ hơn: Recall 89,26 % nghĩa là hệ thống bỏ sót "
  "ít lỗi, nhưng Precision 62,20 % nghĩa là còn nhiều vị trí bị sửa dù vốn đã đúng. Cụ "
  "thể, trong 2.644 vị trí bị thay đổi có 1.530 vị trí đúng (TP), 930 vị trí sửa nhầm "
  "(FP) và 184 lỗi bị bỏ sót (FN). Đây là hạn chế cần nêu trung thực: mô hình có xu hướng "
  "chuẩn hoá cả những chỗ không sai, đặc biệt với tên riêng và từ hiếm.")
p("Về mặt vận hành, việc đóng gói không làm thay đổi chất lượng: kết quả trên được tạo ra "
  "từ chính checkpoint được mount vào container, với cùng cấu hình suy luận. Đây là bằng "
  "chứng cho thấy môi trường container tái lập đúng môi trường huấn luyện.")

h2("4.4. Kịch bản kiểm thử sau triển khai")
p("Bảng dưới đây liệt kê các kiểm thử cần thực hiện sau khi container khởi chạy. Cột kết "
  "quả được điền sau khi chạy thực tế trên máy chủ, kèm ảnh chụp màn hình làm minh chứng.")
table(["STT", "Kiểm thử", "Cách thực hiện", "Kỳ vọng"],
      [["1", "Container hoạt động", "docker compose ps", "api và nginx ở trạng thái Up (healthy)"],
       ["2", "Mô hình đã nạp", "curl -s localhost/api/status", '{"model_loaded": true}'],
       ["3", "Sửa lỗi bằng AI", "POST /api/correct_spelling (mode=ai)", "200, có corrected_text"],
       ["4", "Sửa lỗi bằng luật", "POST /api/correct_spelling (mode=rule)", "200, phản hồi dưới 0,5 giây"],
       ["5", "So sánh hai động cơ", "POST /api/compare_engines", "200, có ai và rule"],
       ["6", "Sinh nhiễu", "POST /api/generate_noise", "200, có noisy_text"],
       ["7", "Chặn truy cập trực tiếp", "curl http://<IP>:8000", "Không kết nối được"],
       ["8", "TLS và chuyển hướng", "curl -I http://<domain>", "301 sang HTTPS"]],
      "Bảng 4.2. Kịch bản kiểm thử API và hạ tầng", widths=[1.2, 4.2, 5.6, 5.0],
      font_size=11)
figure("inference.png", "Hình 4.1. Luồng suy luận từ văn bản lỗi tới kết quả trả về", width=15.5)

p("Sau khi khởi chạy dịch vụ, endpoint /api/status trả về trạng thái sẵn sàng cùng tên "
  "checkpoint đang nạp. Giao diện sửa lỗi nhận câu đầu vào, gọi mô hình và hiển thị kết quả "
  "theo hai cột: câu gốc với các từ sai được đánh dấu đỏ, câu đã sửa với các từ thay thế "
  "được đánh dấu xanh. Bên dưới là bảng chi tiết từng vị trí sai kèm phân loại lỗi.")
figure("ui_result_ai.png", "Hình 4.2. Giao diện sửa lỗi với kết quả thực tế từ mô hình ViT5",
       width=15.0)
p("Trong lần chạy minh hoạ, câu đầu vào gồm 24 từ được xử lý trong 5.258 ms, phát hiện và "
  "sửa 5 vị trí. Đây là mức thời gian cao hơn trung bình 2,3 giây/câu đo trên toàn tập test "
  "vì lần gọi đầu tiên phải khởi tạo bộ nhớ đệm cho mô hình. Kết quả cũng cho thấy hạn chế "
  "đã nêu: một số vị trí được sửa chưa đúng, phản ánh đúng chỉ số Precision 62,20 %.")
figure("ui_compare.png", "Hình 4.3. Đối sánh trực tiếp ViT5 và Rule Engine trên cùng một câu",
       width=12.5)
p("Chức năng đối sánh chạy song song hai động cơ trên cùng một câu. Với câu thử nghiệm, cả "
  "hai đều trả về câu đã sửa đúng; mô hình ViT5 mất 1.213 ms còn động cơ luật gần như tức "
  "thời. Sự khác biệt này phù hợp với thiết kế: động cơ luật tra từ điển, còn mô hình phải "
  "giải mã bằng beam search.")

h2("4.5. Đánh giá ưu điểm và hạn chế")
h3("4.5.1. Ưu điểm")
bullet("cùng một image chạy giống nhau ở máy cá nhân, CI và máy chủ.",
       bold_lead="Môi trường đồng nhất: ")
bullet("thay checkpoint hoặc mã nguồn đều đi qua pipeline, có tag để truy vết.",
       bold_lead="Cập nhật có kiểm soát: ")
bullet("healthcheck, restart policy và rollback giúp giảm thời gian gián đoạn.",
       bold_lead="Tự phục hồi: ")
bullet("API không lộ cổng, TLS bắt buộc, khoá SSH nằm trong secrets.",
       bold_lead="Bảo mật cơ bản: ")

h3("4.5.2. Hạn chế")
bullet("một câu tốn khoảng 2,3 giây CPU; chưa đáp ứng được tải đồng thời lớn.",
       bold_lead="Hiệu năng: ")
bullet("930 vị trí bị sửa nhầm trên tập test cho thấy mô hình còn chuẩn hoá quá mức.",
       bold_lead="Độ chính xác phát hiện: ")
bullet("hệ thống chưa có hàng đợi, chưa có giám sát tập trung và cảnh báo.",
       bold_lead="Thiếu thành phần quan sát: ")
bullet("chưa có bộ kiểm thử tự động cho mã nguồn, pipeline mới dừng ở kiểm tra cú pháp.",
       bold_lead="Kiểm thử: ")
doc.add_page_break()

# ================================================================== CHƯƠNG 5
h1("CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")

h2("5.1. Kết quả đạt được")
p("Đồ án đã đưa một hệ thống học sâu từ môi trường thí nghiệm lên mô hình vận hành theo "
  "chuẩn DevOps: đóng gói bằng Docker multi-stage, điều phối bằng Docker Compose, đặt sau "
  "Nginx với TLS và giới hạn tần suất, triển khai trên Ubuntu 24.04 LTS và tự động hoá "
  "bằng GitHub Actions. Các cấu hình được mô tả bằng mã nên có thể tái lập trên một máy "
  "chủ mới trong vài lệnh.")
p("Về chất lượng mô hình, hệ thống đạt Exact Match 73,48 %, BLEU-4 97,21, CER 1,25 % và "
  "Detection F1 73,31 % trên 935 câu test, vượt baseline luật ở cả tám chỉ số. Kết quả này "
  "được tạo ra từ chính checkpoint được mount vào container, chứng minh việc đóng gói "
  "không làm sai lệch hành vi của mô hình.")

h2("5.2. Bài học kinh nghiệm")
bullet("với mô hình lớn, tách trọng số khỏi image giúp giảm thời gian build và cho phép "
       "thay mô hình nhanh.", bold_lead="Thiết kế image: ")
bullet("healthcheck phải phản ánh trạng thái thật của ứng dụng, không chỉ kiểm tra cổng "
       "đã mở.", bold_lead="Kiểm tra sức khoẻ: ")
bullet("pin phiên bản thư viện là cách rẻ nhất để tránh lỗi tái lập khó tìm.",
       bold_lead="Quản lý phụ thuộc: ")
bullet("một pipeline có bước kiểm tra chất lượng mô hình giúp tránh triển khai nhầm "
       "checkpoint.", bold_lead="Chất lượng đi cùng tốc độ: ")

h2("5.3. Hướng phát triển")
bullet("thêm worker chạy GPU hoặc dùng ONNX Runtime để giảm thời gian suy luận.",
       bold_lead="Tăng tốc suy luận: ")
bullet("đưa các job vào hàng đợi (Redis/RabbitMQ) để xử lý request dài mà không chặn "
       "tiến trình web.", bold_lead="Hàng đợi tác vụ: ")
bullet("bổ sung Prometheus, Grafana, Loki và cảnh báo qua Telegram.",
       bold_lead="Giám sát và log tập trung: ")
bullet("viết bộ kiểm thử cho các hàm tiền xử lý và API, chạy trong pipeline trước khi "
       "build image.", bold_lead="Kiểm thử tự động: ")
bullet("chuyển từ Compose sang Kubernetes khi số lượng dịch vụ và tải tăng.",
       bold_lead="Mở rộng hạ tầng: ")
bullet("bổ sung cơ chế bảo vệ tên riêng và từ hiếm để giảm dương tính giả khi phát hiện "
       "lỗi.", bold_lead="Cải thiện chất lượng: ")

h2("5.4. Kết luận")
p("Việc áp dụng DevOps cho một hệ thống học sâu không chỉ là đóng gói ứng dụng mà còn là "
  "thay đổi cách làm việc: mọi thay đổi đều có phiên bản, mọi lần triển khai đều lặp lại "
  "được và mọi sự cố đều có đường quay lui. Đây là nền tảng để tiếp tục mở rộng hệ thống "
  "sửa lỗi chính tả tiếng Việt thành một dịch vụ ổn định, có thể cập nhật mô hình thường "
  "xuyên mà không gián đoạn người dùng.")

# ================================================================== TÀI LIỆU
h1("TÀI LIỆU THAM KHẢO")
for i, ref in enumerate([
    "VSEC Corpus – Vietnamese Spelling Error Correction Corpus, "
    "https://github.com/VSEC2021/VSEC",
    "VietAI/vit5-base, Hugging Face Model Hub, https://huggingface.co/VietAI/vit5-base",
    "Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified "
    "Text-to-Text Transformer. Journal of Machine Learning Research, 21(140).",
    "Docker Documentation – Multi-stage builds, https://docs.docker.com/build/building/multi-stage/",
    "Nginx Documentation – ngx_http_proxy_module, https://nginx.org/en/docs/http/ngx_http_proxy_module.html",
    "GitHub Docs – GitHub Actions workflow syntax, "
    "https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions",
    "Forsgren, N., Humble, J., Kim, G. (2018). Accelerate: The Science of Lean Software "
    "and DevOps. IT Revolution Press.",
], 1):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.left_indent = Cm(1.0)
    par.paragraph_format.first_line_indent = Cm(-0.5)
    par.paragraph_format.space_after = Pt(4)
    par.paragraph_format.line_spacing = 1.3
    r = par.add_run(f"[{i}] ")
    set_font(r, size=12, bold=True)
    r = par.add_run(ref)
    set_font(r, size=12)

footer_page_numbers()
doc.save(OUT)
print("saved", OUT)
print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))
