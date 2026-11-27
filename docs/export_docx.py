# -*- coding: utf-8 -*-
"""Xuất báo cáo sang PDF bằng Word COM (có thử lại) rồi render từng trang."""
import os
import time
import win32com.client
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX = os.path.join(os.path.dirname(HERE), "BaoCao_DevOps_VietnameseSpellChecker_final.docx")
PDF = os.path.join(HERE, "report_preview.pdf")
OUT = os.path.join(HERE, "renders")
os.makedirs(OUT, exist_ok=True)

if os.path.exists(PDF):
    os.remove(PDF)

app = win32com.client.DispatchEx("Word.Application")
try:
    app.Visible = False
except Exception:
    pass
word = app

doc = None
for attempt in range(3):
    try:
        doc = word.Documents.Open(DOCX)
        print("opened, attempt", attempt + 1, flush=True)
        break
    except Exception as exc:
        print("open failed:", exc, flush=True)
        time.sleep(3)

if doc is None:
    raise SystemExit("cannot open document")

try:
    for toc in doc.TablesOfContents:
        toc.Update()
    print("toc updated", flush=True)
except Exception as exc:
    print("toc warning:", exc, flush=True)

for attempt in range(3):
    try:
        doc.SaveAs(PDF, 17)
        print("saved pdf, attempt", attempt + 1, flush=True)
        break
    except Exception as exc:
        print("save failed:", exc, flush=True)
        time.sleep(3)

try:
    doc.Close(0)
except Exception as exc:
    print("close warning:", exc, flush=True)
try:
    word.Quit()
except Exception as exc:
    print("quit warning:", exc, flush=True)

if not os.path.exists(PDF):
    raise SystemExit("pdf not produced")

pdf = pymupdf.open(PDF)
print("pages:", pdf.page_count, flush=True)
for i, page in enumerate(pdf, 1):
    pix = page.get_pixmap(dpi=100)
    pix.save(os.path.join(OUT, f"doc_{i:02d}.png"))
print("DONE", pdf.page_count)
