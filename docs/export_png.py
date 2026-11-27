# -*- coding: utf-8 -*-
"""Xuất slide sang PNG bằng PowerPoint COM để kiểm tra trực quan."""
import os
import sys
import win32com.client

HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(os.path.dirname(HERE), "Slide_DevOps_VietnameseSpellChecker.pptx")
OUT = os.path.join(HERE, "renders")
os.makedirs(OUT, exist_ok=True)

app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(PPTX, WithWindow=False)
print("slides:", pres.Slides.Count, flush=True)
for i in range(1, pres.Slides.Count + 1):
    path = os.path.join(OUT, f"slide_{i:02d}.png")
    pres.Slides(i).Export(path, "PNG", 1600, 900)
    print("exported", path, flush=True)
pres.Close()
app.Quit()
print("DONE")
