"""
corpus_adder.py
───────────────
A simple Tkinter app to clean Arabic text and append it
as a single line to corpus.txt.

- Removes everything except Arabic letters and spaces
- Collapses all newlines into a single sentence
- Appends the cleaned sentence as a new line in corpus.txt
"""

import tkinter as tk
from tkinter import messagebox
import re
import os

CORPUS_FILE = "tools/add Text from paste to file using GUI/corpus.txt"


def clean_arabic(text: str) -> str:
    """
    1. Collapse all newlines/tabs into spaces
    2. Keep only Arabic Unicode letters and spaces
    3. Collapse multiple spaces into one
    4. Strip leading/trailing whitespace
    """
    # Replace newlines and tabs with space
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Keep only Arabic letters (U+0600–U+06FF) and spaces
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def add_to_corpus():
    raw = input_box.get("1.0", tk.END)
    cleaned = clean_arabic(raw)

    if not cleaned:
        messagebox.showwarning("نص فارغ", "لا يوجد نص عربي صالح للإضافة.")
        return

    # Append to corpus.txt as a new line
    with open(CORPUS_FILE, "a", encoding="utf-8") as f:
        f.write(cleaned + "\n")

    # Show preview of what was added
    preview_var.set(
        f"✓  أُضيف: {cleaned[:80]}{'...' if len(cleaned) > 80 else ''}")

    # Update line count
    update_count()

    # Clear input box
    input_box.delete("1.0", tk.END)
    input_box.focus()


def update_count():
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            lines = [l for l in f.readlines() if l.strip()]
        count_var.set(f"corpus.txt  ←  {len(lines)} جملة")
    else:
        count_var.set("corpus.txt  ←  0 جملة")


def clear_input():
    input_box.delete("1.0", tk.END)
    input_box.focus()


# ── Window setup ─────────────────────────────────────────────────
root = tk.Tk()
root.title("إضافة نص عربي إلى corpus.txt")
root.resizable(False, False)
root.configure(bg="#1e1e2e")

# ── Fonts & colors ────────────────────────────────────────────────
BG = "#1e1e2e"
CARD = "#2a2a3d"
ACCENT = "#7c5cbf"
ACCENT_H = "#9b7de0"
TEXT = "#e0e0f0"
SUBTEXT = "#888aaa"
GREEN = "#50fa7b"
RED = "#ff5555"

FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_LABEL = ("Segoe UI", 10)
FONT_BTN = ("Segoe UI", 11, "bold")
FONT_INPUT = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 9)

PAD = 18

# ── Title ─────────────────────────────────────────────────────────
tk.Label(
    root, text="📝  محرر الكوربس العربي",
    font=FONT_TITLE, bg=BG, fg=TEXT
).grid(row=0, column=0, columnspan=2, pady=(PAD, 6), padx=PAD)

tk.Label(
    root, text="أدخل الجملة أو الفقرة — سيتم تنظيفها وإضافتها كسطر واحد",
    font=FONT_LABEL, bg=BG, fg=SUBTEXT
).grid(row=1, column=0, columnspan=2, pady=(0, 10), padx=PAD)

# ── Input box ─────────────────────────────────────────────────────
frame = tk.Frame(root, bg=CARD, bd=0, highlightthickness=2,
                 highlightbackground=ACCENT)
frame.grid(row=2, column=0, columnspan=2, padx=PAD, sticky="ew")

input_box = tk.Text(
    frame,
    width=52, height=7,
    font=FONT_INPUT,
    bg=CARD, fg=TEXT,
    insertbackground=TEXT,
    relief="flat",
    padx=10, pady=8,
    wrap="word",
)
input_box.pack(fill="both")

# ── Buttons ───────────────────────────────────────────────────────
btn_frame = tk.Frame(root, bg=BG)
btn_frame.grid(row=3, column=0, columnspan=2, pady=12, padx=PAD, sticky="ew")
btn_frame.columnconfigure(0, weight=1)
btn_frame.columnconfigure(1, weight=1)

add_btn = tk.Button(
    btn_frame, text="➕  إضافة إلى corpus.txt",
    font=FONT_BTN, bg=ACCENT, fg="white",
    activebackground=ACCENT_H, activeforeground="white",
    relief="flat", cursor="hand2", padx=10, pady=8,
    command=add_to_corpus,
)
add_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

clear_btn = tk.Button(
    btn_frame, text="🗑  مسح",
    font=FONT_BTN, bg="#3a3a50", fg=SUBTEXT,
    activebackground="#4a4a60", activeforeground=TEXT,
    relief="flat", cursor="hand2", padx=10, pady=8,
    command=clear_input,
)
clear_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

# ── Bind Enter (Ctrl+Enter to add) ───────────────────────────────
root.bind("<Control-Return>", lambda e: add_to_corpus())

# ── Preview label ─────────────────────────────────────────────────
preview_var = tk.StringVar(value="")
tk.Label(
    root, textvariable=preview_var,
    font=FONT_SMALL, bg=BG, fg=GREEN,
    wraplength=420, justify="right"
).grid(row=4, column=0, columnspan=2, padx=PAD, pady=(0, 6))

# ── Divider ───────────────────────────────────────────────────────
tk.Frame(root, bg=CARD, height=1).grid(
    row=5, column=0, columnspan=2, sticky="ew", padx=PAD, pady=4
)

# ── Counter ───────────────────────────────────────────────────────
count_var = tk.StringVar()
update_count()
tk.Label(
    root, textvariable=count_var,
    font=FONT_SMALL, bg=BG, fg=SUBTEXT
).grid(row=6, column=0, columnspan=2, pady=(4, PAD))

# ── Keyboard shortcut hint ────────────────────────────────────────
tk.Label(
    root, text="Ctrl+Enter للإضافة السريعة",
    font=("Segoe UI", 8), bg=BG, fg="#555570"
).grid(row=7, column=0, columnspan=2, pady=(0, 8))

input_box.focus()
root.mainloop()
