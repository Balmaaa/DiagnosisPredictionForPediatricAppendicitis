import tkinter as tk
from tkinter import ttk

def create_label(parent, text, row, column, padx=5, pady=5, sticky="w", font=("Segoe UI", 10), bg="#E3F2FD"):
    label = tk.Label(parent, text=text, font=font, bg=bg)
    label.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
    return label


def create_entry(parent, variable, row, column, width=15, padx=5, pady=5):
    entry = tk.Entry(parent, textvariable=variable, width=width)
    entry.grid(row=row, column=column, padx=padx, pady=pady)
    return entry


def create_combobox(parent, variable, values, row, column, width=15, state="readonly", padx=5, pady=5):
    combo = ttk.Combobox(parent, textvariable=variable, values=values, width=width, state=state)
    combo.grid(row=row, column=column, padx=padx, pady=pady)
    return combo


def create_section_frame(parent, title, font=("Segoe UI", 12, "bold"), bg="#E3F2FD", padx=15, pady=15):
    frame = tk.LabelFrame(parent, text=title, font=font, bg=bg, padx=padx, pady=pady)
    frame.pack(fill="x", pady=5, padx=10)
    return frame


def create_title(parent, text, font=("Segoe UI", 20, "bold"), bg="#F5F7FA", fg="#1565C0"):
    label = tk.Label(parent, text=text, font=font, bg=bg, fg=fg)
    label.pack(pady=(10, 20))
    return label


def create_button(parent, text, command, bg, fg="white", padx=20, pady=10, font=("Segoe UI", 11, "bold")):
    button = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, padx=padx, pady=pady, font=font)
    return button


def create_result_label(parent, variable, row, column, font=("Segoe UI", 11, "bold"), bg="#E3F2FD", fg="#1565C0", padx=15, pady=5):
    label = tk.Label(parent, textvariable=variable, font=font, bg=bg, fg=fg)
    label.grid(row=row, column=column, padx=padx, pady=pady, sticky="w")
    return label