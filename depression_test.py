import tkinter as tk
import ttkbootstrap as ttk
from tkinter import scrolledtext, filedialog, messagebox
import tensorflow as tf
import numpy as np
import json
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Load Models and Preprocessing Info ---
try:
    print("--- Loading trained models and preprocessing info... ---")
    cnn_model = tf.keras.models.load_model('cnn_model.keras')
    bilstm_model = tf.keras.models.load_model('bilstm_model.keras')

    with open('tokenizer.json') as f:
        tokenizer_json = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    with open('config.json') as f:
        config = json.load(f)
        MAX_LEN = config['max_len']
    print("--- Models and info loaded successfully. Ready to launch GUI. ---")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# --- 2. Helper and Prediction Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', '', text)
    return text

def predict_message(message):
    cleaned_message = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned_message])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=MAX_LEN)
    
    cnn_pred = cnn_model.predict(padded_sequence, verbose=0)[0][0]
    bilstm_pred = bilstm_model.predict(padded_sequence, verbose=0)[0][0]
    
    return cnn_pred, bilstm_pred

# --- 3. GUI Application Logic ---
def get_risk_style(score):
    if score < 0.4: return "Low Risk", "success"
    elif score < 0.7: return "Moderate Risk", "warning"
    else: return "High Risk", "danger"

def handle_prediction():
    user_text = text_area.get("1.0", "end").strip()
    if not user_text:
        status_var.set("Status: Please enter some text.")
        return

    status_var.set("Status: Predicting...")
    root.update_idletasks()

    cnn_p, bilstm_p = predict_message(user_text)
    update_results(cnn_p, bilstm_p)
    status_var.set("Status: Ready")

def update_results(cnn_p, bilstm_p):
    cnn_risk_text, cnn_style = get_risk_style(cnn_p)
    cnn_result_var.set(f"CNN Prediction: {cnn_p*100:.2f}% ({cnn_risk_text})")
    cnn_result_label.configure(bootstyle=f"{cnn_style}")
    cnn_progress_bar.configure(bootstyle=f"{cnn_style}", value=cnn_p * 100)
    
    bilstm_risk_text, bilstm_style = get_risk_style(bilstm_p)
    bilstm_result_var.set(f"Bi-LSTM Prediction: {bilstm_p*100:.2f}% ({bilstm_risk_text})")
    bilstm_result_label.configure(bootstyle=f"{bilstm_style}")
    bilstm_progress_bar.configure(bootstyle=f"{bilstm_style}", value=bilstm_p * 100)

def clear_fields():
    text_area.delete("1.0", "end")
    cnn_result_var.set("CNN Prediction: -")
    cnn_result_label.configure(bootstyle="secondary")
    cnn_progress_bar.configure(bootstyle="secondary", value=0)
    bilstm_result_var.set("Bi-LSTM Prediction: -")
    bilstm_result_label.configure(bootstyle="secondary")
    bilstm_progress_bar.configure(bootstyle="secondary", value=0)
    status_var.set("Status: Ready")

# --- 4. Menu Bar Functions ---
def open_text_file():
    filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All files", "*.*")])
    if not filepath:
        return
    clear_fields()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_area.insert("1.0", f.read())
    status_var.set(f"Loaded: {os.path.basename(filepath)}")

def save_results():
    filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All files", "*.*")])
    if not filepath:
        return
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("--- Analyzed Text ---\n")
        f.write(text_area.get("1.0", "end").strip())
        f.write("\n\n--- Prediction Results ---\n")
        f.write(cnn_result_var.get() + "\n")
        f.write(bilstm_result_var.get() + "\n")
    status_var.set(f"Results saved to: {os.path.basename(filepath)}")

def show_about():
    messagebox.showinfo("About Depression Risk Detector", "Machine Learning (I)) Final Project\n\nThis application uses AI models (CNN and Bi-LSTM) to analyze text and predict the likelihood of it being related to depression.\n\nGroup Members:\n1. Alfa Radithya Fanany - 5025231008\n2. Muhammad Iqbal Shafarel - 5025231080\n3. Faiz Adli Nugraha - 5025231174")

# --- 5. Create the GUI Window and Widgets ---
root = ttk.Window(themename="vapor")
root.title("Depression Risk Detector")
root.geometry("600x525")

# --- Create Menu Bar ---
menubar = ttk.Menu(root)
root.config(menu=menubar)

# File Menu
file_menu = ttk.Menu(menubar, tearoff=False)
menubar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open Text File...", command=open_text_file)
file_menu.add_command(label="Save Results...", command=save_results)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Themes Menu
style = ttk.Style()
theme_menu = ttk.Menu(menubar, tearoff=False)
menubar.add_cascade(label="Themes", menu=theme_menu)
for theme in sorted(style.theme_names()):
    theme_menu.add_radiobutton(label=theme, command=lambda t=theme: style.theme_use(t))

# Help Menu
help_menu = ttk.Menu(menubar, tearoff=False)
menubar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About...", command=show_about)

main_frame = ttk.Frame(root, padding="15")
main_frame.pack(fill="both", expand=True)
main_frame.columnconfigure(0, weight=1) 
input_frame = ttk.LabelFrame(main_frame, text="Enter Text to Analyze", padding="10")
input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
input_frame.columnconfigure(0, weight=1)
text_area = scrolledtext.ScrolledText(input_frame, wrap="word", height=10, font=("Segoe UI", 10))
text_area.grid(row=0, column=0, sticky="nsew")

button_frame = ttk.Frame(main_frame)
button_frame.grid(row=1, column=0, columnspan=2, pady=5)
predict_button = ttk.Button(button_frame, text="Predict", command=handle_prediction, bootstyle="primary", width=15)
predict_button.pack(side="left", padx=5)
clear_button = ttk.Button(button_frame, text="Clear", command=clear_fields, bootstyle="secondary", width=15)
clear_button.pack(side="left", padx=5)

results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
results_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
results_frame.columnconfigure(0, weight=1)

cnn_result_var = tk.StringVar(value="CNN Prediction: -")
cnn_result_label = ttk.Label(results_frame, textvariable=cnn_result_var, font=("Segoe UI", 12, "bold"), bootstyle="secondary")
cnn_result_label.grid(row=0, column=0, sticky="w", pady=(5,0))
cnn_progress_bar = ttk.Progressbar(results_frame, bootstyle="secondary-striped", length=300)
cnn_progress_bar.grid(row=1, column=0, sticky="ew", pady=(5, 10), padx=5)

bilstm_result_var = tk.StringVar(value="Bi-LSTM Prediction: -")
bilstm_result_label = ttk.Label(results_frame, textvariable=bilstm_result_var, font=("Segoe UI", 12, "bold"), bootstyle="secondary")
bilstm_result_label.grid(row=2, column=0, sticky="w", pady=5)
bilstm_progress_bar = ttk.Progressbar(results_frame, bootstyle="secondary-striped", length=300)
bilstm_progress_bar.grid(row=3, column=0, sticky="ew", pady=(5, 10), padx=5)

status_var = tk.StringVar(value="Status: Ready")
status_bar = ttk.Label(main_frame, textvariable=status_var, relief="sunken", anchor="w", padding=5)
status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(15, 0))

root.mainloop()
