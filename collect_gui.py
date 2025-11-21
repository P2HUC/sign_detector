import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import threading
import queue
import time
from pathlib import Path

class SignLanguageCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Collector")
        
        # Data directory
        self.DATA_DIR = Path('./data')
        self.DATA_DIR.mkdir(exist_ok=True)
        
        # Camera setup
        self.cap = self.open_first_available_camera()
        if self.cap is None:
            messagebox.showerror("Error", "No available camera found!")
            self.root.quit()
            return
        
        # UI Setup
        self.setup_ui()
        
        # Variables
        self.collecting = False
        self.current_sign = None
        self.counter = 0
        self.images_to_capture = 100
        self.queue = queue.Queue()
        
        # Start the video capture in a separate thread
        self.stop_event = threading.Event()
        self.video_thread = threading.Thread(target=self.update_video, daemon=True)
        self.video_thread.start()
        
        # Start processing the queue
        self.process_queue()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_frame = ttk.Label(main_frame)
        self.video_frame.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Sign selection
        ttk.Label(controls_frame, text="Select Sign:").grid(row=0, column=0, padx=5, pady=5)
        self.sign_var = tk.StringVar()
        self.sign_combobox = ttk.Combobox(controls_frame, textvariable=self.sign_var, state="readonly")
        self.sign_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.sign_combobox.bind('<<ComboboxSelected>>', self.on_sign_selected)
        
        # New sign button
        ttk.Button(controls_frame, text="New Sign", command=self.add_new_sign).grid(row=0, column=2, padx=5, pady=5)
        
        # Capture controls
        self.capture_btn = ttk.Button(controls_frame, text="Start Capture", command=self.toggle_capture, state=tk.DISABLED)
        self.capture_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Select or create a sign to begin")
        ttk.Label(controls_frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=3)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(controls_frame, variable=self.progress_var, maximum=100)
        self.progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Update sign list
        self.update_sign_list()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
    def open_first_available_camera(self, preferred_indices=(0, 1, 2, 3)):
        for idx in preferred_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()
        return None
    
    def update_sign_list(self):
        """Update the list of available signs in the combobox"""
        signs = [d.name for d in self.DATA_DIR.iterdir() if d.is_dir()]
        self.sign_combobox['values'] = sorted(signs)
        if signs:
            self.sign_combobox.set(signs[0])
            self.current_sign = signs[0]
            self.capture_btn.config(state=tk.NORMAL)
    
    def add_new_sign(self):
        """Add a new sign to the collection"""
        sign_name = simpledialog.askstring("New Sign", "Enter the name of the new sign:")
        if sign_name and sign_name.strip():
            sign_dir = self.DATA_DIR / sign_name.strip()
            sign_dir.mkdir(exist_ok=True)
            self.update_sign_list()
            self.sign_combobox.set(sign_name)
            self.current_sign = sign_name
            self.capture_btn.config(state=tk.NORMAL)
    
    def on_sign_selected(self, event=None):
        """Handle sign selection from the combobox"""
        self.current_sign = self.sign_var.get()
        self.capture_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Selected sign: {self.current_sign}")
    
    def toggle_capture(self):
        """Toggle image capture on/off"""
        if not self.collecting:
            self.start_collection()
        else:
            self.stop_collection()
    
    def start_collection(self):
        """Start collecting images for the selected sign"""
        if not self.current_sign:
            messagebox.showerror("Error", "Please select or create a sign first!")
            return
        
        self.collecting = True
        # Ensure the sign directory exists before starting
        sign_dir = self.DATA_DIR / self.current_sign
        sign_dir.mkdir(parents=True, exist_ok=True)
        self.counter = len(list(sign_dir.glob('*.jpg')))
        self.capture_btn.config(text="Stop Capture")
        self.sign_combobox.config(state=tk.DISABLED)
        self.status_var.set(f"Capturing images for '{self.current_sign}' - {self.counter} collected")
        self.progress_var.set((self.counter / self.images_to_capture) * 100)
    
    def stop_collection(self):
        """Stop collecting images"""
        self.collecting = False
        self.capture_btn.config(text="Start Capture")
        self.sign_combobox.config(state='readonly')
        self.status_var.set(f"Stopped capture. Total collected for '{self.current_sign}': {self.counter}")
    
    def update_video(self):
        """Update the video feed in a separate thread"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resize frame to fit in the UI
                frame = cv2.resize(frame, (640, 480))
                
                # Add text overlay if collecting
                if self.collecting:
                    cv2.putText(
                        frame, 
                        f"Collecting: {self.current_sign} - {self.counter}",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (0, 255, 0), 
                        2
                    )
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame to queue for the main thread to process
                if self.queue.qsize() < 2:  # Keep queue size small to prevent lag
                    self.queue.put(frame_rgb)
            
            time.sleep(0.03)  # Cap at ~30 FPS
    
    def process_queue(self):
        """Process the queue in the main thread to update the UI"""
        try:
            if not self.queue.empty():
                frame_rgb = self.queue.get_nowait()
                
                # Save image if collecting
                if self.collecting and self.current_sign:
                    sign_dir = self.DATA_DIR / self.current_sign
                    sign_dir.mkdir(parents=True, exist_ok=True)
                    # Use counter and timestamp to avoid accidental overwrite
                    timestamp = int(time.time() * 1000)
                    save_name = f"{self.counter}_{timestamp}.jpg"
                    save_path = sign_dir / save_name
                    # Use Pillow to save the image (works with Unicode paths on Windows)
                    try:
                        img_pil = Image.fromarray(frame_rgb)
                        img_pil.save(str(save_path), format='JPEG')
                        print(f"Saved image to {save_path}")
                        saved = True
                    except Exception as e:
                        print(f"Failed to save image to {save_path}: {e}")
                        self.status_var.set(f"Failed to save image to {save_path}")
                        saved = False
                    self.counter += 1
                    self.status_var.set(f"Capturing images for '{self.current_sign}' - {self.counter} collected")
                    self.progress_var.set((self.counter / self.images_to_capture) * 100)
                    
                    if self.counter >= self.images_to_capture:
                        self.stop_collection()
                
                # Update the display
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
                
        except queue.Empty:
            pass
        
        # Schedule the next update
        self.root.after(10, self.process_queue)
    
    def on_closing(self):
        """Handle window close event"""
        self.stop_event.set()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageCollector(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center the window
    window_width = 800
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    root.mainloop()
