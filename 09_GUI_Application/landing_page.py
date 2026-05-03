#!/usr/bin/env python3
"""
Landing Page for Pediatric Appendicitis Prediction System
Matches the design of prediction_gui.py using ttk themed interface
"""

import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

class LandingPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Pediatric Appendicitis Prediction System")
        self.root.geometry("900x600")
        self.root.configure(bg='white')
        
        # Center the window
        self.center_window()
        
        # Create UI components
        self.create_header()
        self.create_description()
        self.create_start_button()
        self.create_footer()
    
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_header(self):
        """Create the header section with title matching GUI design"""
        title_frame = ttk.Frame(self.root, padding="30")
        title_frame.pack(fill=tk.X)
        
        # Main title - matching GUI font and style
        title_label = ttk.Label(
            title_frame,
            text="Pediatric Appendicitis Prediction System",
            font=('Arial', 28, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle - matching GUI style
        subtitle_label = ttk.Label(
            title_frame,
            text="Clinical Decision Support Tool",
            font=('Arial', 14, 'italic')
        )
        subtitle_label.pack()
    
    def create_description(self):
        """Create the system description section"""
        desc_frame = ttk.LabelFrame(self.root, text="SYSTEM OVERVIEW", padding="20")
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Main description
        main_desc = ttk.Label(
            desc_frame,
            text=(
                "This AI-powered system provides real-time predictions for pediatric appendicitis "
                "using multiple advanced machine learning models. Designed to assist healthcare "
                "professionals in making informed clinical decisions."
            ),
            font=('Arial', 12),
            wraplength=800,
            justify=tk.LEFT
        )
        main_desc.pack(pady=(0, 20))
        
        # Key features section
        features_frame = ttk.LabelFrame(desc_frame, text="KEY FEATURES", padding="15")
        features_frame.pack(fill=tk.X, pady=10)
        
        features_text = ttk.Label(
            features_frame,
            text=(
                "• Multiple AI Models: Decision Tree, Gradient Boosting, XGBoost, and Transformer\n"
                "• Real-time Predictions with Confidence Scores\n"
                "• Support for Laboratory Results and Clinical Examination Data\n"
                "• High Accuracy: XGBoost model achieves 85.3% accuracy\n"
                "• Medical Safety Focus: High Sensitivity for Critical Cases\n"
                "• User-Friendly Interface Designed for Healthcare Professionals"
            ),
            font=('Arial', 11),
            wraplength=750,
            justify=tk.LEFT
        )
        features_text.pack()
        
        # Technical specifications
        tech_frame = ttk.LabelFrame(desc_frame, text="TECHNICAL SPECIFICATIONS", padding="15")
        tech_frame.pack(fill=tk.X, pady=10)
        
        tech_text = ttk.Label(
            tech_frame,
            text=(
                "• Input Features: 30 clinical and laboratory parameters\n"
                "• Processing Time: < 1 second per prediction\n"
                "• Model Validation: Cross-validated on pediatric patient datasets\n"
                "• Output: Diagnosis prediction with probability scores"
            ),
            font=('Arial', 11),
            wraplength=750,
            justify=tk.LEFT
        )
        tech_text.pack()
    
    def create_start_button(self):
        """Create the start button section"""
        button_frame = ttk.Frame(self.root, padding="20")
        button_frame.pack(fill=tk.X)
        
        # Start button - matching GUI styling
        start_button = ttk.Button(
            button_frame,
            text="START SYSTEM",
            command=self.start_main_system,
            style='Accent.TButton'
        )
        start_button.pack(pady=10)
        
        # Configure button style for emphasis
        style = ttk.Style()
        style.configure(
            'Accent.TButton',
            font=('Arial', 16, 'bold'),
            padding=(20, 10)
        )
        
        # Instructions
        instruction_label = ttk.Label(
            button_frame,
            text="Click 'START SYSTEM' to begin using the prediction tool",
            font=('Arial', 10, 'italic'),
            foreground='gray'
        )
        instruction_label.pack(pady=(5, 0))
    
    def create_footer(self):
        """Create the footer section"""
        footer_frame = ttk.Frame(self.root, padding="15")
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        footer_text = ttk.Label(
            footer_frame,
            text="Version 1.0 | Developed for Medical Research Purposes | "
                 "© 2026 Pediatric Appendicitis Prediction System",
            font=('Arial', 9),
            foreground='gray'
        )
        footer_text.pack()
        
        disclaimer_label = ttk.Label(
            footer_frame,
            text="This system is intended for research and educational purposes only. "
                 "Always consult with qualified healthcare professionals for medical decisions.",
            font=('Arial', 8),
            foreground='darkgray',
            wraplength=800
        )
        disclaimer_label.pack(pady=(5, 0))
    
    def start_main_system(self):
        """Launch the main GUI system"""
        try:
            # Get the directory of the landing page
            current_dir = os.path.dirname(os.path.abspath(__file__))
            gui_path = os.path.join(current_dir, 'prediction_gui.py')
            
            # Launch the main GUI
            subprocess.Popen([sys.executable, gui_path])
            
            # Close the landing page
            self.root.destroy()
            
        except Exception as e:
            # Show error message
            error_msg = f"Error launching system: {str(e)}"
            tk.messagebox.showerror("Error", error_msg)

def main():
    """Main function to run the landing page"""
    root = tk.Tk()
    app = LandingPage(root)
    root.mainloop()

if __name__ == "__main__":
    main()
