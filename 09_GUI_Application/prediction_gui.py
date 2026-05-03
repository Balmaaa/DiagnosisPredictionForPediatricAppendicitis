import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import joblib
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureEmbedding(nn.Module):
    def __init__(self, feature_info, embed_dim=64):
        super().__init__()
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        numerical_features = [f for f, i in feature_info.items() if i['type'] == 'numerical']
        self.numerical_embeddings = nn.ModuleDict({f: nn.Linear(1, embed_dim) for f in numerical_features})
        self.categorical_embeddings = nn.ModuleDict({
            f: nn.Embedding(i['unique_values'], embed_dim)
            for f, i in feature_info.items() if i['type'] == 'categorical'
        })
        self.numerical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.categorical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.feature_positional_encoding = nn.Parameter(torch.randn(len(feature_info), embed_dim) * 0.1)

    def forward(self, x_dict):
        embeddings = []
        if 'numerical' in x_dict:
            data = x_dict['numerical']
            for i, feat in enumerate(self.numerical_embeddings.keys()):
                if i < data.shape[1]:
                    e = self.numerical_embeddings[feat](data[:, i:i+1]).unsqueeze(1)
                    embeddings.append(e + self.numerical_token.unsqueeze(0))
        for feat, val in x_dict.items():
            if feat != 'numerical' and feat in self.categorical_embeddings:
                e = self.categorical_embeddings[feat](val).unsqueeze(1)
                embeddings.append(e + self.categorical_token.unsqueeze(0))
        if embeddings:
            x = torch.cat(embeddings, dim=1)
            x = x + self.feature_positional_encoding[:x.size(1)].unsqueeze(0)
        else:
            x = torch.zeros(x_dict['numerical'].size(0), 1, self.embed_dim)
        return x

class AdvancedTabularTransformer(nn.Module):
    def __init__(self, feature_info, embed_dim=64, num_heads=8, num_layers=4, num_classes=2, dropout=0.3):
        super().__init__()
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_embedding = FeatureEmbedding(feature_info, embed_dim)
        self.num_heads = min(num_heads, embed_dim // 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=self.num_heads, dropout=dropout,
            dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, embed_dim // 4), nn.GELU(), nn.Dropout(dropout * 0.25),
            nn.Linear(embed_dim // 4, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x_dict):
        x = self.feature_embedding(x_dict)
        x = self.transformer(x)
        x_pooled = torch.mean(x, dim=1)
        return self.classifier(x_pooled)


NUMERICAL_FEATURES = [
    'Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 'WBC_Count',
    'RBC_Count', 'Hemoglobin', 'RDW', 'Segmented_Neutrophils',
    'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage']

CATEGORICAL_FEATURES = [
    'Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite',
    'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 'Severity',
    'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness',
    'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine']

ALL_30_FEATURES = [
    'Age', 'Weight', 'Height', 'BMI', 'Sex', 'Body_Temperature',
    'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 'Nausea',
    'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 'Severity',
    'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness', 'Psoas_Sign',
    'Neutrophilia', 'Neutrophil_Percentage', 'WBC_Count', 'RBC_Count', 'Hemoglobin',
    'RDW', 'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP',
    'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine']

LAB_FIELDS = ['WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW',
              'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage']

LAB_MISSING_INDICATORS = [f"{lab}_missing" for lab in LAB_FIELDS]
ALL_FEATURES = ALL_30_FEATURES + LAB_MISSING_INDICATORS

CLINICAL_DEFAULTS = {
    'Age': 10, 'Weight': 35, 'Height': 140, 'BMI': 18,
    'Body_Temperature': 36.5, 'Neutrophil_Percentage': 65.0,
    'WBC_Count': 8.5, 'RBC_Count': 4.5, 'Hemoglobin': 12.5,
    'RDW': 14.0, 'Segmented_Neutrophils': 55.0,
    'Thrombocyte_Count': 250.0, 'CRP': 10.0}

ZERO_MEANS_MISSING = {'WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW',
                      'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage', 'Segmented_Neutrophils'}


class PredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pediatric Appendicitis Prediction System")
        self.root.geometry("2000x900")
        self.models = {}
        self.metadata = {}
        self.transformer_info = {}
        self.available_models = []
        self.load_models()
        self.create_widgets()

    def load_models(self):
        model_dir = Path(__file__).parent / 'saved_models'
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}"); return

        metadata_path = model_dir / 'metadata.pkl'
        if metadata_path.exists():
            try: self.metadata = joblib.load(metadata_path)
            except Exception as e: print(f"Warning: metadata load failed: {e}")

        pkl_models = {'Decision Tree': 'Decision_Tree.pkl',
                       'Gradient Boosting': 'Gradient_Boosting.pkl',
                       'XGBoost': 'XGBoost.pkl'}
        for name, fname in pkl_models.items():
            path = model_dir / fname
            if not path.exists(): continue
            try:
                obj = joblib.load(path)
                if isinstance(obj, dict):
                    model = None
                    for k in ['model', 'best_estimator_', 'trained_model']:
                        if k in obj: model = obj[k]; break
                    if model is None: continue
                    self.models[name] = model
                else:
                    self.models[name] = obj
                print(f"Loaded {name}")
            except Exception as e: print(f"Warning: {name} load failed: {e}")

        pt_path = model_dir / 'Transformer.pt'
        if pt_path.exists():
            try:
                ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
                model = AdvancedTabularTransformer(
                    feature_info=ckpt['feature_info'], embed_dim=ckpt['embed_dim'],
                    num_heads=ckpt['num_heads'], num_layers=ckpt['num_layers'],
                    dropout=ckpt['dropout'], num_classes=2)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                self.transformer_info = {
                    'model': model,
                    'normalization_means': ckpt['normalization_means'],
                    'normalization_stds': ckpt['normalization_stds'],
                    'feature_info': ckpt['feature_info'],
                    'temperature': ckpt.get('temperature', 1.0)}
                self.models['Transformer'] = 'transformer'
                print(f"Transformer loaded (temperature={self.transformer_info['temperature']:.4f})")
            except Exception as e: print(f"Warning: Transformer load failed: {e}")

        self.available_models = list(self.models.keys())
        print(f"Available models: {self.available_models}")

    def preprocess_input(self, input_data):
        fv = {}
        available_labs = input_data.pop('_available_labs', [])
        
        # Process all features
        for feat in ALL_30_FEATURES:
            if feat in available_labs:
                # Lab feature is available - use the actual value
                val = input_data.get(feat)
                try: fv[feat] = float(val)
                except: fv[feat] = 0.0
            elif feat in LAB_FIELDS:
                if feat in available_labs:
                    fv[feat] = float(input_data.get(feat, 0.0))
                else:
                    # DO NOT inject fake signal
                    fv[feat] = CLINICAL_DEFAULTS.get(feat, 0.0)
            else:
                # Non-lab feature - use the value or default
                val = input_data.get(feat)
                if val is None or val == '':
                    fv[feat] = float(CLINICAL_DEFAULTS.get(feat, 0))
                else:
                    try: fv[feat] = float(val)
                    except: fv[feat] = 0.0

        missing = {}
        for lab in LAB_FIELDS:
            if lab in available_labs:
                # Lab is available - check if it's actually zero (missing) vs provided
                val = fv.get(lab, 0.0)
                if lab in ZERO_MEANS_MISSING and val == 0.0:
                    missing[f"{lab}_missing"] = 1
                else:
                    missing[f"{lab}_missing"] = 0
                fv[lab] = float(CLINICAL_DEFAULTS.get(lab, 0.0))
            else:
                # Lab is not available - mark as missing
                missing[f"{lab}_missing"] = 1
                fv[lab] = 0.0  # Use neutral zero instead of medical default

        vec = [fv[f] for f in ALL_30_FEATURES] + [missing.get(m, 0) for m in LAB_MISSING_INDICATORS]
        return np.array([vec], dtype=np.float64)

    def predict(self, model_name, features_array):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        if model_name == 'Transformer':
            # Transformer uses all 38 features (30 base + 8 missing indicators)
            return self._predict_transformer(features_array)
        else:
            # sklearn models (Decision Tree, Gradient Boosting, XGBoost) 
            # were trained with ONLY 30 features - slice to first 30
            if isinstance(features_array, np.ndarray):
                sklearn_features = features_array[:, :30]  # Take only first 30 features
            else:
                # Handle list or other input types
                sklearn_features = np.array(features_array)[:, :30]
            return self._predict_sklearn(model_name, sklearn_features)

    def _predict_sklearn(self, model_name, features_array):
        model = self.models[model_name]
        processed = features_array.copy()
        scaler = self.metadata.get('numerical_scalers')
        if scaler is not None and hasattr(scaler, 'mean_'):
            try:
                # For sklearn models, use 30-feature scaling
                # Create feature to index mapping for ALL_30_FEATURES
                feat_to_idx = {f: i for i, f in enumerate(ALL_30_FEATURES)}
                
                # Find numerical features within the 30 features
                numerical_indices = []
                for num_feat in NUMERICAL_FEATURES:
                    if num_feat in feat_to_idx and feat_to_idx[num_feat] < 30:
                        numerical_indices.append(feat_to_idx[num_feat])
                
                if numerical_indices:
                    # Extract and scale numerical features
                    num_block = processed[:, numerical_indices]
                    scaled_block = scaler.transform(num_block)
                    processed[:, numerical_indices] = scaled_block
            except: pass
        pred = model.predict(processed)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(processed)[0]
            prob_yes, prob_no = float(proba[1]), float(proba[0])
        else:
            prob_yes = prob_no = 0.5
        diag = "Appendicitis" if int(pred) == 1 else "No Appendicitis"
        return {'diagnosis': diag, 'confidence': max(prob_yes, prob_no),
                'prob_appendicitis': prob_yes, 'prob_no_appendicitis': prob_no}

    def _predict_transformer(self, features_array):
        info = self.transformer_info
        model = info['model']
        nm, ns = info['normalization_means'], info['normalization_stds']
        temperature = info.get('temperature', 1.0)
        feature_info = info['feature_info']

        # Build feature dict matching training format:
        # numerical features (scaled) + categorical features (as LongTensor)
        feat_to_idx = {f: i for i, f in enumerate(ALL_FEATURES)}

        # Extract and scale numerical features in NUMERICAL_FEATURES order
        num_indices = [feat_to_idx[f] for f in NUMERICAL_FEATURES if f in feat_to_idx]
        num_block = features_array[:, num_indices].astype(np.float32)
        # Apply StandardScaler normalization (same as training)
        for i in range(min(num_block.shape[1], len(nm))):
            num_block[:, i] = (num_block[:, i] - nm[i]) / (ns[i] + 1e-8)

        X_dict = {'numerical': torch.FloatTensor(num_block)}

        # Extract categorical features as LongTensor
        for cat_feat in CATEGORICAL_FEATURES:
            if cat_feat in feat_to_idx and cat_feat in feature_info:
                val = int(features_array[0, feat_to_idx[cat_feat]])
                # Clamp to valid range for embedding
                max_val = feature_info[cat_feat].get('unique_values', 2)
                val = max(0, min(val, max_val - 1))
                X_dict[cat_feat] = torch.LongTensor([val])

        with torch.no_grad():
            logit = model(X_dict).squeeze().item()
            prob = torch.sigmoid(torch.tensor(logit / temperature)).item()
        prob_yes = prob
        prob_no = 1.0 - prob
        pred = 1 if prob > 0.5 else 0
        diag = "Appendicitis" if pred == 1 else "No Appendicitis"
        return {'diagnosis': diag, 'confidence': max(prob_yes, prob_no),
                'prob_appendicitis': prob_yes, 'prob_no_appendicitis': prob_no}

    
    def create_widgets(self):
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Label(title_frame, text="Pediatric Appendicitis Prediction System",
                  font=('Arial', 20, 'bold')).pack()
        ttk.Label(title_frame, text="Clinical Decision Support Tool",
                  font=('Arial', 12, 'italic')).pack()

        model_frame = ttk.LabelFrame(self.root, text="Model Selection", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)
        ttk.Label(model_frame, text="Select Model:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar()
        cb = ttk.Combobox(model_frame, textvariable=self.model_var,
                          values=self.available_models, state="readonly", width=25, font=('Arial', 11))
        cb.pack(side=tk.LEFT, padx=10)
        if self.available_models: self.model_var.set(self.available_models[0])

        
        main = ttk.Frame(self.root)
        main.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        self.root.columnconfigure(0, weight=1); self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        main.columnconfigure(0, weight=1); main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)

        self._build_input_form(main)
        self._build_output_section(main)
        self._build_controls()

    def _build_input_form(self, parent):
        frame = ttk.LabelFrame(parent, text="PATIENT INPUT DATA", padding="15")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        canvas = tk.Canvas(frame, width=800)
        sb = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        self._build_demographics(inner)
        self._build_clinical(inner)
        self._build_laboratory(inner)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_demographics(self, parent):
        sf = ttk.LabelFrame(parent, text="SEGMENT 1: DEMOGRAPHIC INFORMATION", padding="12")
        sf.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        self.demo_vars = {}
        fields = [("Age (years):", "Age", 0, 0, "float"), ("Weight (kg):", "Weight", 0, 1, "float"),
                  ("Height (cm):", "Height", 1, 0, "float"), ("BMI:", "BMI", 1, 1, "float"),
                  ("Sex:", "Sex", 2, 0, "sex")]
        for lbl, name, r, c, ft in fields:
            ttk.Label(sf, text=lbl, font=('Arial', 11, 'bold')).grid(row=r, column=c*2, padx=12, pady=8, sticky=tk.W)
            if ft == "sex":
                v = tk.StringVar(value="Male")
                ttk.Combobox(sf, textvariable=v, values=["Male", "Female"], state="readonly", width=15).grid(row=r, column=c*2+1, padx=12, pady=8)
            else:
                v = tk.DoubleVar(value=0.0)
                ttk.Entry(sf, textvariable=v, width=16, font=('Arial', 10)).grid(row=r, column=c*2+1, padx=12, pady=8)
            self.demo_vars[name] = v

    def _build_clinical(self, parent):
        sf = ttk.LabelFrame(parent, text="SEGMENT 2: CLINICAL SYMPTOMS", padding="12")
        sf.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        self.clinical_vars = {}
        fields = [
            ("Body Temp (C):", "Body_Temperature", 0, 0, "float"),
            ("Lower Right Pain:", "Lower_Right_Abd_Pain", 0, 1, "yesno"),
            ("Migratory Pain:", "Migratory_Pain", 1, 0, "yesno"),
            ("Loss of Appetite:", "Loss_of_Appetite", 1, 1, "yesno"),
            ("Nausea:", "Nausea", 2, 0, "yesno"),
            ("Coughing Pain:", "Coughing_Pain", 2, 1, "yesno"),
            ("Dysuria:", "Dysuria", 3, 0, "yesno"),
            ("Stool Changes:", "Stool", 3, 1, "stool"),
            ("Peritonitis:", "Peritonitis", 4, 0, "yesno"),
            ("Severity:", "Severity", 4, 1, "severity"),
            ("Contralateral Rebound:", "Contralateral_Rebound_Tenderness", 5, 0, "yesno"),
            ("Ipsilateral Rebound:", "Ipsilateral_Rebound_Tenderness", 5, 1, "yesno"),
            ("Psoas Sign:", "Psoas_Sign", 6, 0, "yesno")]
        for lbl, name, r, c, ft in fields:
            ttk.Label(sf, text=lbl, font=('Arial', 11, 'bold')).grid(row=r, column=c*2, padx=12, pady=6, sticky=tk.W)
            if ft == "float":
                v = tk.DoubleVar(value=36.5)
                ttk.Entry(sf, textvariable=v, width=16, font=('Arial', 10)).grid(row=r, column=c*2+1, padx=12, pady=6)
            elif ft == "yesno":
                v = tk.StringVar(value="no")
                ttk.Combobox(sf, textvariable=v, values=["yes", "no"], state="readonly", width=15).grid(row=r, column=c*2+1, padx=12, pady=6)
            elif ft == "severity":
                v = tk.StringVar(value="uncomplicated")
                ttk.Combobox(sf, textvariable=v, values=["uncomplicated", "complicated"], state="readonly", width=15).grid(row=r, column=c*2+1, padx=12, pady=6)
            elif ft == "stool":
                v = tk.StringVar(value="normal")
                ttk.Combobox(sf, textvariable=v, values=["normal", "constipation", "diarrhea"], state="readonly", width=15).grid(row=r, column=c*2+1, padx=12, pady=6)
            self.clinical_vars[name] = v

    def _build_laboratory(self, parent):
        sf = ttk.LabelFrame(parent, text="SEGMENT 3: LABORATORY RESULTS", padding="12")
        sf.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        self.lab_vars = {}
        fields = [
            ("WBC Count (x10^9/L):", "WBC_Count", 0, 0, "float"),
            ("RBC Count (x10^12/L):", "RBC_Count", 0, 1, "float"),
            ("Hemoglobin (g/dL):", "Hemoglobin", 1, 0, "float"),
            ("RDW (%):", "RDW", 1, 1, "float"),
            ("Segmented Neutro (%):", "Segmented_Neutrophils", 2, 0, "float"),
            ("Thrombocyte (x10^9/L):", "Thrombocyte_Count", 2, 1, "float"),
            ("CRP (mg/L):", "CRP", 3, 0, "float"),
            ("Neutrophil %:", "Neutrophil_Percentage", 3, 1, "float"),
            ("Neutrophilia:", "Neutrophilia", 4, 0, "yesno"),
            ("Ketones in Urine:", "Ketones_in_Urine", 4, 1, "ketones"),
            ("RBC in Urine:", "RBC_in_Urine", 5, 0, "yesno"),
            ("WBC in Urine:", "WBC_in_Urine", 5, 1, "yesno")]
        for lbl, name, r, c, ft in fields:
            ttk.Label(sf, text=lbl, font=('Arial', 11, 'bold')).grid(row=r, column=c*2, padx=12, pady=6, sticky=tk.W)
            if ft == "float":
                v = tk.DoubleVar(value=0.0)
                ttk.Entry(sf, textvariable=v, width=16, font=('Arial', 10)).grid(row=r, column=c*2+1, padx=12, pady=6)
            elif ft == "yesno":
                v = tk.StringVar(value="no")
                ttk.Combobox(sf, textvariable=v, values=["yes", "no"], state="readonly", width=15).grid(row=r, column=c*2+1, padx=12, pady=6)
            elif ft == "ketones":
                v = tk.StringVar(value="no")
                ttk.Combobox(sf, textvariable=v, values=["no", "+", "++", "+++"], state="readonly", width=15).grid(row=r, column=c*2+1, padx=12, pady=6)
            self.lab_vars[name] = v

    def _build_output_section(self, parent):
        frame = ttk.LabelFrame(parent, text="PREDICTION RESULTS", padding="15")
        frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        self.results_text = scrolledtext.ScrolledText(frame, height=35, width=80,
                                                      font=('Courier', 10), wrap=tk.WORD, state='disabled')
        self.results_text.pack(fill=tk.BOTH, expand=True)
        msg = """PEDIATRIC APPENDICITIS PREDICTION SYSTEM
============================================

Clinical Decision Support Tool

INSTRUCTIONS:
1. Enter patient data on the left
2. Select a model from the dropdown
3. Click "Predict Diagnosis"
4. Review diagnosis and confidence

NOTE: Lab values are optional. Prediction
proceeds even with partial inputs.

MEDICAL DISCLAIMER:
For clinical decision support only.
Always exercise professional judgment.

============================================
"""
        self.results_text.config(state='normal')
        self.results_text.insert(1.0, msg)
        self.results_text.config(state='disabled')

    def _build_controls(self):
        bf = ttk.Frame(self.root)
        bf.grid(row=3, column=0, columnspan=2, pady=15)
        style = ttk.Style()
        style.configure("Predict.TButton", font=('Arial', 11, 'bold'), padding=10)
        style.configure("Clear.TButton", font=('Arial', 11), padding=10)
        self.predict_button = ttk.Button(bf, text="Predict Diagnosis",
                                         command=self.on_predict, style="Predict.TButton")
        self.predict_button.pack(side=tk.LEFT, padx=20)
        ttk.Button(bf, text="Clear Form", command=self.on_clear, style="Clear.TButton").pack(side=tk.LEFT, padx=15)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  font=('Arial', 11)).grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

    
    def _is_lab_value_available(self, value):
        """Check if lab value is available (not missing/zero)"""
        if value is None:
            return False
        if isinstance(value, str):
            return value.lower().strip() not in ['', '0', 'no', 'absent']
        if isinstance(value, (int, float)):
            return value != 0
        return True
    
    def _collect_inputs(self):
        d = {}
        
        # Collect demographics and clinical variables normally
        for name, var in self.demo_vars.items(): 
            d[name] = var.get()
        for name, var in self.clinical_vars.items(): 
            d[name] = var.get()
        
        # Handle lab variables - only include available ones
        available_labs = {}
        for name, var in self.lab_vars.items():
            raw_value = var.get()
            
            if self._is_lab_value_available(raw_value):
                # Only include lab values that are actually provided
                # All lab values are treated as numerical except categorical yes/no
                if isinstance(raw_value, str):
                    # Categorical lab values (yes/no)
                    available_labs[name] = raw_value
                else:
                    # Numerical lab values
                    available_labs[name] = float(raw_value)
        
        # Add available labs to the input dictionary
        d.update(available_labs)
        d['_available_labs'] = list(available_labs.keys())  # Track which labs are available
        
        return d

    def _encode_inputs(self, raw):
        enc = {}
        available_labs = raw.pop('_available_labs', [])  # Remove tracking variable
        
        for k, v in raw.items():
            if isinstance(v, str):
                vl = v.lower().strip()
                if k == 'Sex': enc[k] = 1 if vl == 'male' else 0
                elif vl in ('yes', 'no'): enc[k] = 1 if vl == 'yes' else 0
                elif k == 'Stool': enc[k] = {'normal': 0, 'constipation': 1, 'diarrhea': 2}.get(vl, 0)
                elif k == 'Severity': enc[k] = {'uncomplicated': 0, 'complicated': 1}.get(vl, 0)
                elif k == 'Ketones_in_Urine': enc[k] = {'no': 0, '+': 1, '++': 2, '+++': 3}.get(vl, 0)
                else:
                    try: enc[k] = float(v)
                    except: enc[k] = 0.0
            else:
                try: enc[k] = float(v)
                except: enc[k] = 0.0
        
        # Store available labs for preprocessing
        enc['_available_labs'] = available_labs
        return enc

    def on_predict(self):
        try:
            self.status_var.set("Processing prediction...")
            self.predict_button.config(state="disabled")

            model_name = self.model_var.get()
            if not model_name:
                messagebox.showwarning("No Model", "Please select a model.")
                self.predict_button.config(state="normal"); self.status_var.set("Ready"); return

            raw = self._collect_inputs()
            
            # Validate demographic data (cannot be zero)
            demographic_errors = []
            if raw.get('Age', 0) <= 0:
                demographic_errors.append("Age must be greater than 0")
            if raw.get('Weight', 0) <= 0:
                demographic_errors.append("Weight must be greater than 0")
            if raw.get('Height', 0) <= 0:
                demographic_errors.append("Height must be greater than 0")
            if raw.get('BMI', 0) <= 0:
                demographic_errors.append("BMI must be greater than 0")
            
            if demographic_errors:
                error_msg = "Please correct the following demographic errors:\n\n" + "\n".join(f"• {error}" for error in demographic_errors)
                messagebox.showwarning("Invalid Demographic Data", error_msg)
                self.predict_button.config(state="normal")
                self.status_var.set("Ready")
                return
            
            enc = self._encode_inputs(raw)

            features = self.preprocess_input(enc)
            result = self.predict(model_name, features)

            # Format patient data for display
            patient_data = []
            
            # Demographics
            patient_data.append(f"Age: {raw.get('Age', 'N/A')} years")
            patient_data.append(f"Sex: {'Male' if raw.get('Sex', 0) == 1 else 'Female'}")
            patient_data.append(f"Weight: {raw.get('Weight', 'N/A')} kg")
            patient_data.append(f"Height: {raw.get('Height', 'N/A')} cm")
            patient_data.append(f"BMI: {raw.get('BMI', 'N/A')}")
            patient_data.append(f"Body Temperature: {raw.get('Body_Temperature', 'N/A')}°C")
            
            # Clinical Symptoms
            symptoms = []
            symptom_list = [
                ("Lower Right Abdominal Pain", "Lower_Right_Abd_Pain"),
                ("Migratory Pain", "Migratory_Pain"),
                ("Loss of Appetite", "Loss_of_Appetite"),
                ("Nausea", "Nausea"),
                ("Coughing Pain", "Coughing_Pain"),
                ("Dysuria", "Dysuria"),
                ("Peritonitis", "Peritonitis"),
                ("Rebound Tenderness", "Ipsilateral_Rebound_Tenderness"),
                ("Psoas Sign", "Psoas_Sign")
            ]
            
            for display_name, field_name in symptom_list:
                value = raw.get(field_name, "no")
                # Show actual dropdown value (yes/no) with proper capitalization
                if isinstance(value, str):
                    status = value.capitalize()  # "yes" -> "Yes", "no" -> "No"
                else:
                    status = "Present" if value == 1 else "Absent"
                symptoms.append(f"{display_name}: {status}")
            
            severity = raw.get('Severity', 'uncomplicated')
            symptoms.append(f"Severity: {severity}")
            
            patient_data.append("\nClinical Symptoms:")
            for symptom in symptoms:
                patient_data.append(f"  • {symptom}")
            
            # Laboratory Results
            lab_results = []
            if raw.get('WBC_Count', 0) > 0: lab_results.append(f"WBC Count: {raw.get('WBC_Count')} K/μL")
            if raw.get('RBC_Count', 0) > 0: lab_results.append(f"RBC Count: {raw.get('RBC_Count')} M/μL")
            if raw.get('Hemoglobin', 0) > 0: lab_results.append(f"Hemoglobin: {raw.get('Hemoglobin')} g/dL")
            if raw.get('RDW', 0) > 0: lab_results.append(f"RDW: {raw.get('RDW')}%")
            if raw.get('Segmented_Neutrophils', 0) > 0: lab_results.append(f"Segmented Neutrophils: {raw.get('Segmented_Neutrophils')}%")
            if raw.get('Thrombocyte_Count', 0) > 0: lab_results.append(f"Thrombocyte Count: {raw.get('Thrombocyte_Count')} K/μL")
            if raw.get('CRP', 0) > 0: lab_results.append(f"CRP: {raw.get('CRP')} mg/L")
            if raw.get('Neutrophil_Percentage', 0) > 0: lab_results.append(f"Neutrophil Percentage: {raw.get('Neutrophil_Percentage')}%")
            if raw.get('Neutrophilia', 0) == 1: lab_results.append("Neutrophilia: Present")
            
            urine_results = []
            if raw.get('Ketones_in_Urine', 0) == 1: urine_results.append("Ketones")
            if raw.get('RBC_in_Urine', 0) == 1: urine_results.append("RBC")
            if raw.get('WBC_in_Urine', 0) == 1: urine_results.append("WBC")
            if urine_results:
                lab_results.append(f"Urine: {', '.join(urine_results)}")
            
            if lab_results:
                patient_data.append("\nLaboratory Results:")
                for lab_result in lab_results:
                    patient_data.append(f"  • {lab_result}")

            # Safely extract prediction results
            if isinstance(result, dict):
                diagnosis = result.get('diagnosis', 'Unknown')
                confidence = result.get('confidence', 0)
                prob_appendicitis = result.get('prob_appendicitis', 0)
                prob_no_appendicitis = result.get('prob_no_appendicitis', 0)
            else:
                diagnosis = str(result)
                confidence = 0
                prob_appendicitis = 0
                prob_no_appendicitis = 0

            output = f"""
PREDICTION RESULT
==================

PATIENT DATA
------------
{chr(10).join(patient_data)}

PREDICTION
----------
Model: {model_name}
Diagnosis: {diagnosis}
Confidence: {confidence:.1%}

Appendicitis Probability: {prob_appendicitis:.1%}
No Appendicitis Probability: {prob_no_appendicitis:.1%}

==================

MEDICAL DISCLAIMER:
This prediction is for clinical decision
support only. Always exercise professional
medical judgment.
"""
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, output)
            self.results_text.config(state='disabled')
            self.status_var.set(f"Prediction complete using {model_name}")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")
            self.status_var.set("Error")
        finally:
            self.predict_button.config(state="normal")

    def on_clear(self):
        for name, var in self.demo_vars.items():
            if name == 'Sex': var.set('Male')
            elif name == 'Age': var.set(10.0)
            elif name == 'Weight': var.set(35.0)
            elif name == 'Height': var.set(140.0)
            elif name == 'BMI': var.set(18.0)
            elif name == 'Neutrophil_Percentage': var.set(65.0)
            else: var.set(0.0)
        for name, var in self.clinical_vars.items():
            if name == 'Body_Temperature': var.set(37.5)
            elif name == 'Severity': var.set('uncomplicated')
            elif name == 'Stool': var.set('normal')
            else: var.set('no')
        for name, var in self.lab_vars.items():
            if isinstance(var, tk.StringVar): var.set('no')
            else: var.set(0.0)
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Form cleared. Enter new patient data and click Predict.")
        self.results_text.config(state='disabled')
        self.status_var.set("Form cleared")


def main():
    root = tk.Tk()
    app = PredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()