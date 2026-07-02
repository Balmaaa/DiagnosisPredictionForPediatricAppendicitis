# Pediatric Appendicitis Prediction System - GUI Application

## Overview

This is a comprehensive Graphical User Interface (GUI) application for pediatric appendicitis diagnosis prediction using multiple AI models. The system implements the research from your SP Proposal, providing a user-friendly interface for medical professionals to input patient data and receive appendicitis predictions.

## Features

### **AI Models Available**
- **Transformer Model** - Deep learning approach (main contribution)
- **Decision Tree** - Traditional machine learning
- **Gradient Boosting** - Ensemble method
- **XGBoost** - Optimized gradient boosting

### **Comprehensive Data Input**
- **Demographic Information**: Age, Weight, Height, BMI, Sex
- **Clinical Symptoms**: Pain patterns, nausea, fever, physical examination findings
- **Laboratory Results**: Blood counts, inflammatory markers, urine analysis
- **Imaging Findings**: Ultrasound results, radiological findings

### **Key Features**
- Model selection interface
- Real-time prediction
- Confidence scores
- Medical interpretation
- Input validation
- Form reset functionality
- Professional medical reports

## Installation

### Prerequisites
- Python 3.8 or higher
- Trained models from the main project
- Preprocessing pipeline

### Setup Instructions

1. **Navigate to the GUI Application Directory**
   ```bash
   cd "09_GUI_Application"
   ```

2. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Model Availability**
   Ensure the following directories contain trained models:
   - `../05_Transformer_Model/` - Transformer model files
   - `../06_Decision_Trees/` - Decision Tree model files
   - `../07_Gradient_Boosting/` - Gradient Boosting model files
   - `../08_XGBoost/` - XGBoost model files
   - `../04_Preprocessing Pipeline/` - Preprocessing pipeline

## Usage

### Starting the Application

1. **Run the GUI Application**
   ```bash
   python prediction_gui.py
   ```

2. **Alternative: Run Backend Test**
   ```bash
   python backend_predictor.py
   ```

### Using the Interface

#### 1. **Model Selection**
   - Choose from 4 available AI models
   - Each model has different performance characteristics
   - XGBoost typically shows best overall accuracy
   - Transformer shows highest specificity

#### 2. **Data Input Tabs**

**Demographic Information Tab:**
- Enter patient's basic demographic data
- Age, Weight, Height, BMI
- Sex selection

**Clinical Symptoms Tab:**
- Input clinical presentation
- Pain characteristics
- Physical examination findings
- Symptom severity

**Laboratory Results Tab:**
- Enter blood test results
- Inflammatory markers
- Urine analysis results

**Imaging Findings Tab:**
- Ultrasound findings
- Radiological observations
- Specific imaging measurements

#### 3. **Prediction Process**
1. Complete all relevant fields
2. Select desired AI model
3. Click "Predict Diagnosis"
4. Review results and confidence scores
5. Follow medical recommendations

#### 4. **Results Interpretation**

**Positive Prediction (Appendicitis):**
- WARNING: High risk detected
- Immediate medical evaluation recommended
- Consider surgical consultation
- Monitor for worsening symptoms

**Negative Prediction (No Appendicitis):**
- OK: Low risk assessment
- Continue monitoring
- Consider alternative diagnoses
- Re-evaluate if symptoms worsen

## Model Performance

Based on the comprehensive evaluation:

| Model | Accuracy | Sensitivity | Specificity | Best Use Case |
|--------|----------|-------------|-------------|---------------|
| **XGBoost** | 73.08% | 74.05% | 71.65% | Overall screening |
| **Gradient Boosting** | 72.44% | 72.97% | 71.65% | Balanced performance |
| **Decision Tree** | 70.51% | 66.49% | 76.38% | Quick assessment |
| **Transformer** | 69.87% | 63.24% | 79.53% | High specificity needs |

## Medical Guidelines

### **Important Disclaimer**
- This system is for **medical reference only**
- Clinical judgment should always prevail
- Not a substitute for professional medical evaluation
- Use as a decision support tool, not definitive diagnosis

### **Recommended Workflow**
1. Complete comprehensive patient assessment
2. Use system as additional decision support
3. Consider model confidence scores
4. Evaluate in context of clinical presentation
5. Follow standard medical protocols

## Technical Architecture

### Backend Components
- **ModelLoader**: Loads trained AI models
- **AppendicitisPredictor**: Handles preprocessing and inference
- **Data Validation**: Ensures input quality
- **Medical Interpretation**: Provides clinical context

### Frontend Components
- **Tabbed Interface**: Organized data input
- **Model Selection**: Choose AI model
- **Real-time Processing**: Immediate predictions
- **Results Display**: Comprehensive output
- **Error Handling**: Robust error management

### Data Processing Pipeline
1. Input validation
2. Data preprocessing (same as training)
3. Model inference
4. Post-processing
5. Results formatting

## Troubleshooting

### Common Issues

**Models Not Loading:**
- Verify model files exist in correct directories
- Check file permissions
- Ensure preprocessing pipeline is available

**Prediction Errors:**
- Check all required fields are filled
- Verify input data ranges
- Ensure model compatibility

**GUI Issues:**
- Update tkinter if needed
- Check Python version compatibility
- Verify all dependencies installed

### Error Messages

**"Model not loaded":**
- Check model files exist
- Verify file paths
- Restart application

**"Input validation failed":**
- Check required fields
- Verify data ranges
- Ensure proper data types

**"Prediction error":**
- Check model availability
- Verify data preprocessing
- Review input data quality

## Development

### File Structure
```
09_GUI_Application/
├── prediction_gui.py            # Main GUI application (optimized)
├── backend_predictor.py         # Backend prediction logic
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── ../                          # Parent project directories
    ├── 04_Preprocessing Pipeline/
    ├── 05_Transformer_Model/
    ├── 06_Decision_Trees/
    ├── 07_Gradient_Boosting/
    └── 08_XGBoost/
```

### Extending the Application

**Adding New Models:**
1. Train new model in respective directory
2. Update ModelLoader class
3. Add model selection option
4. Test integration

**Enhancing GUI:**
1. Modify prediction_gui.py
2. Add new input fields as needed
3. Update validation logic
4. Test user experience

**Improving Backend:**
1. Enhance backend_predictor.py
2. Add new preprocessing steps
3. Improve error handling
4. Optimize performance

## Support

For technical support or questions:
1. Check this README file
2. Review error messages
3. Verify installation steps
4. Consult the main project documentation

## License

This GUI application is part of the Pediatric Appendicitis Prediction System research project. Use according to academic and research guidelines.

---

**Version**: 1.0.0  
**Last Updated**: 2026-02-02  
**Compatible with**: SP Proposal Requirements  
**Python Version**: 3.8+
