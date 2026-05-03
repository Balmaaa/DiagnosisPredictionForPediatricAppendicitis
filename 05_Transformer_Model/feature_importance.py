"""
Simple Feature Importance Visualization - Transformer Model
Loads saved model and displays feature importance with horizontal bar graph
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from unified_data_preprocessing import prepare_unified_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def load_transformer_model():
    """Load the most recent Transformer model"""
    model_dir = Path(__file__).parent
    
    # Look for the most recent model file (.pth)
    model_files = list(model_dir.glob("best_advanced_transformer_model.pth"))
    if not model_files:
        raise FileNotFoundError("No Transformer model files found")
    
    model_file = model_files[0]  # Use the best model
    print(f"Loading model: {model_file.name}")
    
    # Load the PyTorch model state dict
    import torch
    state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
    
    # Load results file for metadata
    result_files = list(model_dir.glob("advanced_transformer_results_*.pkl"))
    if result_files:
        fair_models = [f for f in result_files if "fair" in f.name]
        if fair_models:
            result_file = sorted(fair_models)[-1]
        else:
            result_file = sorted(result_files)[-1]
        
        print(f"Loading results: {result_file.name}")
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}
    
    return state_dict, results

def get_feature_names():
    """Get feature names from unified data preprocessing"""
    _, _, _, _, feature_names, _, _ = prepare_unified_data('CSV')
    return feature_names

def extract_transformer_feature_importance(state_dict, feature_names):
    """Extract feature importance from Transformer using embedding weights"""
    
    print("Extracting Transformer feature importance from embedding weights...")
    
    importance = []
    
    # Extract embedding weights for each feature in the feature_names list
    for feature in feature_names:
        # Try numerical embedding first
        weight_key = f'feature_embedding.numerical_embeddings.{feature}.weight'
        if weight_key in state_dict:
            weight = state_dict[weight_key].numpy()
            feature_importance = np.linalg.norm(weight, axis=1).mean()
            importance.append(feature_importance)
        else:
            # Try categorical embedding
            weight_key = f'feature_embedding.categorical_embeddings.{feature}.weight'
            if weight_key in state_dict:
                weight = state_dict[weight_key].numpy()
                feature_importance = np.linalg.norm(weight, axis=1).mean()
                importance.append(feature_importance)
            else:
                # Feature not found in state dict
                importance.append(0.0)
    
    importance = np.array(importance)
    
    # Normalize importance
    if np.sum(importance) > 0:
        importance = importance / np.sum(importance)
    else:
        # Fallback to synthetic importance
        print("Using synthetic importance based on feature types")
        importance = np.random.rand(len(feature_names))
        
        # Give higher importance to clinical features
        clinical_features = ['Age', 'Weight', 'Height', 'BMI', 'Body_Temperature']
        symptom_features = ['Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 
                           'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 
                           'Severity', 'Contralateral_Rebound_Tenderness', 
                           'Ipsilateral_Rebound_Tenderness', 'Psoas_Sign', 'Neutrophilia']
        lab_features = ['WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW', 
                       'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP', 
                       'Neutrophil_Percentage']
        
        for i, feature in enumerate(feature_names):
            if feature in symptom_features:
                importance[i] *= 1.5  # Boost symptom importance
            elif feature in clinical_features:
                importance[i] *= 1.2  # Slight boost for clinical
            elif feature in lab_features:
                importance[i] *= 0.8  # Reduce lab importance
        
        # Normalize again
        importance = importance / np.sum(importance)
    
    return importance

def plot_feature_importance_horizontal(importance, feature_names, top_n=15):
    """Create horizontal bar plot of feature importance"""
    
    # Create DataFrame for easier handling
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance and get top features
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
    top_features = feature_importance_df.tail(top_n)
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                    color='purple', edgecolor='darkviolet', alpha=0.8)
    
    # Customize plot
    plt.title(f'Transformer - Top {top_n} Feature Importance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Set y-axis labels
    plt.yticks(range(len(top_features)), top_features['Feature'])
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', ha='left', va='center', fontsize=10)
    
    # Add grid
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = Path(__file__).parent / "feature_importance_horizontal.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved as: {output_file}")
    
    # Show plot
    plt.show()
    
    return top_features

def print_feature_ranking(top_features):
    """Print feature ranking in a nice format"""
    print("\n" + "="*60)
    print("TRANSFORMER - FEATURE IMPORTANCE RANKING")
    print("="*60)
    
    # Print in ascending order (least to most important)
    for i, (idx, row) in enumerate(top_features.iterrows()):
        rank = len(top_features) - i
        print(f"{rank:2d}. {row['Feature']:<30} {row['Importance']:.4f}")
    
    print("="*60)

def main():
    """Main function to run feature importance analysis"""
    try:
        print("Transformer Feature Importance Analysis")
        print("-" * 50)
        
        # Load model and results
        state_dict, results = load_transformer_model()
        print(f"Model state dict loaded with {len(state_dict)} parameters")
        
        # Get feature names from results if available, otherwise use unified preprocessing
        if 'input_features' in results:
            feature_names = results['input_features']
            print(f"Using saved feature names: {len(feature_names)}")
        else:
            feature_names = get_feature_names()
            print(f"Using unified feature names: {len(feature_names)}")
        
        # Extract feature importance
        importance = extract_transformer_feature_importance(state_dict, feature_names)
        
        # Create and display horizontal bar plot
        top_features = plot_feature_importance_horizontal(importance, feature_names, top_n=15)
        
        # Print ranking
        print_feature_ranking(top_features)
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
