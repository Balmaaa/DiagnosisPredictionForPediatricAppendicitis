import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparison:
    """Compare performance of all models for appendicitis prediction"""
    
    def __init__(self):
        self.models_data = {}
        self.comparison_metrics = {}
        
    def load_model_results(self):
        """Load results from all model folders"""
        
        base_path = Path(__file__).parent
        
        model_folders = {
            'Transformer': base_path / "05_Transformer_Model",
            'Decision Tree': base_path / "06_Decision_Trees", 
            'Gradient Boosting': base_path / "07_Gradient_Boosting",
            'XGBoost': base_path / "08_XGBoost"
        }
        
        for model_name, folder_path in model_folders.items():
            print(f"Loading results for {model_name}...")
            
            result_files = list(folder_path.glob("*results*.pkl"))
            
            if not result_files:
                print(f"No results found for {model_name}")
                continue
                
            # Load the most recent result file
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            
            try:
                with open(latest_file, 'rb') as f:
                    results = pickle.load(f)
                
                self.models_data[model_name] = results
                print(f"Loaded {model_name} results from {latest_file.name}")
                
            except Exception as e:
                print(f"Error loading {model_name} results: {e}")
    
    def create_comparison_table(self):
        """Create comparison table of all models"""
        
        comparison_data = []
        
        for model_name, results in self.models_data.items():
            if 'final_metrics' in results:
                metrics = results['final_metrics']
                
                # Calculate F1 Score
                precision = metrics.get('precision', 0)
                recall = metrics.get('sensitivity', 0)
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0
                
                comparison_data.append({
                    'Model': model_name,
                    'Dataset': results.get('dataset_name', 'Unknown'),
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Sensitivity': metrics.get('sensitivity', 0),
                    'Specificity': metrics.get('specificity', 0),
                    'F1 Score': f1_score,
                    'PPV': metrics.get('ppv', 0),
                    'NPV': metrics.get('npv', 0),
                    'True Positives': metrics.get('tp', 0),
                    'True Negatives': metrics.get('tn', 0),
                    'False Positives': metrics.get('fp', 0),
                    'False Negatives': metrics.get('fn', 0)
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
        
        print("\n" + "="*100)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*100)
        
        # Group by dataset
        for dataset in self.comparison_df['Dataset'].unique():
            print(f"\n{'='*60}")
            print(f"DATASET: {dataset.upper()}")
            print(f"{'='*60}")
            
            dataset_df = self.comparison_df[self.comparison_df['Dataset'] == dataset]
            
            # Format the table
            print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1 Score':<10}")
            print("-" * 80)
            
            for _, row in dataset_df.iterrows():
                print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Sensitivity']:<12.4f} {row['Specificity']:<12.4f} {row['F1 Score']:<10.4f}")
                
                # Add confusion matrix for each model
                print(f"\n{row['Model']} Confusion Matrix:")
                print("Predicted")
                print("        Yes    No")
                print(f"Actual Yes     {row['True Positives']:<6} {row['False Negatives']:<6}")
                print(f"       No      {row['False Positives']:<6} {row['True Negatives']:<6}")
                print("-" * 40)
            
            # Find best model for each metric
            best_accuracy = dataset_df.loc[dataset_df['Accuracy'].idxmax()]
            best_sensitivity = dataset_df.loc[dataset_df['Sensitivity'].idxmax()]
            best_specificity = dataset_df.loc[dataset_df['Specificity'].idxmax()]
            
            print(f"\nBest Performing Models:")
            print(f"  Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
            print(f"  Sensitivity: {best_sensitivity['Model']} ({best_sensitivity['Sensitivity']:.4f})")
            print(f"  Specificity: {best_specificity['Model']} ({best_specificity['Specificity']:.4f})")
    
    def plot_model_comparison(self, save_path=None):
        """Create visualization of model comparison"""
        
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5CF6']
        
        for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten()[:5])):
            # Group by dataset
            for dataset_idx, dataset in enumerate(self.comparison_df['Dataset'].unique()):
                dataset_df = self.comparison_df[self.comparison_df['Dataset'] == dataset]
                
                x_pos = np.arange(len(dataset_df))
                width = 0.35
                
                # Create bars
                bars = ax.bar(
                    x_pos + dataset_idx * width, 
                    dataset_df[metric], 
                    width, 
                    label=dataset,
                    color=colors[dataset_idx],
                    alpha=0.8
                )
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.01,
                        f'{height:.3f}',
                        ha='center', 
                        va='bottom',
                        fontsize=8
                    )
            
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos + width/2)
            ax.set_xticklabels(dataset_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide the unused subplot
        axes.flatten()[5].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_feature_importance(self):
        """Analyze and compare feature importance across models"""
        
        print("\n" + "="*100)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*100)
        
        for model_name, results in self.models_data.items():
            if 'feature_importance' in results:
                print(f"\n{'='*60}")
                print(f"TOP FEATURES - {model_name.upper()}")
                print(f"{'='*60}")
                
                feature_importance = results['feature_importance']
                print(feature_importance.to_string(index=False))
                print()
    
    def generate_summary_report(self, save_path=None):
        """Generate comprehensive summary report"""
        
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
        
        report = []
        report.append("="*100)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("Pediatric Appendicitis Prediction System")
        report.append("="*100)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall summary
        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 50)
        
        # Find best overall model
        best_overall = self.comparison_df.loc[self.comparison_df['Accuracy'].idxmax()]
        report.append(f"Best Overall Model: {best_overall['Model']} ({best_overall['Dataset']})")
        report.append(f"Best Accuracy: {best_overall['Accuracy']:.4f}")
        report.append("")
        
        # Dataset summary
        for dataset in self.comparison_df['Dataset'].unique():
            dataset_df = self.comparison_df[self.comparison_df['Dataset'] == dataset]
            
            report.append(f"DATASET: {dataset.upper()}")
            report.append("-" * 30)
            
            for _, row in dataset_df.iterrows():
                report.append(f"{row['Model']}:")
                report.append(f"  Accuracy: {row['Accuracy']:.4f}")
                report.append(f"  Precision: {row['Precision']:.4f}")
                report.append(f"  Sensitivity: {row['Sensitivity']:.4f}")
                report.append(f"  Specificity: {row['Specificity']:.4f}")
                report.append(f"  F1 Score: {row['F1 Score']:.4f}")
                report.append(f"  PPV: {row['PPV']:.4f}")
                report.append(f"  NPV: {row['NPV']:.4f}")
                report.append("")
        
        # Medical interpretation
        report.append("MEDICAL INTERPRETATION:")
        report.append("-" * 50)
        report.append("• Sensitivity (True Positive Rate): Ability to correctly identify appendicitis cases")
        report.append("• Specificity (True Negative Rate): Ability to correctly identify non-appendicitis cases")
        report.append("• F1 Score: Harmonic mean of precision and sensitivity")
        report.append("• PPV (Positive Predictive Value): Probability that positive prediction is correct")
        report.append("• NPV (Negative Predictive Value): Probability that negative prediction is correct")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)
        
        # Find models with best medical metrics
        best_sensitivity = self.comparison_df.loc[self.comparison_df['Sensitivity'].idxmax()]
        best_specificity = self.comparison_df.loc[self.comparison_df['Specificity'].idxmax()]
        
        report.append(f"• For screening (high sensitivity needed): {best_sensitivity['Model']} ({best_sensitivity['Sensitivity']:.4f})")
        report.append(f"• For confirmation (high specificity needed): {best_specificity['Model']} ({best_specificity['Specificity']:.4f})")
        report.append("• Consider ensemble approach for balanced performance")
        report.append("")
        
        report.append("="*100)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to: {save_path}")
        
        return report_text
    
    def save_comparison_results(self, base_filename="model_comparison"):
        """Save all comparison results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison table
        if hasattr(self, 'comparison_df'):
            csv_file = f"{base_filename}_table_{timestamp}.csv"
            self.comparison_df.to_csv(csv_file, index=False)
            print(f"Comparison table saved to: {csv_file}")
        
        # Save plots
        plot_file = f"{base_filename}_plot_{timestamp}.png"
        self.plot_model_comparison(save_path=plot_file)
        
        # Save summary report
        report_file = f"{base_filename}_report_{timestamp}.txt"
        self.generate_summary_report(save_path=report_file)
        
        return {
            'csv_file': csv_file,
            'plot_file': plot_file, 
            'report_file': report_file
        }

def main():
    """Main function for model comparison"""
    
    print("="*60)
    print("MODEL COMPARISON FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("="*60)
    
    try:
        # Initialize comparison
        comparison = ModelComparison()
        
        # Load all model results
        comparison.load_model_results()
        
        # Create and display comparison table
        comparison.create_comparison_table()
        comparison.print_comparison_table()
        
        # Analyze feature importance
        comparison.analyze_feature_importance()
        
        # Generate plots
        comparison.plot_model_comparison()
        
        # Generate summary report
        summary_report = comparison.generate_summary_report()
        print("\n" + summary_report)
        
        # Save all results
        saved_files = comparison.save_comparison_results()
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON COMPLETED")
        print(f"{'='*60}")
        print("Files saved:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
