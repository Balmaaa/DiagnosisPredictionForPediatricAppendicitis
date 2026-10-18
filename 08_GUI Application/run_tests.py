from backend_predictor import AppendicitisPredictor

TEST_CASES = [
    {
        "name": "1. Appendicitis with Complete Labs",
        "data": {
            "Age": 10, "Sex": "female", "Weight": 35, "Height": 138, "BMI": 18.4,
            "Body_Temperature": 38.2,
            "Migratory_Pain": "Yes", "Lower_Right_Abd_Pain": "Yes", "Contralateral_Rebound_Tenderness": "Yes",
            "Coughing_Pain": "Yes", "Nausea": "Yes", "Loss_of_Appetite": "Yes", "Dysuria": "No",
            "Stool": "normal", "Peritonitis": "local", "Psoas_Sign": "Yes", "Ipsilateral_Rebound_Tenderness": "Yes",
            "Appendix_Diameter": 8.5, "WBC_Count": 14.2, "CRP": 25.0, "Neutrophil_Percentage": 85.0,
            "Segmented_Neutrophils": 78.0, "Neutrophilia": "Yes", "RBC_Count": 4.5, "Hemoglobin": 13.0,
            "RDW": 13.5, "Thrombocyte_Count": 310, "Ketones_in_Urine": "+", "RBC_in_Urine": "No", "WBC_in_Urine": "Yes",
            "Appendix_on_US": "Yes", "Free_Fluids": "Yes", "Target_Sign": "Yes"
        }
    },
    {
        "name": "2. Appendicitis without Labs (Missing Lab Test)",
        "data": {
            "Age": 12, "Sex": "male", "Weight": 42, "Height": 148, "BMI": 19.1,
            "Body_Temperature": 37.9,
            "Migratory_Pain": "Yes", "Lower_Right_Abd_Pain": "Yes", "Contralateral_Rebound_Tenderness": "Yes",
            "Coughing_Pain": "Yes", "Nausea": "Yes", "Loss_of_Appetite": "Yes", "Dysuria": "No",
            "Stool": "constipation", "Peritonitis": "No", "Psoas_Sign": "Yes", "Ipsilateral_Rebound_Tenderness": "Yes",
            "Appendix_on_US": "Yes", "Target_Sign": "Yes"
        }
    },
    {
        "name": "3. No Appendicitis with Complete Labs",
        "data": {
            "Age": 8, "Sex": "male", "Weight": 28, "Height": 125, "BMI": 17.9,
            "Body_Temperature": 36.8,
            "Migratory_Pain": "No", "Lower_Right_Abd_Pain": "No", "Contralateral_Rebound_Tenderness": "No",
            "Coughing_Pain": "No", "Nausea": "No", "Loss_of_Appetite": "No", "Dysuria": "No",
            "Stool": "normal", "Peritonitis": "No", "Psoas_Sign": "No", "Ipsilateral_Rebound_Tenderness": "No",
            "Appendix_Diameter": 5.0, "WBC_Count": 6.5, "CRP": 1.2, "Neutrophil_Percentage": 52.0,
            "Segmented_Neutrophils": 50.0, "Neutrophilia": "No", "RBC_Count": 4.6, "Hemoglobin": 13.5,
            "RDW": 12.8, "Thrombocyte_Count": 250, "Ketones_in_Urine": "No", "RBC_in_Urine": "No", "WBC_in_Urine": "No",
            "Appendix_on_US": "Yes", "Free_Fluids": "No"
        }
    },
    {
        "name": "4. No Appendicitis without Labs",
        "data": {
            "Age": 9, "Sex": "female", "Weight": 30, "Height": 130, "BMI": 17.7,
            "Body_Temperature": 36.9,
            "Migratory_Pain": "No", "Lower_Right_Abd_Pain": "No", "Contralateral_Rebound_Tenderness": "No",
            "Coughing_Pain": "No", "Nausea": "No", "Loss_of_Appetite": "No", "Dysuria": "No",
            "Stool": "normal", "Peritonitis": "No", "Psoas_Sign": "No", "Ipsilateral_Rebound_Tenderness": "No",
            "Appendix_on_US": "No"
        }
    },
    {
        "name": "5. Comprehensive Test with Ultrasound Complications",
        "data": {
            "Age": 11, "Sex": "male", "Weight": 38, "Height": 142, "BMI": 18.8,
            "Body_Temperature": 38.5,
            "Migratory_Pain": "Yes", "Lower_Right_Abd_Pain": "Yes", "Contralateral_Rebound_Tenderness": "Yes",
            "Coughing_Pain": "Yes", "Nausea": "Yes", "Loss_of_Appetite": "Yes", "Dysuria": "No",
            "Stool": "normal", "Peritonitis": "local", "Psoas_Sign": "Yes", "Ipsilateral_Rebound_Tenderness": "Yes",
            "Appendix_Diameter": 9.2, "WBC_Count": 15.1, "CRP": 30.5, "Neutrophil_Percentage": 88.0,
            "Segmented_Neutrophils": 81.0, "Neutrophilia": "Yes", "RBC_Count": 4.2, "Hemoglobin": 12.5,
            "RDW": 14.0, "Thrombocyte_Count": 330, "Ketones_in_Urine": "++", "RBC_in_Urine": "No", "WBC_in_Urine": "+",
            "Appendix_on_US": "Yes", "Free_Fluids": "Yes", "Target_Sign": "Yes",
            "Appendicular_Abscess": "Yes", "Pathological_Lymph_Nodes": "Yes", 
            "Bowel_Wall_Thickening": "Yes", "Ileus": "No", "Meteorism": "No",
            "Abscess_Location": "local", "Lymph_Node_Location": "right lower quadrant",
            "Conglomerate_of_Bowel_Loops": "No", "Coprostasis": "No", "Enteritis": "No"
        }
    }
]

def run_automated_tests():
    print("Initializing Appendicitis Predictor Backend for Automated Testing...")
    predictor = AppendicitisPredictor()
    print("=" * 70)

    for case in TEST_CASES:
        print(f"\n[RUNNING] {case['name']}")
        print("-" * 70)
        try:
            result = predictor.predict_all(case["data"])
            
            print(f"  -> Consensus Diagnosis : {result['diagnosis']}")
            print(f"  -> Model Agreement     : {result['agreement']}/{result['total_models']} ({result['confidence']} Confidence)")
            print(f"  -> Highest Confidence  : {result['highest_model']} ({result['highest_probability']:.2%})")
            print(f"  -> Lab Status          : {'Complete' if result['laboratory_available'] else 'Missing Values'}")
            if not result['laboratory_available']:
                print(f"      Missing Fields     : {result['missing_laboratory_fields']}")
            
            print("  -> Individual Model Breakdown:")
            for model_name, m_res in result["models"].items():
                print(f"     * {model_name:18} : {m_res['diagnosis']:16} (Prob: {m_res['prob_appendicitis']:.2%}, Threshold: {m_res['threshold']:.2f})")

        except Exception as e:
            print(f"  [ERROR] Test failed with exception: {e}")
        print("=" * 70)

if __name__ == "__main__":
    run_automated_tests()