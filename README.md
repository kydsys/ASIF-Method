Below is a paper-style English README suitable for a short paper / workshop paper / supplementary material.
It is strictly derived from your main.py / test.py / metrics.py logic and terminology.

README
1. Overview

  This work presents a confidence-aware and coverage-gated classification framework for automated recognition in long-term monitoring scenarios.
  The framework combines:
  
    Deep convolutional classification (ResNeXt50),
    Post-hoc probability calibration via Temperature Scaling,
    A multi-stage gating strategy based on confidence, class-level precision/recall, and coverage.
  
  The goal is to automatically accept high-confidence, stable classes, while routing uncertain samples to a low-confidence or manual-processing pipeline.
  
  The complete workflow is implemented in three scripts:
    main.py – training, calibration, and allow-list construction
    test.py – calibrated inference and automatic filtering on test data
    metrics.py – evaluation of the automatically accepted results

2. Pipeline Summary
  The proposed pipeline consists of three sequential stages:
  
  1.Training & Calibration
    Train a multi-class classifier on labeled images.
    Apply Temperature Scaling on a validation set to correct overconfidence.
    Select reliable classes using precision, recall, and coverage constraints.
  
  2.Inference & Automatic Gating
    Perform temperature-scaled inference on the test set.
    Automatically accept predictions that satisfy both:
      High confidence,
      Membership in a pre-defined allow-list.
  
  3.Evaluation
    Compute per-class and overall metrics on the automatically accepted subset.
  
3. Directory Structure
  .
  ├── main.py                     # Training + calibration + gate construction
  ├── test.py                     # Test-time inference and gating
  ├── metrics.py                  # Evaluation on accepted samples
  │
  ├── Step1/
  │   ├── model.pth               # Trained ResNeXt50 model
  │   ├── model_T.json            # Learned temperature parameter
  │   ├── label_to_index.json     # Label ↔ index mapping
  │   ├── results.csv             # Validation predictions (raw)
  │   ├── results_t.csv           # Validation predictions (temperature-scaled)
  │   ├── results_t_filtered.csv  # Confidence-filtered validation results
  │   ├── metrics.csv             # Validation metrics
  │   └── gate/
  │       ├── gate_table.csv      # Coverage statistics per class
  │       ├── gate_selected.csv   # Classes passing coverage threshold
  │       ├── allow_list.csv      # Final allow-list
  │       └── gate_coverage.png   # Coverage curve visualization
  │
  ├── integrated_f1_scores.csv    # Final allow-list used for testing
  ├── Results.csv                 # Automatically accepted test samples
  ├── low_config.csv              # Rejected / low-confidence samples
  └── raw/
      └── test_all_scaled.csv     # Full test predictions with probabilities
4. Data Format

  All datasets (training, validation, testing) must be provided as CSV files with the following columns:
    filename,label
    filename: relative path to the image file
    label: class identifier (treated as a string)
    
  Images are loaded using:
    image_path / filename
  
5. Training and Calibration (main.py)
  1 Model
    Backbone: ResNeXt50
    Loss: Cross-Entropy
    Optimizer: AdamW
    Learning rate warm-up and step decay scheduling
    Early stopping based on validation accuracy
  
  2 Temperature Scaling
    After training, the model is calibrated on the validation set by optimizing a single scalar temperature 𝑇 T to minimize negative log-likelihood:softmax(Z/T)
    This step improves the reliability of predicted probabilities without changing classification accuracy.
  
  3 Class Selection Criteria
    A class is considered reliable if it satisfies all of the following:
    Confidence constraint
    Sample-level maximum probability ≥ confidence_threshold
    Performance constraint
    Precision ≥ Recall_Precision
    Recall ≥ Recall_Precision
    Coverage constraint
    Coverage ≥ coverage_threshold
    Classes passing all criteria form the allow-list.
  
6. Test-Time Inference and Gating (test.py)
  1 Inference
    Load the trained model and calibrated temperature 𝑇
    Apply temperature-scaled softmax to all test samples.
  2 Automatic Acceptance Rule
    A test sample is automatically accepted if:
      Predicted class ∈ Allow-list
      AND
      max(probabilities) ≥ confidence_threshold
  3 Output Files
    Results.csv
    Automatically accepted samples.
  
    low_config.csv
    Samples rejected due to low confidence or non-allow-listed predictions.
  
    raw/test_all_scaled.csv
    Full test predictions with calibrated probabilities (for analysis).

7.Evaluation (metrics.py)
  Evaluation is performed only on automatically accepted samples.
  1.Metrics
    For each class:
    True Positives (TP)
    False Positives (FP)
    False Negatives (FN)
    True Negatives (TN)
    Precision
    Recall
    F1-score
  Additionally, overall accuracy is reported for the accepted subset.
  2 Output
    metrics.csv
    Columns:
      Species, TP, FP, FN, TN, Recall, Precision, F1-Score
8.Key Characteristics of the Method
    Post-hoc calibration improves probability interpretability.
    Dual-level gating:
    Sample-level (confidence),
    Class-level (precision, recall, coverage).
    Designed for long-term, real-world monitoring with asymmetric error costs.
    Naturally balances automation rate and reliability.

9.Typical Use Case
  Ecological or wildlife image monitoring
  Continuous camera trap deployment
  Scenarios requiring high-precision automatic recognition with controlled risk

10. Reproducibility
  Recommended execution order:
  # 1. Train, calibrate, and build allow-list
  python main.py
  
  # 2. Run calibrated inference and automatic filtering
  python test.py
  
  # 3. Evaluate automatic results
  python metrics.py

Datasets: see [code/DATASETS.md](code/DATASETS.md).


