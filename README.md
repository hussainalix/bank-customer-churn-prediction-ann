# Bank Customer Attrition Predictive Analytics Dashboard

An enterprise-grade, deployment-ready Streamlit web application that leverages an optimized multi-layer Artificial Neural Network (ANN) to evaluate and forecast customer churn risk metrics.

## Structural Features & Optimizations

* **Regularized Deep Architecture:** Built using TensorFlow/Keras with explicit Dropout layers (0.2) to mitigate overfitting during variance evaluation.
* **Automated Early Stopping:** Configured gradient descent convergence hooks using validation loss delta monitoring with a structural patience limit of 5 epochs.
* **Business-Centric Optimization:** Implemented a fine-tuned classification threshold constraint of 0.35 to maximize model Recall, minimizing high-value retention classification gaps.
* **Robust Preprocessing Pipeline:** Features explicit standard scaling parameters, vector array mapping transformations, and robust handling of categorical metrics.

## Local Execution Instructions

To execute this dashboard environment on your local server terminal, execute the following commands in sequence:

1. Clone or download the system directory repository containing `app.py` and `Bank_Churn.csv`.
2. Install the necessary system dependencies and libraries:
   ```bash
   pip install streamlit pandas numpy scikit-learn tensorflow
