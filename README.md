# 🩺 Breast Cancer Diagnosis Classifier

A clean, interpretable machine learning pipeline for classifying breast cancer tumors as **Malignant (M)** or **Benign (B)** using **Logistic Regression**. This project applies statistical preprocessing, feature selection, and threshold tuning to optimize classification performance.

---

## 📊 Dataset Overview

- **Source:** Breast Cancer Wisconsin Diagnostic Dataset  
- **Format:** CSV (`breast cancer data.csv`)  
- **Samples:** 569  
- **Features:** 32 clinical features + 1 target column (`diagnosis`)  

---

## 🧪 Workflow Summary

### 🔹 1. Data Preprocessing
- Dropped null and non-informative columns  
- Encoded categorical labels (`M` → 1, `B` → 0)  
- Performed statistical summary and data type inspections  

### 🔹 2. Feature Selection
- **Variance Thresholding:** Removed near-constant features  
- **Correlation Filtering:** Removed highly correlated features (correlation > 0.90)  
- **Result:** Reduced features from **32 ➝ 7**, retaining most predictive power  

### 🔹 3. Model Building
- **Algorithm:** Logistic Regression (scikit-learn)
- **Train-Test Split:** 80% training / 20% testing  
- **Pipeline:** Cleaned ➝ Reduced ➝ Labeled ➝ Modeled  

### 🔹 4. Evaluation Metrics
- ✅ **Accuracy**
- 🎯 **Precision**
- 🔁 **Recall**
- 📏 **F1-Score**
- 🧾 **Confusion Matrix**
- 📈 **ROC-AUC Curve**

### 🔹 5. Threshold Tuning
- Evaluated custom thresholds: `0.3`, `0.5`, `0.7`, `0.9`  
- Demonstrated effect on **recall vs. precision** trade-off  
- ROC curves plotted for each threshold

---

## ✅ Results Summary

| Metric       | Score       |
|--------------|-------------|
| **Accuracy** | `~93.9%`    |
| **Precision**| `~92.8%`    |
| **Recall**   | `~90.7%`    |
| **F1-Score** | `~91.7%`    |
| **ROC-AUC**  | Visualized via ROC Curve |

---

## 📉 Why Logistic Regression?

Logistic Regression is:
- 🔍 **Interpretable**  
- ⚡ **Fast to train**  
- 🧮 **Probabilistic**: Uses the **sigmoid function** for predictions  
- ✅ Well-suited for binary classification like `Malignant vs Benign`

---

## 🧠 Sigmoid Function

The **sigmoid function** transforms raw model outputs into probabilities between 0 and 1.

```
y = 1 / (1 + e^(-x))
```

- S-shaped curve
- Ideal for binary classification
- Underpins logistic regression output

---

## 🚀 How to Run

1. Clone the repository or use the code in your environment
2. Ensure your dataset is named `breast cancer data.csv`
3. Install required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4. Run the notebook or Python script

---

## 🛠 Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 🔮 Future Enhancements

- Apply **cross-validation** for more robust performance
- Explore advanced models (e.g., Random Forest, SVM)
- Automate preprocessing with pipelines
- Use **SHAP** or **LIME** for model explainability

---

## 📁 **How to Use**

### Clone the Repo
```bash
git clone https://github.com/your-username/breast-cancer-classification.git
```

### Open the Notebook
```bash
jupyter notebook Breast_Cancer_Logistic_Regression.ipynb
```

--- 

## 🙌 Acknowledgements
- **Dataset:** UCI Breast Cancer Wisconsin Dataset

- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

--- 

## 📧 **Contact**
Made with ❤️ by [Chaiithra Thota]

Connect on LinkedIn:(https://www.linkedin.com/in/chaiithrathota/)

Connect on Twitter: (https://x.com/DebugDiary_)

---

## 📌 Author Notes

This project showcases a minimalist yet effective approach to medical classification tasks. By focusing on **feature reduction**, **model transparency**, and **threshold control**, it delivers a performant and explainable classifier.

---
```
