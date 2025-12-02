# **Model Card: Adult Census Income Prediction Model**

## **1\. Model Details**

* **Model Name:** Adult Census Income Prediction Model  
* **Version:** 1.0  
* **Developers:** \[Joshua Drake\]  
* **Date:** December 1, 2025  
* **Model Type:** Classification Model  
* **Architecture:** Scikit-learn RandomForestClassifier  
  * *Brief Description:* The RandomForestClassifier is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) of the individual trees. It is known for its robustness against overfitting and good performance on various datasets.

## **2\. Intended Use**

* **Primary Use Cases:**  
  * Predicting whether an individual's income exceeds $50K based on demographic and employment data.  
  * Exploring the relationship between various features and income levels.  
  * Serving as a baseline model for more complex income prediction systems.  
* **Target Users:**  
  * Data scientists and machine learning engineers developing or deploying income prediction systems.  
  * Researchers interested in socio-economic factors influencing income.  
* **Out-of-Scope Use Cases:**  
  * Making definitive statements about an individual's actual income.  
  * Making hiring, lending, or any other critical individual-level decisions.  
  * Use in highly sensitive contexts where fairness and bias mitigation have not been thoroughly addressed and validated. This model is for illustrative and educational purposes and should not be used in real-world applications without extensive auditing and regulatory compliance.

## **3\. Training Data**

* **Dataset:** Adult Census Income dataset (census.csv)  
* **Source:** Derived from the UCI Adult dataset.  
* **Description:** The dataset contains demographic information from the 1994 Census, including features such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country. The target variable is 'salary', indicating whether an individual's income is \<=50K or \>50K.  
* **Number of Samples (Training):** 32,561  
* **Features Used:**  
  * age (numerical)  
  * workclass (categorical)  
  * fnlwgt (numerical)  
  * education (categorical)  
  * education\_num (numerical)  
  * marital\_status (categorical)  
  * occupation (categorical)  
  * relationship (categorical)  
  * race (categorical)  
  * sex (categorical)  
  * capital\_gain (numerical)  
  * capital\_loss (numerical)  
  * hours\_per\_week (numerical)  
  * native\_country (categorical)  
* **Label:** salary (binary: \<=50K or \>50K)

## **4\. Evaluation Data**

* **Dataset:** A held-out test set from the Adult Census Income dataset (census.csv).  
* **Description:** This data was not used during model training and serves as an independent evaluation set.  
* **Number of Samples (Test):** 16,281

## **5\. Metrics**

The model's performance is evaluated using the following metrics:

* **Precision:** 0.7327  
  * Formula: $Precision \= \\frac{TP}{TP \+ FP}$  
* **Recall:** 0.6397  
  * Formula: $Recall \= \\frac{TP}{TP \+ FN}$  
* **F-beta Score (with Beta=0.5):** 0.7120  
  * Formula: $F\_\\beta \= (1 \+ \\beta^2) \\cdot \\frac{Precision \\cdot Recall}{(\\beta^2 \\cdot Precision) \+ Recall}$  
  * For $\\beta \= 0.5$: $F\_{0.5} \= 1.25 \\cdot \\frac{Precision \\cdot Recall}{(0.25 \\cdot Precision) \+ Recall}$

## **6\. Performance**

### **Overall Model Performance**

* **Precision:** 0.7327  
* **Recall:** 0.6397  
* **F-beta (Beta=0.5) Score:** 0.7120

### **Performance on Categorical Slices**

(This section will contain summarized insights from slice\_output.txt. You should read the slice\_output.txt and describe any interesting observations or disparities here. Provide a few examples from the file.)

* **Workclass:**  
  * Federal-gov: Precision: 0.7286 | Recall: 0.6711 | F1: 0.7163  
  * Private: Precision: 0.7366 | Recall: 0.6264 | F1: 0.7116  
  * Never-worked: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 (Note: Count is 1, indicating unreliable metrics)  
* **Race:**  
  * White: Precision: 0.7319 | Recall: 0.6447 | F1: 0.7126  
  * Black: Precision: 0.7941 | Recall: 0.5806 | F1: 0.7397  
* **Sex:**  
  * Male: Precision: 0.7274 | Recall: 0.6432 | F1: 0.7088  
  * Female: Precision: 0.7638 | Recall: 0.6204 | F1: 0.7301  
* **Other notable slices:**  
  * native\_country: Many countries like Vietnam, Portugal, and France have a Recall of 0.0000 and Precision of 1.0000 (or vice versa) due to extremely low sample counts (often 0 in the test set for specific labels), making these metrics unreliable.

**Observations from Slice Performance:**

The model generally performs consistently across the major demographic groups. For example, precision and recall for Male (Precision 0.73, Recall 0.64) and Female (Precision 0.76, Recall 0.62) are comparable, though there is a slight trade-off where the model is more precise but less sensitive for females. Similarly, the White and Black racial groups show comparable F-beta scores (0.71 vs 0.74), suggesting no gross disparity in overall predictive power for these specific metrics, although the Black subgroup has a notably lower recall (0.58 vs 0.64), meaning the model misses more high-earners in this group.

However, performance on less frequent categories is highly volatile. For instance, the Never-worked and Without-pay workclasses show perfect scores (1.0 across the board), but this is due to a sample size of only 1 in the test set. Similarly, many native\_country categories show 0.00 or 1.00 metrics because they have zero or very few positive instances in the evaluation data. This indicates that while the model handles the majority classes well, its generalization to rare subpopulations is statistically uncertain and likely unreliable.

## **7\. Limitations and Bias**

* **Data Bias:** The training data (1994 Census) reflects societal biases and inequalities of that specific time. The model will learn and perpetuate these biases. For example, historical wage gaps based on gender or race might be reflected in the model's predictions.  
* **Lack of Causal Inference:** This model is predictive, not causal. It identifies correlations but does not explain *why* certain features are associated with higher income.  
* **Limited Features:** The features available are a snapshot of demographic and employment data. Many other factors (e.g., socioeconomic background, geographic location within the US, specific job roles, soft skills, economic climate) that influence income are not included, limiting the model's completeness.  
* **Generalization:** Performance on unseen data might vary, especially for subpopulations that are underrepresented in the training data. The F-beta score prioritizes precision, which might lead to the model being more conservative in predicting '\>50K', potentially missing some true positives (lower recall).  
* **Fairness:** As observed in the slice performance, there are likely disparities in model performance across different demographic groups. Deploying such a model in sensitive applications without rigorous fairness audits and mitigation strategies could lead to unfair outcomes.

## **8\. Ethical Considerations**

* **Discrimination:** Using this model in areas like hiring, loan applications, or social benefits could lead to discriminatory outcomes if not carefully monitored and mitigated for bias.  
* **Transparency:** While RandomForest is more interpretable than deep learning models, the exact decision-making process for an individual prediction can still be opaque.  
* **Privacy:** The dataset contains sensitive personal information, and any deployment must adhere to strict data privacy regulations.

## **9\. Future Improvements**

* **More Diverse Data:** Incorporate more recent and diverse datasets to reduce historical biases and improve generalization.  
* **Feature Engineering:** Create more informative features (e.g., interaction terms, polynomial features) or gather additional relevant data.  
* **Hyperparameter Tuning:** Systematically tune the RandomForestClassifier's hyperparameters (e.g., n\_estimators, max\_depth, min\_samples\_leaf) for optimal performance.  
* **Model Selection:** Experiment with other classification algorithms (e.g., Gradient Boosting Machines like LightGBM/XGBoost, Logistic Regression, Neural Networks).  
* **Bias Mitigation Techniques:** Implement techniques like re-weighting training samples, adversarial debiasing, or post-processing predictions to reduce unfairness.  
* **Explainable AI (XAI):** Integrate XAI tools (e.g., SHAP, LIME) to better understand model decisions and identify potential biases.

