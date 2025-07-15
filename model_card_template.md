# Model Card: Adult Census Income Prediction Model

## 1. Model Details

* **Model Name:** Adult Census Income Prediction Model
* **Version:** 1.0
* **Developers:** [Joshua Drake]
* **Date:** [Current Date, e.g., July 15, 2025]
* **Model Type:** Classification Model
* **Architecture:** Scikit-learn RandomForestClassifier
    * *Brief Description:* The RandomForestClassifier is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) of the individual trees. It is known for its robustness against overfitting and good performance on various datasets.

## 2. Intended Use

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

## 3. Training Data

* **Dataset:** Adult Census Income dataset (`census.csv`)
* **Source:** Derived from the UCI Adult dataset.
* **Description:** The dataset contains demographic information from the 1994 Census, including features such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country. The target variable is 'salary', indicating whether an individual's income is <=50K or >50K.
* **Number of Samples (Training):** [Number of training samples from your `train` split]
* **Features Used:**
    * `age` (numerical)
    * `workclass` (categorical)
    * `fnlwgt` (numerical)
    * `education` (categorical)
    * `education_num` (numerical)
    * `marital_status` (categorical)
    * `occupation` (categorical)
    * `relationship` (categorical)
    * `race` (categorical)
    * `sex` (categorical)
    * `capital_gain` (numerical)
    * `capital_loss` (numerical)
    * `hours_per_week` (numerical)
    * `native_country` (categorical)
* **Label:** `salary` (binary: `<=50K` or `>50K`)

## 4. Evaluation Data

* **Dataset:** A held-out test set from the Adult Census Income dataset (`census.csv`).
* **Description:** This data was not used during model training and serves as an independent evaluation set.
* **Number of Samples (Test):** [Number of test samples from your `test` split]

## 5. Metrics

The model's performance is evaluated using the following metrics:

* **Precision:** The ratio of true positive predictions to the total positive predictions (true positives + false positives). It measures the accuracy of positive predictions.
    * Formula: $Precision = \frac{TP}{TP + FP}$
* **Recall:** The ratio of true positive predictions to the total actual positives (true positives + false negatives). It measures the model's ability to identify all relevant instances.
    * Formula: $Recall = \frac{TP}{TP + FN}$
* **F-beta Score (with Beta=0.5):** A weighted harmonic mean of precision and recall, where recall is weighted `beta` times more than precision. With $\beta = 0.5$, precision is weighted more heavily than recall. This is useful when false positives are considered more costly than false negatives.
    * Formula: $F_\beta = (1 + \beta^2) \cdot \frac{Precision \cdot Recall}{(\beta^2 \cdot Precision) + Recall}$
    * For $\beta = 0.5$: $F_{0.5} = 1.25 \cdot \frac{Precision \cdot Recall}{(0.25 \cdot Precision) + Recall}$

## 6. Performance

### Overall Model Performance

* **Precision:** [Your overall precision value, e.g., 0.7376]
* **Recall:** [Your overall recall value, e.g., 0.6288]
* **F-beta (Beta=0.5) Score:** [Your overall F1 score value, e.g., 0.6789]

### Performance on Categorical Slices

(This section will contain summarized insights from `slice_output.txt`. You should read the `slice_output.txt` and describe any interesting observations or disparities here. Provide a few examples from the file.)

* **Workclass:**
    * `Federal-gov`: [e.g., Higher precision, good recall]
    * `Private`: [e.g., Balanced performance, representing the majority class]
    * `Never-worked`: [e.g., Often shows perfect scores due to very low sample counts, indicating unreliable metrics for this slice.]
* **Race:**
    * `White`: [e.g., Performance on this majority group often drives overall metrics.]
    * `Black`: [e.g., May show lower recall or precision, indicating potential disparities.]
* **Sex:**
    * `Male`: [e.g., Often higher performance due to dataset biases.]
    * `Female`: [e.g., May show lower performance, highlighting potential gender-based biases.]
* **Other notable slices:**
    * [Mention any other features like 'education' or 'native_country' where you observed significant performance differences or where the sample size was very small, making metrics unreliable.]

**Observations from Slice Performance:**

[Write 2-3 paragraphs here. Discuss patterns you observe in `slice_output.txt`. For instance:]

* "The model generally performs well across most common categories, with `Private` workclass showing robust metrics due to its large representation in the dataset."
* "However, performance on less frequent categories, such as `Never-worked` workclass or certain `native_country` values, often yields perfect (or near-perfect) precision/recall, but these are likely misleading due to very small sample sizes. This indicates that metrics for these rare slices are not statistically significant and the model's generalization to such instances is uncertain."
* "Disparities in performance were observed across certain demographic features like `race` and `sex`. For instance, the model showed [describe specific metric differences, e.g., slightly lower recall for the 'Female' group compared to 'Male'], which could indicate potential biases in the dataset or model. Further investigation and fairness analyses would be required if this model were to be used in a real-world application."

## 7. Limitations and Bias

* **Data Bias:** The training data (1994 Census) reflects societal biases and inequalities of that specific time. The model will learn and perpetuate these biases. For example, historical wage gaps based on gender or race might be reflected in the model's predictions.
* **Lack of Causal Inference:** This model is predictive, not causal. It identifies correlations but does not explain *why* certain features are associated with higher income.
* **Limited Features:** The features available are a snapshot of demographic and employment data. Many other factors (e.g., socioeconomic background, geographic location within the US, specific job roles, soft skills, economic climate) that influence income are not included, limiting the model's completeness.
* **Generalization:** Performance on unseen data might vary, especially for subpopulations that are underrepresented in the training data. The F-beta score prioritizes precision, which might lead to the model being more conservative in predicting '>50K', potentially missing some true positives (lower recall).
* **Fairness:** As observed in the slice performance, there are likely disparities in model performance across different demographic groups. Deploying such a model in sensitive applications without rigorous fairness audits and mitigation strategies could lead to unfair outcomes.

## 8. Ethical Considerations

* **Discrimination:** Using this model in areas like hiring, loan applications, or social benefits could lead to discriminatory outcomes if not carefully monitored and mitigated for bias.
* **Transparency:** While RandomForest is more interpretable than deep learning models, the exact decision-making process for an individual prediction can still be opaque.
* **Privacy:** The dataset contains sensitive personal information, and any deployment must adhere to strict data privacy regulations.

## 9. Future Improvements

* **More Diverse Data:** Incorporate more recent and diverse datasets to reduce historical biases and improve generalization.
* **Feature Engineering:** Create more informative features (e.g., interaction terms, polynomial features) or gather additional relevant data.
* **Hyperparameter Tuning:** Systematically tune the RandomForestClassifier's hyperparameters (e.g., `n_estimators`, `max_depth`, `min_samples_leaf`) for optimal performance.
* **Model Selection:** Experiment with other classification algorithms (e.g., Gradient Boosting Machines like LightGBM/XGBoost, Logistic Regression, Neural Networks).
* **Bias Mitigation Techniques:** Implement techniques like re-weighting training samples, adversarial debiasing, or post-processing predictions to reduce unfairness.
* **Explainable AI (XAI):** Integrate XAI tools (e.g., SHAP, LIME) to better understand model decisions and identify potential biases.
