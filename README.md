# Predictive Modeling for Customer Subscription in Banking Marketing Campaigns: Identifying Key Factors Using Machine Learning

## Project Overview
This project focuses on predicting customer subscription to term deposits in direct marketing campaigns conducted by a European bank. The primary goal is to identify key factors that influence customer behavior and build predictive models using machine learning techniques to optimize marketing strategies.

### Data Source:
- **Dataset**: Bank Marketing Dataset from direct marketing campaigns (public dataset).
- **Features**: 16 attributes including demographic and economic features (e.g., age, job, education, balance, loan status) as well as campaign-related features (e.g., contact type, campaign duration, number of contacts, and previous campaign outcomes).
- **Target**: Binary classification – whether the customer subscribes to a term deposit (`subscribe`).

## Objectives:
1. **Exploratory Data Analysis**: Understanding the key trends and patterns in the data.
2. **Feature Engineering**: Applying preprocessing techniques, such as encoding categorical variables, handling class imbalance, and feature selection.
3. **Modeling**: Developing and evaluating multiple machine learning models to predict customer subscription.
4. **Evaluation**: Comparing models using evaluation metrics like accuracy, precision, recall, F1 score, and AUC.

## Methodology

### 1. Data Preparation:
- **Handling Missing Values**: Rows with missing values (< 1% of total) were removed to maintain data integrity.
- **Categorical Encoding**: Categorical variables were encoded using One-Hot Encoding, and the 'unknown' category was handled separately.
- **Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the target variable distribution in the training set.
- **Train-Test Split**: Data was split into 80% training and 20% testing sets, ensuring stratified sampling to preserve class distribution.

### 2. Feature Selection:
- **Correlation Analysis**: A correlation matrix was used to identify key variables with significant impact on customer subscription.
- **Feature Importance**: Feature importance scores from tree-based models (Random Forest, XGBoost) were used to highlight the most relevant factors affecting customer behavior.

### 3. Model Building:
The following machine learning models were developed using Python:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **CatBoost**
- **LightGBM**

Each model underwent **hyperparameter tuning** using GridSearchCV to optimize performance.

### 4. Evaluation Metrics:
Models were evaluated based on:
- **Accuracy**: Overall correctness of the predictions.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Ability to capture true positives out of all actual positives.
- **F1 Score**: Harmonic mean of precision and recall, prioritizing balance between them.
- **AUC**: Area Under the ROC Curve, which assesses model performance across various thresholds.

## Results

### Model Comparison:


| Model              | Best Parameters                                 | Training Accuracy | Testing Accuracy | Training Optimal Threshold | Testing Optimal Threshold | Training Precision | Testing Precision | Training Recall | Testing Recall | Training F1 Score | Testing F1 Score |
|--------------------|------------------------------------------------|-------------------|------------------|---------------------------|--------------------------|--------------------|-------------------|------------------|-----------------|--------------------|-------------------|
| **Logistic Regression** | `{'model__C': 0.1, 'model__solver': 'liblinear'}` | 0.7173            | 0.9123          | 0.5613                    | 0.8950                   | 0.6250             | 0.6667            | 0.2206           | 0.2373         | 0.3261             | 0.3500            |
| **Random Forest**   | `{'model__max_depth': 20, 'model__min_samples_split': 7, 'model__n_estimators': 50}` | 0.8633            | 0.9123          | 0.5615                    | 0.6946                   | 0.5223             | 0.5854            | 0.6029           | 0.4068         | 0.5597             | 0.4800            |
| **XGBoost**         | `{'model__learning_rate': 0.2, 'model__max_depth': 7, 'model__n_estimators': 500}` | 0.8691            | 0.9056          | 0.6762                    | 0.7101                   | 0.7263             | 0.5366            | 0.5074           | 0.3729         | 0.5974             | 0.4400            |
| **CatBoost**        | `{'model__depth': 10, 'model__iterations': 500, 'model__learning_rate': 0.2}` | 0.8720            | 0.9106          | 0.6684                    | 0.7529                   | 0.7447             | 0.5750            | 0.5147           | 0.3898         | 0.6087             | 0.4646            |
| **LightGBM**        | `{'model__learning_rate': 0.1, 'model__max_depth': 15, 'model__n_estimators': 200}` | 0.8720            | 0.9073          | 0.7555                    | 0.7555                   | 0.7500             | 0.5500            | 0.4853           | 0.3729         | 0.5893             | 0.4444            |

### Summary of Findings

1. **Training Accuracy:**
   - **LightGBM** exhibits the highest training accuracy (**87.20%**), closely followed by **CatBoost** (**87.20%**) and **XGBoost** (**86.91%**). **Random Forest** shows a training accuracy of **86.33%**, while **Logistic Regression** lags significantly at **71.73%**, indicating its lower capacity to capture complex relationships in the data.

2. **Testing Accuracy:**
   - All models achieve comparable performance on the testing set, with **Logistic Regression** and **Random Forest** both demonstrating impressive testing accuracy at **91.23%**. **CatBoost** and **LightGBM** show slightly lower accuracies (**91.06%** and **90.73%**, respectively), hinting at potential overfitting.

3. **Precision, Recall, and F1 Score:**
   - **Logistic Regression** leads with the highest testing precision (**66.67%**), indicating its reliability in positive class predictions. However, its low recall (**23.73%**) suggests it misses many actual positive instances.
   - **Random Forest** strikes a better balance, with a testing precision of **58.54%** and a recall of **40.68%**, resulting in the highest F1 score on the test set (**48.00%**).
   - **XGBoost**, **CatBoost**, and **LightGBM** perform similarly, exhibiting lower testing precision and recall compared to Random Forest.

### Recommendations

**Best Model:** Based on the evaluation metrics, **Random Forest** emerges as the recommended model. Its balanced performance across accuracy, precision, and recall positions it as a robust choice for this classification task.

### Advantages of Random Forest:
- **Robustness to Overfitting:** It effectively mitigates overfitting due to its ensemble nature.
- **Feature Importance:** Offers insights into feature importance, aiding in model interpretation.
- **Versatility:** Capable of handling both numerical and categorical data without extensive preprocessing.

### Disadvantages of Random Forest:
- **Interpretability:** More complex than Logistic Regression, making interpretation challenging.
- **Prediction Speed:** Generally slower in making predictions compared to simpler models.

  ### Insights on Class Imbalance

A crucial observation from the results is the insufficient representation of class 1 in the dataset. With only 136 instances of class 1 compared to 1247 instances of class 0, the imbalance negatively impacts model performance. Despite employing various strategies like SMOTE, hyperparameter tuning, and model selection, the model struggles to achieve a satisfactory recall and F1 score for class 1. This indicates that the model may have difficulty generalizing from such a limited amount of data.

To further improve the F1 score for class 1, consider the following approaches:

**1. Data Augmentation** - Explore techniques to synthesize new samples for the minority class beyond SMOTE, such as data augmentation techniques tailored to your specific features.

**2. Collect More Data** - If feasible, obtaining additional labeled data for class 1 could significantly enhance model training and evaluation.

**3. Alternative Evaluation Metrics** - While focusing on the F1 score is vital, it may also be beneficial to assess other metrics, such as the area under the precision-recall curve (AUC-PR), which can provide additional insights into the model’s performance with respect to the minority class.

### Conclusion
In summary, while the Random Forest model demonstrates solid performance overall, the insufficiency of data for class 1 presents a challenge that hinders the achievement of a higher F1 score. Continued focus on handling class imbalance and exploring innovative data augmentation methods may yield better results in future iterations.

## Business Impact:
The project provides actionable insights that can help banks refine their marketing strategies by focusing on customers more inclined to subscribe, reducing wasted efforts, and improving campaign ROI. Furthermore, the machine learning models can be integrated into a decision-support system for real-time campaign optimization.


