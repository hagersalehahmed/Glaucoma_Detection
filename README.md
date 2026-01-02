# Disease_prediction



### Dataset
 This study utilized UCI's machine learning repository to aggregate the benchmark Chronic Kidney Disease Dataset https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease. The dataset includes 400 cases of CKD, 150 of which are negative and 250 of which are positive. The dataset is composed of 24 features that are categorized into 13 categorical features and 11 numeric features, with one class label having two values: 1 and 0. 

### Dataset
 This study utilized UCI's machine learning repository to aggregate the benchmark Chronic Kidney Disease Dataset \cite{dataset}. The dataset includes 400 cases of CKD, 150 of which are negative and 250 of which are positive. The dataset is composed of 24 features that are categorized into 13 categorical features and 11 numeric features, with one class label having two values: 1 and 0. 

 ### Featur selection methods
 1. Particle swarm optimization (PSO) 
 2. Genetic algorithm (GA) 

 ### Baseline ML models
1. Support vector machines (SVM) 
2. Decision tree (DT) 
3. Random Forest (RF) 
4. Naive Bayes (NB)
5. Logistic regression (LR)
 ### Stacking model

A stacking model, also known as stacked generalization or simply "stacking," is an ensemble machine learning technique that combines multiple models (often referred to as "base models" or "level-0 models") to improve the overall predictive performance. The key idea is to leverage the strengths of different models and reduce their weaknesses by stacking them together.

#### How Stacking Works
1. Base Models (Level-0 Models):

Multiple diverse models (e.g., decision trees, logistic regression, SVMs, etc.) are trained on the same dataset.
Each base model makes predictions, which are then used as input features for the next stage, the meta-model.
Meta-Model (Level-1 Model):

2. The meta-model, also called the "blender" or "stacker," is trained on the predictions made by the base models.

The purpose of the meta-model is to learn how to best combine the base models' predictions to improve the final output.
#### Steps

1. Step 1: Train Base Models

Train several models on the training dataset, e.g., a Random Forest, a Support Vector Machine (SVM), and .
2. Step 2: Generate Predictions

For each base model, generate predictions for both the training data and the test data. These predictions become the new features.
3. Step 3: Train the Meta-Model

Using the predictions from the base models as input features, train a meta-model (e.g., a linear regression or another more complex model) to make the final predictions.
4. Step 4: Make Final Predictions

The meta-model takes the predictions from the base models and outputs the final prediction.
#### Why Use Stacking?
Improved Accuracy: By combining different models, stacking often achieves better performance than any single model.
Diverse Models: It leverages the strengths of various models. For example, one model might handle outliers well, while another might capture the overall trend better.
Flexibility: You can use different types of models as base models and a different type as the meta-model.

 ### Evaluation models
 Classification performance is  measured using precision, recall, f-measure, and accuracy metrics

 ### Explainable Artificial Intelligence  (XAI)
XAI refers to a set of methods and techniques in artificial intelligence (AI) that aim to make the decision-making processes of AI systems transparent, understandable, and interpretable by humans.

Tools:
1. LIME (Local Interpretable Model-Agnostic Explanations): Explains individual predictions by approximating the black-box model with a local, interpretable model.
2. SHAP (SHapley Additive exPlanations): Provides a unified measure of feature importance based on cooperative game theory.
