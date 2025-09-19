# Customer Churn Prediction with K-Nearest Neighbors

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

print("Step 1: Libraries Imported.")

# Step 2: Load the Dataset
# The Telco Customer Churn dataset is now loaded from the local file provided.
try:
    df = pd.read_csv('Telco_customer_churn.csv')
    print("Step 2: Dataset loaded successfully from local file.")
except FileNotFoundError:
    print("Error: The file 'Telco_customer_churn.csv' was not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()

# Step 3: Data Preprocessing and Feature Engineering
print("Step 3: Starting data preprocessing.")

# Handle 'TotalCharges' column, which is initially an object type due to some empty strings.
# The column name from the CSV file is "Total Charges" with a space.
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(inplace=True)

# Convert the target variable 'Churn Label' to a numerical value (1 for 'Yes', 0 for 'No').
# The target column from the CSV is 'Churn Label'.
df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

# Separate features (X) and target (y)
X = df.drop(['Churn Label', 'Total Charges'], axis=1) # Exclude both the target and the one-hot encoded version
y = df['Churn Label']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines for numerical and categorical data
numerical_transformer = StandardScaler() # Scaling is crucial for KNN
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Data preprocessing steps defined.")

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Step 4: Data split into training and testing sets.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Step 5: Hyperparameter Tuning with GridSearchCV
print("\nStep 5: Starting hyperparameter tuning for the KNN model...")
# Define the model within a pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', KNeighborsClassifier())])

# Define the parameter grid for KNN
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],
    'classifier__p': [1, 2] # p=1 for Manhattan distance, p=2 for Euclidean distance
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Hyperparameter tuning complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation F1-Score: {grid_search.best_score_:.4f}")

# Step 6: Train the final model with the best parameters
print("Step 6: Training the final model with best parameters.")
# Use the best_estimator_ from the GridSearchCV object, which is already trained
final_model = grid_search.best_estimator_

# Step 7: Make predictions and evaluate the model
print("\nStep 7: Evaluating the model on the test set.")
y_pred = final_model.predict(X_test)

# Check if the classifier has multiple classes before calling predict_proba
if len(final_model.named_steps['classifier'].classes_) == 2:
    y_proba = final_model.predict_proba(X_test)[:, 1]
    can_plot_roc = True
else:
    print("\nWarning: The classifier only learned a single class. The ROC curve cannot be plotted.")
    y_proba = np.zeros(len(y_test)) # Dummy array to prevent plotting errors
    can_plot_roc = False

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.4f}")

# Step 8: Visualization
print("\nStep 8: Generating visualizations.")

# Pair Plot for a subset of features to visualize relationships
print("Generating Pair Plot... (may take a moment)")
df_subset = df[['Monthly Charges', 'Total Charges', 'Tenure Months', 'Churn Label']].copy()
sns.pairplot(df_subset, hue='Churn Label', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Key Features by Churn Status', y=1.02)
plt.savefig('plot.png')
plt.show()

# ROC Curve and AUC - only if we have two classes
if can_plot_roc:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='KNN Classifier')
    roc_display.plot()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot([0, 1], [0, 1], 'k--') # Add diagonal line for random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
