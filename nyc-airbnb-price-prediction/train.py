import wandb
from wandb.sklearn import plot_summary
from preprocessing import load_data, preprocess_data, create_pipeline

# Initialize Weights and Biases project
wandb.init(project="NYC-Airbnb-Price-Prediction", entity="your_wandb_username")

# Load and preprocess the data
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Create the model pipeline
model = create_pipeline()

# Train the model
model.fit(X_train, y_train)

# Evaluate model on test set
test_score = model.score(X_test, y_test)
print(f"Test R^2 Score: {test_score}")

# Log model performance to WandB
wandb.log({"test_score": test_score})

# Log predictions and feature importance
wandb.sklearn.plot_summary(model, X_test, y_test)

