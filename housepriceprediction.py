import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Loading dataset...")

# Load Kaggle dataset (train.csv must be in same folder)
data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Training model...")

# Select only required columns
data = data[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]]
data = data.dropna()

# Define X and y
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training finished!")

# Accuracy
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score:.2f}")

# Sample prediction
prediction = model.predict([[2000, 3, 2]])[0]
print(f"Predicted price for 2000 sqft, 3 bed, 2 bath = ${prediction:,.2f}")
