import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
file_path = "/mnt/data/Housing.csv"
df = pd.read_csv(file_path)

# Convert categorical variables using One-Hot Encoding
categorical_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", 
                       "airconditioning", "prefarea", "furnishingstatus"]
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Split data into features and target
X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multiple Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear_model.predict(X_test)

# Evaluate the model
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print(f"Mean Absolute Error: {mae_linear}")
