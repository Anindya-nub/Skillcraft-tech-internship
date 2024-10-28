# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the datasets
train_data = pd.read_csv('Task1/train.csv')
test_data = pd.read_csv('Task1/test.csv')

# Feature selection: Selecting a subset of features relevant to house pricing.
# Adjust these columns based on domain knowledge
selected_features = [
    'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea',
    'GarageCars', 'GarageArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'YearRemodAdd'
]

# Extracting features (X) and target (y) from the training data
X_train = train_data[selected_features]
y_train = train_data['SalePrice']

# Handling missing values (using mean imputation for simplicity)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# For the test data, ensure you also have the 'ID' column and the same features
X_test = test_data[selected_features]
X_test_imputed = imputer.transform(X_test)
test_ids = test_data['Id']  # Preserve 'Id' for the final submission

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train_imputed, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_imputed)

# Create a DataFrame for submission with 'Id' and 'SalePrice'
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred
})

# Save the submission file
submission.to_csv('Task1/house_price_submission.csv', index=False)

# Output the submission file to check
print(submission.head())
