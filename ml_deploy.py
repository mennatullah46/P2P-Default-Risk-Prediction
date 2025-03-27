# from flask import Flask, request, jsonify, render_template
# import joblib
# import numpy as np
# import os

# # Ensure the template folder path is correctly handled on Windows
# template_folder_path = os.path.join(os.getcwd(), 'templates')

# app = Flask(__name__, template_folder=template_folder_path)

# # Load model and scaler
# try:
#     model = joblib.load('logistic_regression.pkl')
#     scaler = joblib.load('scaler.pkl')
# except Exception as e:
#     print(f"Error loading model or preprocessing components: {e}")
#     exit(1)

# # List of used features (from your provided list)
# used_features = [
#     'LPCustomerPrincipalPayments', 'ClosedStatusFlag', 'LPCustomerPayments', 'BorrowerAPR',
#     'LoanNumber', 'ListingNumber', 'LoanMonthsSinceOrigination', 'EstimatedEffectiveYield',
#     'LoanOriginationYear', 'EstimatedReturn', 'EstimatedLoss', 'BorrowerRate', 'LenderYield',
#     'MonthlyLoanPayment', 'LoanCurrentDaysDelinquent', 'Investors', 'LPInterestandFees',
#     'LPServiceFees', 'LoanOriginalAmount', 'ListingCategory(numeric)', 'Term', 'ProsperScore',
#     'ProsperRating(numeric)', 'TimeSinceCreditPulled', 'LPNetPrincipalLoss', 'DaysToOrigination',
#     'LPGrossPrincipalLoss', 'LoanFirstDefaultedCycleNumber', 'RevolvingCreditBalance',
#     'OpenRevolvingMonthlyPayment', 'EmploymentStatusDuration', 'CreditAgeAtListing',
#     'BankcardUtilization', 'CreditHistoryLength', 'CreditScoreRangeUpper', 'CreditScoreRangeLower',
#     'AvailableBankcardCredit', 'OpenCreditLines', 'TotalTrades', 'DebtToIncomeRatio',
#     'TradesNeverDelinquent(percentage)', 'CurrentCreditLines'
# ]

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html', used_features=used_features)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()

#         # Check if all required features are present
#         missing_features = [feature for feature in used_features if feature not in data]
#         if missing_features:
#             return jsonify({'error': f"Missing features: {', '.join(missing_features)}"}), 400

#         # Extract feature values dynamically
#         features = [data[feature] for feature in used_features]

#         features_np = np.array([features]).reshape(1, -1)
#         features_scaled = scaler.transform(features_np)
#         prediction = model.predict(features_scaled)[0]

#         return jsonify({'prediction': int(prediction)})

#     except KeyError as e:
#         return jsonify({'error': f"Missing feature: {e}"}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))  # Get the port from environment variable or default to 5000
#     app.run(host='0.0.0.0', port=port, debug=True)



import streamlit as st
import joblib
import numpy as np

# Load model and scaler
try:
    model = joblib.load('logistic_regression.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or preprocessing components: {e}")
    st.stop()

# Features that require scaling (41 scaled features)
scaled_features = [
    'LPCustomerPrincipalPayments', 'LPCustomerPayments', 'BorrowerAPR', 
    'LoanMonthsSinceOrigination', 'EstimatedEffectiveYield', 'EstimatedReturn', 
    'EstimatedLoss', 'BorrowerRate', 'LenderYield', 'MonthlyLoanPayment', 'LPInterestandFees', 
    'LoanCurrentDaysDelinquent', 'Investors', 'LPServiceFees', 'LoanOriginalAmount',
    'Term', 'DaysToOrigination', 'TimeSinceCreditPulled', 'LoanFirstDefaultedCycleNumber', 
    'LPNetPrincipalLoss', 'LPGrossPrincipalLoss', 'RevolvingCreditBalance', 'EmploymentStatusDuration', 
    'OpenRevolvingMonthlyPayment', 'CreditScoreRangeLower', 'BankcardUtilization', 
    'CreditHistoryLength', 'AvailableBankcardCredit', 'CreditScoreRangeUpper', 
    'CreditAgeAtListing', 'OnTimePaymentRatio', 'ProsperPrincipalBorrowed', 'OpenCreditLines', 
    'DebtToIncomeRatio', 'TotalTrades', 'TradesNeverDelinquent(percentage)', 
    'TotalProsperLoans', 'PercentFunded', 'ProsperPrincipalOutstanding', 
    'ScorexChangeAtTimeOfListing', 'TotalProsperPaymentsBilled'
]

# All features (46 original features)
all_features = [
    'LPCustomerPrincipalPayments', 'ClosedStatusFlag', 'LPCustomerPayments', 'BorrowerAPR', 
    'LoanMonthsSinceOrigination', 'EstimatedEffectiveYield', 'LoanOriginationYear', 'EstimatedReturn', 
    'EstimatedLoss', 'BorrowerRate', 'LenderYield', 'MonthlyLoanPayment', 'LPInterestandFees', 
    'LoanCurrentDaysDelinquent', 'Investors', 'LPServiceFees', 'LoanOriginalAmount', 
    'ListingCategory(numeric)', 'Term', 'ProsperScore', 'DaysToOrigination', 'ProsperRating(numeric)', 
    'TimeSinceCreditPulled', 'LoanFirstDefaultedCycleNumber', 'LPNetPrincipalLoss', 'LPGrossPrincipalLoss', 
    'RevolvingCreditBalance', 'EmploymentStatusDuration', 'OpenRevolvingMonthlyPayment', 
    'CreditScoreRangeLower', 'BankcardUtilization', 'CreditHistoryLength', 'AvailableBankcardCredit', 
    'CreditScoreRangeUpper', 'CreditAgeAtListing', 'OnTimePaymentRatio', 'ProsperPrincipalBorrowed', 
    'OpenCreditLines', 'DebtToIncomeRatio', 'TotalTrades', 'TradesNeverDelinquent(percentage)', 
    'TotalProsperLoans', 'PercentFunded', 'ProsperPrincipalOutstanding', 'ScorexChangeAtTimeOfListing', 
    'TotalProsperPaymentsBilled'
]

# Streamlit UI
st.title("Loan Default Prediction")
st.write("Enter the required features to predict loan default.")

# User input form
input_data = {feature: st.number_input(f"{feature}", value=0.0) for feature in all_features}

# Predict button
if st.button("Predict"):
    try:
        # Extract values for only the scaled features
        scaled_input_values = np.array([[input_data[feature] for feature in scaled_features]])

        # Apply scaling
        scaled_values = scaler.transform(scaled_input_values)[0]

        # Reconstruct the full feature vector
        final_input = []
        for feature in all_features:
            if feature in scaled_features:
                final_input.append(scaled_values[scaled_features.index(feature)])  # Scaled value
            else:
                final_input.append(input_data[feature])  # Non-scaled value

        # Convert to NumPy array
        final_input = np.array([final_input])

        # Make prediction
        prediction = model.predict(final_input)[0]

        st.success(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")
    except Exception as e:
        st.error(f"Error: {e}")


