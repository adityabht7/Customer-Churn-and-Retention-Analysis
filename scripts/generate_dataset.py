"""
Synthetic Benchmark Dataset Generator for Customer Churn & Retention Analysis
Generates 7,043 realistic records mirroring the industry-standard Telco & SaaS
churn distribution with enhanced telemetry (SupportTickets, UsageFrequency, SatisfactionScore).
"""

import numpy as np
import pandas as pd
import os

def generate_churn_dataset(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # Customer IDs
    chars = np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=(n_samples, 4))
    nums = np.random.randint(1000, 9999, size=n_samples)
    customer_ids = [f"{num}-{c1}{c2}{c3}{c4}" for num, (c1, c2, c3, c4) in zip(nums, chars)]
    
    # Demographics
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.505, 0.495])
    senior = np.random.choice([0, 1], size=n_samples, p=[0.838, 0.162])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.483, 0.517])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.299, 0.701])
    
    # Tenure in months (1 - 72) with realistic distribution (bimodal: many new, many long-term)
    tenure_mix = np.random.choice(["new", "mid", "long"], size=n_samples, p=[0.40, 0.30, 0.30])
    tenure = np.zeros(n_samples, dtype=int)
    for i, mix in enumerate(tenure_mix):
        if mix == "new":
            tenure[i] = np.random.randint(1, 15)
        elif mix == "mid":
            tenure[i] = np.random.randint(15, 45)
        else:
            tenure[i] = np.random.randint(45, 73)
            
    # Contract type correlated with tenure
    contracts = []
    for t in tenure:
        if t <= 12:
            c = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.85, 0.12, 0.03])
        elif t <= 36:
            c = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.50, 0.35, 0.15])
        else:
            c = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.25, 0.40, 0.35])
        contracts.append(c)
    contracts = np.array(contracts)
    
    # Phone services
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.46, 0.54]))
            
    # Internet services
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.44, 0.34, 0.22])
    
    # Ancillary services
    def generate_ancillary(service_prob: float):
        result = []
        for is_val in internet_service:
            if is_val == "No":
                result.append("No internet service")
            else:
                result.append("Yes" if np.random.rand() < service_prob else "No")
        return result
        
    online_security = generate_ancillary(0.38)
    online_backup = generate_ancillary(0.44)
    device_protection = generate_ancillary(0.44)
    tech_support = generate_ancillary(0.39)
    streaming_tv = generate_ancillary(0.49)
    streaming_movies = generate_ancillary(0.50)
    
    # Billing & Payment
    paperless = np.random.choice(["Yes", "No"], size=n_samples, p=[0.592, 0.408])
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.336, 0.228, 0.219, 0.217]
    )
    
    # Monthly Charges based on services
    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if phone_service[i] == "Yes":
            base += 10.0
            if multiple_lines[i] == "Yes":
                base += 10.0
        if internet_service[i] == "DSL":
            base += 25.0
        elif internet_service[i] == "Fiber optic":
            base += 50.0
            
        if online_security[i] == "Yes": base += 5.0
        if online_backup[i] == "Yes": base += 5.0
        if device_protection[i] == "Yes": base += 5.0
        if tech_support[i] == "Yes": base += 5.0
        if streaming_tv[i] == "Yes": base += 10.0
        if streaming_movies[i] == "Yes": base += 10.0
        
        noise = np.random.normal(0, 2.5)
        charge = max(18.5, min(120.0, round(base + noise, 2)))
        monthly_charges.append(charge)
        
    monthly_charges = np.array(monthly_charges)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Enhanced Telemetry: Support Tickets, Usage Frequency, CSAT Score
    support_tickets = []
    usage_freq = []
    satisfaction_score = []
    
    for i in range(n_samples):
        # Higher tickets for customers without TechSupport or on Fiber optic
        mean_tickets = 1.2
        if tech_support[i] == "No" and internet_service[i] != "No":
            mean_tickets += 1.5
        if payment_methods[i] == "Electronic check":
            mean_tickets += 0.5
        t_count = min(6, int(np.random.poisson(mean_tickets)))
        support_tickets.append(t_count)
        
        # Satisfaction Score (1 to 5)
        csat = 5 - min(4, int(np.random.poisson(t_count * 0.7)))
        if contracts[i] == "Month-to-month" and t_count > 2:
            csat = max(1, csat - 1)
        satisfaction_score.append(csat)
        
        # Usage Frequency
        if streaming_tv[i] == "Yes" or streaming_movies[i] == "Yes":
            u = np.random.choice(["Daily", "Weekly", "Bi-weekly", "Monthly"], p=[0.60, 0.25, 0.10, 0.05])
        else:
            u = np.random.choice(["Daily", "Weekly", "Bi-weekly", "Monthly"], p=[0.35, 0.35, 0.20, 0.10])
        usage_freq.append(u)
        
    # Realistic Churn Probability Model (Logit)
    churn_log_odds = []
    for i in range(n_samples):
        score = -2.6  # baseline intercept calibrated for realistic ~26.5% churn
        
        # Contract effect (strongest driver)
        if contracts[i] == "Month-to-month":
            score += 1.25
        elif contracts[i] == "Two year":
            score -= 1.6
        elif contracts[i] == "One year":
            score -= 0.7
            
        # Tenure effect
        if tenure[i] < 6:
            score += 0.95
        elif tenure[i] < 15:
            score += 0.45
        elif tenure[i] > 48:
            score -= 1.1
            
        # Internet & Tech Support
        if internet_service[i] == "Fiber optic":
            score += 0.55
        if tech_support[i] == "No" and internet_service[i] != "No":
            score += 0.45
        if online_security[i] == "No" and internet_service[i] != "No":
            score += 0.35
            
        # Payment method
        if payment_methods[i] == "Electronic check":
            score += 0.50
        elif "automatic" in payment_methods[i]:
            score -= 0.45
            
        # Senior citizen
        if senior[i] == 1:
            score += 0.20
            
        # Monthly Charges
        if monthly_charges[i] > 85:
            score += 0.40
            
        # Support tickets & CSAT
        score += (support_tickets[i] * 0.30)
        score -= ((satisfaction_score[i] - 3) * 0.35)
        
        # Random shock
        score += np.random.normal(0, 0.4)
        churn_log_odds.append(score)
        
    probs = 1 / (1 + np.exp(-np.array(churn_log_odds)))
    churn_labels = ["Yes" if p > 0.50 else "No" for p in probs]
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportTickets": support_tickets,
        "UsageFrequency": usage_freq,
        "SatisfactionScore": satisfaction_score,
        "Churn": churn_labels
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    out_path = "data/telco_customer_churn.csv"
    print("Generating realistic customer churn dataset...")
    df = generate_churn_dataset(n_samples=7043, random_state=42)
    df.to_csv(out_path, index=False)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    print(f"Dataset generated successfully at '{out_path}'.")
    print(f"Total Customers: {len(df):,}")
    print(f"Overall Churn Rate: {churn_rate:.2f}%")
    print(f"Churn Breakdown: \n{df['Churn'].value_counts()}")
