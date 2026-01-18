import pandas as pd
import numpy as np
import os

def generate_leads_data(n=1500):
    np.random.seed(42)
    
    data = {
        'Lead_ID': [f'LEAD_{i:04d}' for i in range(1, n + 1)],
        'Lead_Source': np.random.choice(['Organic Search', 'Referral', 'Social Media', 'Google Ads', 'Direct Traffic', 'API'], n, p=[0.25, 0.15, 0.20, 0.20, 0.15, 0.05]),
        'Industry': np.random.choice(['Technology', 'Education', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'], n),
        'Company_Size': np.random.choice(['Small', 'Medium', 'Enterprise'], n, p=[0.5, 0.3, 0.2]),
        'Total_Web_Visits': np.random.poisson(lam=5, size=n),
        'Avg_Time_Per_Visit': np.random.uniform(1, 15, size=n),
        'Page_Views_Per_Visit': np.random.uniform(1, 8, size=n),
        'Email_Opens': np.random.poisson(lam=2, size=n),
        'Email_Clicks': np.random.poisson(lam=0.5, size=n),
        'Last_Activity': np.random.choice(['Email Opened', 'Webinar Attended', 'Page Visited', 'Form Submitted', 'Chat Conversation'], n),
    }

    df = pd.DataFrame(data)

    # Logic to decide "Converted" based on behavior
    # Engagement score = visits * time + clicks * 5
    engagement_score = (df['Total_Web_Visits'] * df['Avg_Time_Per_Visit']) + (df['Email_Clicks'] * 10) + (df['Email_Opens'] * 2)
    
    # Baseline conversion chance
    conversion_prob = engagement_score / engagement_score.max()
    
    # Add some noise
    noise = np.random.normal(0, 0.1, size=n)
    final_prob = conversion_prob + noise
    
    df['Converted'] = (final_prob > 0.5).astype(int)

    # Save to CSV
    output_path = os.path.join(os.getcwd(), 'leads.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {n} records at: {output_path}")
    return df

if __name__ == "__main__":
    generate_leads_data()
