import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def create_visuals():
    # Load the results
    df = pd.read_csv('data/processed/clustered_data.csv')
    
    # 1. Cluster Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='AnnualIncome', 
        y='SpendingScore', 
        hue='Cluster', 
        palette='bright',
        s=100, 
        edgecolor='black'
    )
    
    # Add descriptions for the clusters (Business Insights)
    plt.title('Customer Segments: Income vs Spending', fontsize=15)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('docs/cluster_chart.png', bbox_inches='tight')
    print("Graph saved to docs/cluster_chart.png")
    plt.show()

if __name__ == "__main__":
    import os
    if not os.path.exists('docs'): os.makedirs('docs')
    create_visuals()