import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os

class CreditDataVisualizer:
    """Generate interactive visualizations for consumer credit data analysis."""
    
    def __init__(self, output_dir='reports/visualizations'):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_risk_distribution_chart(self, df):
        """Create visualization of risk distribution."""
        risk_dist = df.groupby('risk_category').size().reset_index(name='count')
        
        fig = go.Figure(data=[
            go.Pie(labels=risk_dist['risk_category'],
                  values=risk_dist['count'],
                  hole=0.3,
                  marker_colors=['#2ecc71', '#f1c40f', '#e74c3c'])
        ])
        
        fig.update_layout(
            title="Loan Risk Distribution",
            annotations=[dict(text='Risk Profile', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        fig.write_html(os.path.join(self.output_dir, 'risk_distribution.html'))
        return fig

    def create_loan_amount_heatmap(self, df):
        """Generate heatmap of loan amounts by risk category and purpose."""
        pivot_table = df.pivot_table(
            values='loan_amount',
            index='risk_category',
            columns='loan_purpose',
            aggfunc='mean'
        ).round(2)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            text=pivot_table.values.round(0),
            texttemplate='£%{text:,.0f}',
            textfont={"size": 10},
            textcolor='white'
        ))
        
        fig.update_layout(
            title='Average Loan Amount by Risk Category and Purpose',
            xaxis_title='Loan Purpose',
            yaxis_title='Risk Category'
        )
        
        fig.write_html(os.path.join(self.output_dir, 'loan_amount_heatmap.html'))
        return fig

    def plot_time_series_metrics(self, df):
        """Create time series visualization of key metrics."""
        df['month'] = pd.to_datetime(df['start_date']).dt.to_period('M')
        monthly_metrics = df.groupby('month').agg({
            'loan_amount': 'sum',
            'risk_score': 'mean',
            'loan_duration_days': 'mean'
        }).reset_index()
        
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Total Loan Amount', 
                                         'Average Risk Score',
                                         'Average Loan Duration'))
        
        # Total Loan Amount
        fig.add_trace(
            go.Scatter(x=monthly_metrics['month'].astype(str),
                      y=monthly_metrics['loan_amount'],
                      mode='lines+markers',
                      name='Total Loans'),
            row=1, col=1
        )
        
        # Average Risk Score
        fig.add_trace(
            go.Scatter(x=monthly_metrics['month'].astype(str),
                      y=monthly_metrics['risk_score'],
                      mode='lines+markers',
                      name='Risk Score'),
            row=2, col=1
        )
        
        # Average Duration
        fig.add_trace(
            go.Scatter(x=monthly_metrics['month'].astype(str),
                      y=monthly_metrics['loan_duration_days'],
                      mode='lines+markers',
                      name='Duration'),
            row=3, col=1
        )
        
        fig.update_layout(height=900, title_text="Key Metrics Over Time")
        fig.write_html(os.path.join(self.output_dir, 'time_series_metrics.html'))
        return fig

    def create_correlation_matrix(self, df):
        """Generate correlation matrix visualization."""
        numeric_cols = ['loan_amount', 'interest_rate', 'credit_score',
                       'debt_to_income_ratio', 'risk_score', 'loan_duration_days']
        correlation_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=numeric_cols,
            y=numeric_cols,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=correlation_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Correlation Matrix of Numerical Variables',
            width=800,
            height=800
        )
        
        fig.write_html(os.path.join(self.output_dir, 'correlation_matrix.html'))
        return fig

    def generate_dashboard(self, df):
        """Generate complete dashboard with all visualizations."""
        # Create all individual visualizations
        figs = {
            'risk_dist': self.generate_risk_distribution_chart(df),
            'heatmap': self.create_loan_amount_heatmap(df),
            'time_series': self.plot_time_series_metrics(df),
            'correlation': self.create_correlation_matrix(df)
        }
        
        # Create dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Consumer Credit Analysis Dashboard</title>
            <style>
                .dashboard-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .visualization-container {{
                    margin-bottom: 30px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <h1>Consumer Credit Analysis Dashboard</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="visualization-container">
                    {figs['risk_dist'].to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                
                <div class="visualization-container">
                    {figs['heatmap'].to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                
                <div class="visualization-container">
                    {figs['time_series'].to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                
                <div class="visualization-container">
                    {figs['correlation'].to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(self.output_dir, 'dashboard.html'), 'w') as f:
            f.write(dashboard_html)

def generate_visualizations(df):
    """Helper function to generate all visualizations for a dataset."""
    visualizer = CreditDataVisualizer()
    visualizer.generate_dashboard(df)
    return visualizer
