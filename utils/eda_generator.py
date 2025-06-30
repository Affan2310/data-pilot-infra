import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Optional

class EDAGenerator:
    """Generates interactive exploratory data analysis visualizations."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create distribution plot for a numeric column."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'Histogram of {column}',
                    f'Box Plot of {column}',
                    f'Q-Q Plot of {column}',
                    f'Statistics Summary'
                ),
                specs=[[{"type": "histogram"}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=df[column], name="Distribution", nbinsx=30),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=df[column], name="Box Plot"),
                row=1, col=2
            )
            
            # Q-Q plot (approximate)
            from scipy import stats
            sorted_data = np.sort(df[column].dropna())
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_data,
                    mode='markers',
                    name='Q-Q Plot'
                ),
                row=2, col=1
            )
            
            # Statistics table
            stats_data = df[column].describe()
            fig.add_trace(
                go.Table(
                    header=dict(values=['Statistic', 'Value']),
                    cells=dict(values=[
                        list(stats_data.index),
                        [f"{val:.3f}" if isinstance(val, (int, float)) else str(val) 
                         for val in stats_data.values]
                    ])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text=f"Distribution Analysis: {column}",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            # Fallback to simple histogram
            fig = px.histogram(df, x=column, title=f"Distribution of {column}")
            return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create correlation heatmap for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def create_missing_values_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing missing values pattern."""
        # Create binary matrix where 1 = missing, 0 = present
        missing_matrix = df.isnull().astype(int)
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.values,
            x=missing_matrix.columns,
            y=list(range(len(missing_matrix))),
            colorscale=[[0, 'lightblue'], [1, 'red']],
            showscale=True,
            colorbar=dict(
                title="Missing Values",
                tickvals=[0, 1],
                ticktext=["Present", "Missing"]
            )
        ))
        
        fig.update_layout(
            title="Missing Values Pattern",
            xaxis_title="Columns",
            yaxis_title="Row Index",
            height=600
        )
        
        return fig
    
    def create_feature_distribution_grid(self, df: pd.DataFrame, max_cols: int = 4) -> go.Figure:
        """Create a grid of distribution plots for all numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return None
        
        # Limit number of columns to display
        numeric_cols = numeric_cols[:12]  # Show max 12 features
        
        n_cols = min(max_cols, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            height=200 * n_rows,
            title_text="Feature Distributions Overview",
            showlegend=False
        )
        
        return fig
    
    def create_categorical_analysis(self, df: pd.DataFrame, column: str, max_categories: int = 20) -> go.Figure:
        """Create analysis plots for categorical columns."""
        value_counts = df[column].value_counts().head(max_categories)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Top Categories in {column}', 'Percentage Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name="Count"),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=value_counts.index, values=value_counts.values, name="Distribution"),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text=f"Categorical Analysis: {column}",
            showlegend=False
        )
        
        return fig
    
    def create_outlier_analysis(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create outlier analysis visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Box Plot: {column}',
                f'Scatter Plot with Outliers: {column}',
                'Outlier Statistics',
                'Distribution without Outliers'
            ),
            specs=[[{"type": "box"}, {"type": "scatter"}],
                   [{"type": "table"}, {"type": "histogram"}]]
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df[column], name="With Outliers"),
            row=1, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df[column],
                mode='markers',
                name='Data Points'
            ),
            row=1, col=2
        )
        
        # Calculate outliers
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        # Statistics table
        stats_data = [
            ['Total Data Points', len(df)],
            ['Outliers Detected', outlier_count],
            ['Outlier Percentage', f"{outlier_percentage:.2f}%"],
            ['Lower Bound', f"{lower_bound:.3f}"],
            ['Upper Bound', f"{upper_bound:.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in stats_data], 
                                 [row[1] for row in stats_data]])
            ),
            row=2, col=1
        )
        
        # Distribution without outliers
        clean_data = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        fig.add_trace(
            go.Histogram(x=clean_data[column], name="Without Outliers"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Outlier Analysis: {column}",
            showlegend=False
        )
        
        return fig
