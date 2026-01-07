import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Excel Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .stMetric label {
        color: #1f77b4 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #666666 !important;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
    h3 {
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_excel_file(file):
    """Load Excel file"""
    try:
        sheets = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        return sheets
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def identify_numeric_columns(df):
    """Identify numeric columns"""
    numeric_cols = []
    for col in df.columns:
        try:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_data.notna().sum() / len(df) > 0.5:
                numeric_cols.append(col)
        except:
            pass
    return numeric_cols

def identify_categorical_columns(df):
    """Identify categorical columns"""
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
    return categorical_cols

def get_basic_statistics(df, column):
    """Calculate basic statistics"""
    try:
        data = pd.to_numeric(df[column], errors='coerce').dropna()
        if len(data) == 0:
            return None
        
        stats_dict = {
            'Count': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Q1 (25%)': data.quantile(0.25),
            'Q3 (75%)': data.quantile(0.75),
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis()
        }
        return stats_dict
    except:
        return None

def calculate_correlation_metrics(col1_data, col2_data):
    """Calculate correlation metrics"""
    try:
        col1 = pd.to_numeric(col1_data, errors='coerce')
        col2 = pd.to_numeric(col2_data, errors='coerce')
        
        mask = ~(pd.isna(col1) | pd.isna(col2))
        c1 = col1[mask].values
        c2 = col2[mask].values
        
        if len(c1) < 2:
            return None
        
        metrics = {}
        
        try:
            corr, p_value = stats.pearsonr(c1, c2)
            metrics['Pearson r'] = corr
            metrics['Pearson p-value'] = p_value
        except:
            metrics['Pearson r'] = np.nan
            metrics['Pearson p-value'] = np.nan
        
        try:
            spearman, sp_value = stats.spearmanr(c1, c2)
            metrics['Spearman ρ'] = spearman
            metrics['Spearman p-value'] = sp_value
        except:
            metrics['Spearman ρ'] = np.nan
            metrics['Spearman p-value'] = np.nan
        
        try:
            metrics['R²'] = r2_score(c1, c2)
        except:
            metrics['R²'] = np.nan
        
        try:
            metrics['MAE'] = mean_absolute_error(c1, c2)
            metrics['RMSE'] = np.sqrt(mean_squared_error(c1, c2))
            metrics['MAE/RMSE'] = metrics['MAE'] / metrics['RMSE'] if metrics['RMSE'] != 0 else np.nan
        except:
            metrics['MAE'] = np.nan
            metrics['RMSE'] = np.nan
            metrics['MAE/RMSE'] = np.nan
        
        try:
            dot_product = np.dot(c1, c2)
            norm_c1 = np.linalg.norm(c1)
            norm_c2 = np.linalg.norm(c2)
            metrics['Cosine Similarity'] = dot_product / (norm_c1 * norm_c2) if (norm_c1 * norm_c2) != 0 else np.nan
        except:
            metrics['Cosine Similarity'] = np.nan
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(c1, c2)
            metrics['Slope'] = slope
            metrics['Intercept'] = intercept
            metrics['Std Error'] = std_err
        except:
            metrics['Slope'] = np.nan
            metrics['Intercept'] = np.nan
            metrics['Std Error'] = np.nan
        
        return metrics
    except:
        return None

def create_distribution_plot(df, column):
    """Create histogram"""
    try:
        data = pd.to_numeric(df[column], errors='coerce').dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, name='Distribution', nbinsx=30, opacity=0.7))
        fig.update_layout(
            title=f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Frequency',
            height=400
        )
        return fig
    except:
        return None

def create_scatter_plot(df, col1, col2):
    """Create scatter plot with regression"""
    try:
        data1 = pd.to_numeric(df[col1], errors='coerce')
        data2 = pd.to_numeric(df[col2], errors='coerce')
        
        mask = ~(pd.isna(data1) | pd.isna(data2))
        d1 = data1[mask].values
        d2 = data2[mask].values
        
        if len(d1) < 2:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d1, y=d2, mode='markers', name='Data Points',
            marker=dict(size=8, color='blue', opacity=0.6)
        ))
        
        try:
            slope, intercept, _, _, _ = stats.linregress(d1, d2)
            line_x = np.array([d1.min(), d1.max()])
            line_y = slope * line_x + intercept
            fig.add_trace(go.Scatter(
                x=line_x, y=line_y, mode='lines',
                name=f'Regression (y={slope:.2f}x+{intercept:.2f})',
                line=dict(color='red', width=2)
            ))
        except:
            pass
        
        fig.update_layout(
            title=f'{col1} vs {col2}',
            xaxis_title=col1,
            yaxis_title=col2,
            height=500
        )
        return fig
    except:
        return None

def create_box_plot(df, columns):
    """Create box plot"""
    try:
        fig = go.Figure()
        for col in columns:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            fig.add_trace(go.Box(y=data, name=col))
        
        fig.update_layout(
            title='Box Plot Comparison',
            yaxis_title='Values',
            height=500
        )
        return fig
    except:
        return None

def create_correlation_heatmap(df, columns):
    """Create correlation heatmap"""
    try:
        numeric_data = df[columns].apply(pd.to_numeric, errors='coerce')
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(title='Correlation Heatmap', height=600, width=800)
        return fig
    except:
        return None

def create_time_series_plot(df, date_col, value_cols):
    """Create time series plot"""
    try:
        fig = go.Figure()
        dates = pd.to_datetime(df[date_col], errors='coerce')
        
        for col in value_cols:
            values = pd.to_numeric(df[col], errors='coerce')
            mask = ~(pd.isna(dates) | pd.isna(values))
            fig.add_trace(go.Scatter(x=dates[mask], y=values[mask], mode='lines+markers', name=col))
        
        fig.update_layout(
            title='Time Series Analysis',
            xaxis_title=date_col,
            yaxis_title='Values',
            height=500
        )
        return fig
    except:
        return None

def perform_categorical_analysis(df, column):
    """Analyze categorical column"""
    try:
        value_counts = df[column].value_counts()
        result = {
            'Unique Values': df[column].nunique(),
            'Most Common': value_counts.index[0] if len(value_counts) > 0 else None,
            'Most Common Count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'Least Common': value_counts.index[-1] if len(value_counts) > 0 else None,
            'Least Common Count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
        }
        return result, value_counts
    except:
        return None, None

def create_categorical_plot(value_counts, column_name):
    """Create bar plot for categorical data"""
    try:
        top_counts = value_counts.head(20)
        fig = go.Figure(data=[go.Bar(x=top_counts.index, y=top_counts.values)])
        fig.update_layout(
            title=f'Distribution of {column_name} (Top 20)',
            xaxis_title=column_name,
            yaxis_title='Count',
            height=400
        )
        return fig
    except:
        return None

def main():
    st.title("📊 Universal Excel Analysis Dashboard")
    st.markdown("### Analyze any Excel file with comprehensive statistics and visualizations")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file is None:
        st.info("👈 Please upload an Excel file to begin analysis.")
        st.markdown("""
        ### Features:
        - 📊 **Data Overview**: Structure, missing values, preview
        - 📈 **Numeric Analysis**: Statistics, distributions, comparisons
        - 🔗 **Correlation Analysis**: Pearson, Spearman, R², MAE/RMSE
        - 📋 **Categorical Analysis**: Value counts, distributions
        - 📉 **Visualizations**: Scatter plots, time series, heatmaps
        - 💾 **Export**: Download filtered data and statistics
        """)
        return
    
    with st.spinner("Loading Excel file..."):
        sheets = load_excel_file(uploaded_file)
    
    if sheets is None:
        return
    
    st.sidebar.success("✅ File loaded successfully!")
    
    sheet_name = st.sidebar.selectbox("Select Sheet", list(sheets.keys()))
    df = sheets[sheet_name]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📄 **Sheet:** {sheet_name}\n\n📊 **Rows:** {len(df)}\n\n📋 **Columns:** {len(df.columns)}")
    
    numeric_cols = identify_numeric_columns(df)
    categorical_cols = identify_categorical_columns(df)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Numeric", "🔗 Correlation", "📋 Categorical", "📉 Visualizations", "💾 Export"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Numeric Columns", len(numeric_cols))
        with col4:
            st.metric("Categorical Columns", len(categorical_cols))
        
        st.markdown("---")
        
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.subheader("📊 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        if df.isnull().sum().sum() > 0:
            st.subheader("🔍 Missing Values")
            missing_data = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
            fig = go.Figure(data=[go.Bar(x=missing_data.index, y=missing_data.values)])
            fig.update_layout(title='Missing Values by Column', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Numeric Analysis")
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found.")
        else:
            selected_col = st.selectbox("Select Column", numeric_cols)
            
            if selected_col:
                st.subheader(f"📊 Statistics for {selected_col}")
                stats_dict = get_basic_statistics(df, selected_col)
                
                if stats_dict:
                    cols = st.columns(5)
                    for i, (key, value) in enumerate(stats_dict.items()):
                        with cols[i % 5]:
                            st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
                
                st.subheader("📈 Distribution")
                fig = create_distribution_plot(df, selected_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📊 Compare Multiple Columns")
                selected_cols = st.multiselect(
                    "Select columns",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if len(selected_cols) > 0:
                    fig = create_box_plot(df, selected_cols)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    comparison_data = []
                    for col in selected_cols:
                        stats = get_basic_statistics(df, col)
                        if stats:
                            stats['Column'] = col
                            comparison_data.append(stats)
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df = comparison_df[['Column'] + [c for c in comparison_df.columns if c != 'Column']]
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("Correlation Analysis")
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            st.subheader("🔥 Correlation Heatmap")
            selected_heatmap = st.multiselect(
                "Select columns",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )
            
            if len(selected_heatmap) >= 2:
                fig = create_correlation_heatmap(df, selected_heatmap)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔗 Pairwise Correlation")
            
            col1, col2 = st.columns(2)
            with col1:
                corr_col1 = st.selectbox("First Column", numeric_cols, key='c1')
            with col2:
                corr_col2 = st.selectbox("Second Column", [c for c in numeric_cols if c != corr_col1], key='c2')
            
            if corr_col1 and corr_col2:
                metrics = calculate_correlation_metrics(df[corr_col1], df[corr_col2])
                
                if metrics:
                    st.subheader("📊 Metrics")
                    
                    metric_cols = st.columns(4)
                    for i, (key, value) in enumerate(list(metrics.items())):
                        with metric_cols[i % 4]:
                            if isinstance(value, float):
                                st.metric(key, f"{value:.4f}")
                            else:
                                st.metric(key, value)
                    
                    st.subheader("📈 Scatter Plot")
                    fig = create_scatter_plot(df, corr_col1, corr_col2)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    pearson_r = metrics.get('Pearson r', 0)
                    if abs(pearson_r) > 0.7:
                        st.success(f"Strong correlation (r = {pearson_r:.3f})")
                    elif abs(pearson_r) > 0.4:
                        st.info(f"Moderate correlation (r = {pearson_r:.3f})")
                    else:
                        st.warning(f"Weak correlation (r = {pearson_r:.3f})")
    
    with tab4:
        st.header("Categorical Analysis")
        
        if len(categorical_cols) == 0:
            st.warning("No categorical columns found.")
        else:
            selected_cat = st.selectbox("Select Column", categorical_cols)
            
            if selected_cat:
                result, value_counts = perform_categorical_analysis(df, selected_cat)
                
                if result:
                    st.subheader(f"📊 Analysis for {selected_cat}")
                    
                    cols = st.columns(5)
                    for i, (key, value) in enumerate(result.items()):
                        with cols[i % 5]:
                            st.metric(key, value)
                    
                    st.subheader("📊 Distribution")
                    fig = create_categorical_plot(value_counts, selected_cat)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📋 Value Counts")
                    vc_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / value_counts.sum() * 100).round(2)
                    })
                    st.dataframe(vc_df, use_container_width=True, hide_index=True)
    
    with tab5:
        st.header("Advanced Visualizations")
        
        st.subheader("📅 Time Series")
        
        all_cols = list(df.columns)
        date_col = st.selectbox("Date/Time Column", all_cols, key='d')
        value_cols = st.multiselect("Value Columns", numeric_cols, key='v')
        
        if date_col and len(value_cols) > 0:
            if st.button("Generate Time Series"):
                fig = create_time_series_plot(df, date_col, value_cols)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not generate plot. Check date format.")
        
        st.markdown("---")
        st.subheader("📈 Custom Scatter")
        
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox("X-Axis", numeric_cols, key='sx')
        with col2:
            scatter_y = st.selectbox("Y-Axis", numeric_cols, key='sy')
        
        if scatter_x and scatter_y:
            fig = create_scatter_plot(df, scatter_x, scatter_y)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("Export Data")
        
        csv_full = df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Data",
            csv_full,
            f"{sheet_name}_full.csv",
            "text/csv"
        )
        
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            csv_numeric = numeric_df.to_csv(index=False)
            st.download_button(
                "📥 Download Numeric Only",
                csv_numeric,
                f"{sheet_name}_numeric.csv",
                "text/csv"
            )
        
        if len(numeric_cols) > 0:
            summary_stats = df[numeric_cols].describe()
            csv_stats = summary_stats.to_csv()
            st.download_button(
                "📥 Download Statistics",
                csv_stats,
                f"{sheet_name}_stats.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
