import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """Advanced time series analysis and forecasting capabilities."""
    
    def __init__(self):
        self.analysis_results = {}
        self.forecasting_models = {}
        
    def analyze_time_series(self, df: pd.DataFrame, date_col: str, 
                          value_col: str, freq: str = 'D') -> Dict[str, Any]:
        """
        Comprehensive time series analysis including stationarity, seasonality, and forecasting.
        """
        try:
            # Prepare time series data
            ts_df = self._prepare_time_series(df, date_col, value_col, freq)
            
            if ts_df is None or len(ts_df) < 10:
                return {'error': 'Insufficient data for time series analysis'}
            
            results = {
                'data_info': {
                    'total_observations': len(ts_df),
                    'date_range': {
                        'start': ts_df.index.min().strftime('%Y-%m-%d'),
                        'end': ts_df.index.max().strftime('%Y-%m-%d')
                    },
                    'frequency': freq,
                    'missing_values': ts_df[value_col].isnull().sum()
                },
                'statistical_tests': {},
                'decomposition': {},
                'trend_analysis': {},
                'forecasting': {},
                'anomaly_detection': {},
                'visualizations': {}
            }
            
            # 1. Basic statistical analysis
            basic_stats = self._calculate_basic_statistics(ts_df[value_col])
            results['basic_statistics'] = basic_stats
            
            # 2. Stationarity tests
            stationarity_results = self._test_stationarity(ts_df[value_col])
            results['statistical_tests']['stationarity'] = stationarity_results
            
            # 3. Seasonal decomposition
            if len(ts_df) >= 24:  # Need sufficient data for decomposition
                decomposition_results = self._perform_decomposition(ts_df[value_col])
                results['decomposition'] = decomposition_results
            
            # 4. Trend analysis
            trend_analysis = self._analyze_trend(ts_df[value_col])
            results['trend_analysis'] = trend_analysis
            
            # 5. Autocorrelation analysis
            autocorr_results = self._analyze_autocorrelation(ts_df[value_col])
            results['autocorrelation'] = autocorr_results
            
            # 6. Anomaly detection
            anomalies = self._detect_anomalies(ts_df[value_col])
            results['anomaly_detection'] = anomalies
            
            # 7. Forecasting
            if len(ts_df) >= 30:  # Need sufficient data for forecasting
                forecast_results = self._perform_forecasting(ts_df[value_col])
                results['forecasting'] = forecast_results
            
            # 8. Generate insights and recommendations
            insights = self._generate_time_series_insights(results)
            results['insights'] = insights
            
            return results
            
        except Exception as e:
            return {'error': f"Time series analysis failed: {str(e)}"}
    
    def _prepare_time_series(self, df: pd.DataFrame, date_col: str, 
                           value_col: str, freq: str) -> Optional[pd.DataFrame]:
        """Prepare and clean time series data."""
        try:
            # Make a copy and ensure date column is datetime
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            
            # Remove duplicates and sort by date
            ts_df = ts_df.drop_duplicates(subset=[date_col]).sort_values(date_col)
            
            # Set date as index
            ts_df.set_index(date_col, inplace=True)
            
            # Handle missing values
            ts_df[value_col] = pd.to_numeric(ts_df[value_col], errors='coerce')
            
            # Fill missing values using interpolation
            ts_df[value_col] = ts_df[value_col].interpolate(method='time')
            
            # Resample to consistent frequency if needed
            if freq and len(ts_df) > 10:
                ts_df = ts_df.resample(freq).mean()
                ts_df[value_col] = ts_df[value_col].interpolate()
            
            return ts_df
            
        except Exception:
            return None
    
    def _calculate_basic_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistical properties of the time series."""
        return {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'variance': float(series.var()),
            'min': float(series.min()),
            'max': float(series.max()),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else 0
        }
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Perform stationarity tests (ADF and KPSS)."""
        results = {}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            results['adf_test'] = {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                'is_stationary': bool(adf_result[1] < 0.05)
            }
            
            # KPSS test
            kpss_result = kpss(series.dropna())
            results['kpss_test'] = {
                'statistic': float(kpss_result[0]),
                'p_value': float(kpss_result[1]),
                'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                'is_stationary': bool(kpss_result[1] > 0.05)
            }
            
            # Overall assessment
            adf_stationary = results['adf_test']['is_stationary']
            kpss_stationary = results['kpss_test']['is_stationary']
            
            if adf_stationary and kpss_stationary:
                results['overall_assessment'] = 'Stationary'
            elif not adf_stationary and not kpss_stationary:
                results['overall_assessment'] = 'Non-stationary'
            else:
                results['overall_assessment'] = 'Inconclusive'
                
        except Exception as e:
            results['error'] = f"Stationarity tests failed: {str(e)}"
        
        return results
    
    def _perform_decomposition(self, series: pd.Series) -> Dict[str, Any]:
        """Perform seasonal decomposition of the time series."""
        try:
            # Try different periods for decomposition
            periods_to_try = [7, 12, 24, 30, 365]  # Daily, monthly, etc.
            best_decomposition = None
            best_period = None
            
            for period in periods_to_try:
                if len(series) >= 2 * period:
                    try:
                        decomposition = seasonal_decompose(
                            series.dropna(), 
                            model='additive', 
                            period=period
                        )
                        best_decomposition = decomposition
                        best_period = period
                        break
                    except:
                        continue
            
            if best_decomposition is None:
                return {'error': 'Could not perform seasonal decomposition'}
            
            # Calculate component statistics
            trend_strength = 1 - (best_decomposition.resid.var() / (best_decomposition.trend + best_decomposition.resid).var())
            seasonal_strength = 1 - (best_decomposition.resid.var() / (best_decomposition.seasonal + best_decomposition.resid).var())
            
            return {
                'period_used': best_period,
                'trend_strength': float(trend_strength) if not np.isnan(trend_strength) else 0,
                'seasonal_strength': float(seasonal_strength) if not np.isnan(seasonal_strength) else 0,
                'residual_variance': float(best_decomposition.resid.var()),
                'components_summary': {
                    'trend_mean': float(best_decomposition.trend.mean()) if not best_decomposition.trend.isna().all() else 0,
                    'seasonal_amplitude': float(best_decomposition.seasonal.std()) if not best_decomposition.seasonal.isna().all() else 0,
                    'residual_std': float(best_decomposition.resid.std()) if not best_decomposition.resid.isna().all() else 0
                }
            }
            
        except Exception as e:
            return {'error': f"Decomposition failed: {str(e)}"}
    
    def _analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze trend characteristics using various methods."""
        try:
            # Linear trend analysis
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Moving averages
            ma_short = series.rolling(window=min(7, len(series)//4)).mean()
            ma_long = series.rolling(window=min(30, len(series)//2)).mean()
            
            # Trend direction
            if abs(slope) < std_err:
                trend_direction = 'No clear trend'
            elif slope > 0:
                trend_direction = 'Upward trend'
            else:
                trend_direction = 'Downward trend'
            
            # Calculate trend strength
            trend_strength = abs(r_value)
            
            return {
                'linear_trend': {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'direction': trend_direction,
                    'strength': float(trend_strength)
                },
                'moving_averages': {
                    'short_term_trend': 'Increasing' if ma_short.diff().mean() > 0 else 'Decreasing',
                    'long_term_trend': 'Increasing' if ma_long.diff().mean() > 0 else 'Decreasing'
                },
                'volatility': {
                    'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else 0,
                    'max_drawdown': float((series.cummax() - series).max() / series.cummax().max()) if series.cummax().max() != 0 else 0
                }
            }
            
        except Exception as e:
            return {'error': f"Trend analysis failed: {str(e)}"}
    
    def _analyze_autocorrelation(self, series: pd.Series, max_lags: int = 20) -> Dict[str, Any]:
        """Analyze autocorrelation patterns."""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            clean_series = series.dropna()
            if len(clean_series) < max_lags + 1:
                max_lags = len(clean_series) - 1
            
            # Calculate autocorrelation function
            autocorr = acf(clean_series, nlags=max_lags, fft=True)
            
            # Calculate partial autocorrelation function
            partial_autocorr = pacf(clean_series, nlags=max_lags)
            
            # Find significant lags
            significant_lags = []
            confidence_interval = 1.96 / np.sqrt(len(clean_series))
            
            for i, corr in enumerate(autocorr[1:], 1):
                if abs(corr) > confidence_interval:
                    significant_lags.append({
                        'lag': i,
                        'correlation': float(corr)
                    })
            
            return {
                'max_autocorr': float(np.max(autocorr[1:])),
                'min_autocorr': float(np.min(autocorr[1:])),
                'significant_lags': significant_lags[:5],  # Top 5
                'autocorr_decay': 'Fast' if autocorr[1] < 0.5 else 'Slow',
                'ljung_box_test': self._ljung_box_test(clean_series)
            }
            
        except Exception as e:
            return {'error': f"Autocorrelation analysis failed: {str(e)}"}
    
    def _ljung_box_test(self, series: pd.Series) -> Dict[str, Any]:
        """Perform Ljung-Box test for autocorrelation."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            result = acorr_ljungbox(series, lags=min(10, len(series)//4), return_df=True)
            
            return {
                'statistic': float(result['lb_stat'].iloc[-1]),
                'p_value': float(result['lb_pvalue'].iloc[-1]),
                'has_autocorrelation': bool(result['lb_pvalue'].iloc[-1] < 0.05)
            }
        except:
            return {'error': 'Ljung-Box test failed'}
    
    def _detect_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """Detect anomalies in the time series using statistical methods."""
        try:
            clean_series = series.dropna()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(clean_series))
            z_anomalies = clean_series[z_scores > 3]
            
            # IQR method
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            
            # Moving average method
            window_size = min(30, len(clean_series) // 4)
            if window_size > 1:
                rolling_mean = clean_series.rolling(window=window_size, center=True).mean()
                rolling_std = clean_series.rolling(window=window_size, center=True).std()
                ma_anomalies = clean_series[abs(clean_series - rolling_mean) > 2 * rolling_std]
            else:
                ma_anomalies = pd.Series(dtype=float)
            
            return {
                'total_anomalies': {
                    'z_score_method': len(z_anomalies),
                    'iqr_method': len(iqr_anomalies),
                    'moving_average_method': len(ma_anomalies)
                },
                'anomaly_percentage': {
                    'z_score': float(len(z_anomalies) / len(clean_series) * 100),
                    'iqr': float(len(iqr_anomalies) / len(clean_series) * 100),
                    'moving_average': float(len(ma_anomalies) / len(clean_series) * 100)
                },
                'anomaly_summary': {
                    'has_anomalies': len(z_anomalies) > 0 or len(iqr_anomalies) > 0,
                    'severity': 'High' if len(z_anomalies) > len(clean_series) * 0.05 else 'Low'
                }
            }
            
        except Exception as e:
            return {'error': f"Anomaly detection failed: {str(e)}"}
    
    def _perform_forecasting(self, series: pd.Series, forecast_periods: int = 30) -> Dict[str, Any]:
        """Perform forecasting using multiple methods."""
        try:
            clean_series = series.dropna()
            
            if len(clean_series) < 10:
                return {'error': 'Insufficient data for forecasting'}
            
            results = {}
            
            # 1. Simple exponential smoothing
            try:
                exp_smooth = ExponentialSmoothing(clean_series, trend=None, seasonal=None)
                exp_smooth_fit = exp_smooth.fit()
                exp_forecast = exp_smooth_fit.forecast(forecast_periods)
                
                results['exponential_smoothing'] = {
                    'forecast': exp_forecast.tolist(),
                    'aic': float(exp_smooth_fit.aic),
                    'method': 'Simple Exponential Smoothing'
                }
            except:
                pass
            
            # 2. Holt's linear trend method
            try:
                holt = ExponentialSmoothing(clean_series, trend='add', seasonal=None)
                holt_fit = holt.fit()
                holt_forecast = holt_fit.forecast(forecast_periods)
                
                results['holt_linear'] = {
                    'forecast': holt_forecast.tolist(),
                    'aic': float(holt_fit.aic),
                    'method': 'Holt Linear Trend'
                }
            except:
                pass
            
            # 3. ARIMA model (simple auto-selection)
            try:
                # Simple ARIMA(1,1,1) as baseline
                arima = ARIMA(clean_series, order=(1, 1, 1))
                arima_fit = arima.fit()
                arima_forecast = arima_fit.forecast(forecast_periods)
                
                results['arima'] = {
                    'forecast': arima_forecast.tolist(),
                    'aic': float(arima_fit.aic),
                    'method': 'ARIMA(1,1,1)'
                }
            except:
                pass
            
            # 4. Linear trend forecast
            try:
                x = np.arange(len(clean_series))
                slope, intercept, _, _, _ = stats.linregress(x, clean_series.values)
                
                future_x = np.arange(len(clean_series), len(clean_series) + forecast_periods)
                linear_forecast = slope * future_x + intercept
                
                results['linear_trend'] = {
                    'forecast': linear_forecast.tolist(),
                    'method': 'Linear Trend Extrapolation'
                }
            except:
                pass
            
            # Select best model based on AIC
            best_model = None
            best_aic = float('inf')
            
            for model_name, model_results in results.items():
                if 'aic' in model_results and model_results['aic'] < best_aic:
                    best_aic = model_results['aic']
                    best_model = model_name
            
            if best_model:
                results['recommended_model'] = best_model
                results['best_forecast'] = results[best_model]['forecast']
            
            return results
            
        except Exception as e:
            return {'error': f"Forecasting failed: {str(e)}"}
    
    def _generate_time_series_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights and recommendations based on time series analysis."""
        insights = []
        
        # Data quality insights
        data_info = results.get('data_info', {})
        if data_info.get('missing_values', 0) > 0:
            insights.append(f"Dataset has {data_info['missing_values']} missing values - interpolation was applied")
        
        # Stationarity insights
        stationarity = results.get('statistical_tests', {}).get('stationarity', {})
        overall_assessment = stationarity.get('overall_assessment', 'Unknown')
        if overall_assessment == 'Non-stationary':
            insights.append("Time series is non-stationary - consider differencing or transformation for modeling")
        elif overall_assessment == 'Stationary':
            insights.append("Time series is stationary - suitable for most time series models")
        
        # Trend insights
        trend_analysis = results.get('trend_analysis', {})
        linear_trend = trend_analysis.get('linear_trend', {})
        if linear_trend.get('direction') != 'No clear trend':
            direction = linear_trend.get('direction', '')
            strength = linear_trend.get('strength', 0)
            insights.append(f"{direction} detected with strength {strength:.2f}")
        
        # Seasonality insights
        decomposition = results.get('decomposition', {})
        seasonal_strength = decomposition.get('seasonal_strength', 0)
        if seasonal_strength > 0.3:
            insights.append(f"Strong seasonal pattern detected (strength: {seasonal_strength:.2f})")
        elif seasonal_strength > 0.1:
            insights.append(f"Moderate seasonal pattern detected (strength: {seasonal_strength:.2f})")
        
        # Anomaly insights
        anomaly_detection = results.get('anomaly_detection', {})
        anomaly_summary = anomaly_detection.get('anomaly_summary', {})
        if anomaly_summary.get('has_anomalies', False):
            severity = anomaly_summary.get('severity', 'Unknown')
            insights.append(f"Anomalies detected with {severity.lower()} severity - review data quality")
        
        # Forecasting insights
        forecasting = results.get('forecasting', {})
        if 'recommended_model' in forecasting:
            model = forecasting['recommended_model']
            insights.append(f"Best forecasting model identified: {model}")
        
        # Autocorrelation insights
        autocorr = results.get('autocorrelation', {})
        if autocorr.get('ljung_box_test', {}).get('has_autocorrelation', False):
            insights.append("Significant autocorrelation detected - time series models recommended")
        
        if not insights:
            insights.append("Time series analysis completed - review detailed results for patterns")
        
        return insights
    
    def create_time_series_visualizations(self, df: pd.DataFrame, date_col: str, 
                                        value_col: str, results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create comprehensive time series visualizations."""
        visualizations = {}
        
        try:
            # Prepare data
            ts_df = self._prepare_time_series(df, date_col, value_col, 'D')
            if ts_df is None:
                return {}
            
            # 1. Main time series plot
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=ts_df.index,
                y=ts_df[value_col],
                mode='lines',
                name='Original Series',
                line=dict(color='blue')
            ))
            
            # Add trend line if available
            trend_analysis = results.get('trend_analysis', {})
            linear_trend = trend_analysis.get('linear_trend', {})
            if 'slope' in linear_trend and 'intercept' in linear_trend:
                x_trend = np.arange(len(ts_df))
                y_trend = linear_trend['slope'] * x_trend + linear_trend['intercept']
                fig_main.add_trace(go.Scatter(
                    x=ts_df.index,
                    y=y_trend,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash')
                ))
            
            fig_main.update_layout(
                title='Time Series with Trend',
                xaxis_title='Date',
                yaxis_title=value_col,
                height=500
            )
            visualizations['main_plot'] = fig_main
            
            # 2. Decomposition plot if available
            if 'decomposition' in results and 'error' not in results['decomposition']:
                # Create decomposition visualization (simplified)
                fig_decomp = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Trend Strength', 'Seasonal Strength', 'Autocorrelation', 'Residual Analysis'],
                    specs=[[{"type": "indicator"}, {"type": "indicator"}],
                           [{"type": "scatter"}, {"type": "histogram"}]]
                )
                
                decomp_data = results['decomposition']
                
                # Add indicators
                fig_decomp.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=decomp_data.get('trend_strength', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Trend Strength"},
                    gauge={'axis': {'range': [None, 1]}}
                ), row=1, col=1)
                
                fig_decomp.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=decomp_data.get('seasonal_strength', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Seasonal Strength"},
                    gauge={'axis': {'range': [None, 1]}}
                ), row=1, col=2)
                
                fig_decomp.update_layout(height=600, title='Time Series Decomposition Analysis')
                visualizations['decomposition'] = fig_decomp
            
            # 3. Forecast plot if available
            forecasting = results.get('forecasting', {})
            if 'best_forecast' in forecasting:
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=ts_df.index,
                    y=ts_df[value_col],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                last_date = ts_df.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=len(forecasting['best_forecast']),
                    freq='D'
                )
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecasting['best_forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_forecast.update_layout(
                    title=f'Forecast using {forecasting.get("recommended_model", "Best Model")}',
                    xaxis_title='Date',
                    yaxis_title=value_col,
                    height=500
                )
                visualizations['forecast'] = fig_forecast
            
            return visualizations
            
        except Exception as e:
            return {'error': f"Visualization creation failed: {str(e)}"}
        
    def generate_human_friendly_time_series_report(self, results: Dict[str, Any]) -> str:
        try:
            lines = [
                f"📈 Time Series Analysis Report",
                "",
                "📅 Data Info:",
                f"- Observations: {results['data_info'].get('total_observations')}",
                f"- Date Range: {results['data_info']['date_range'].get('start')} to {results['data_info']['date_range'].get('end')}",
                f"- Frequency: {results['data_info'].get('frequency')}",
                f"- Missing Values: {results['data_info'].get('missing_values')}",
                "",
                "🧪 Stationarity Tests:",
                f"- ADF Test: p={results['statistical_tests']['stationarity']['adf_test'].get('p_value'):.4f}, Stationary: {results['statistical_tests']['stationarity']['adf_test'].get('is_stationary')}",
                f"- KPSS Test: p={results['statistical_tests']['stationarity']['kpss_test'].get('p_value'):.4f}, Stationary: {results['statistical_tests']['stationarity']['kpss_test'].get('is_stationary')}",
                f"- Overall: {results['statistical_tests']['stationarity'].get('overall_assessment')}",
                "",
                "📊 Decomposition:",
                f"- Trend Strength: {results['decomposition'].get('trend_strength'):.2f}",
                f"- Seasonal Strength: {results['decomposition'].get('seasonal_strength'):.2f}",
                f"- Residual Variance: {results['decomposition'].get('residual_variance'):.2f}",
                "",
                "📈 Trend Analysis:",
                f"- Linear Trend: {results['trend_analysis']['linear_trend'].get('direction')} (R²={results['trend_analysis']['linear_trend'].get('r_squared'):.2f})",
                f"- Short-term trend: {results['trend_analysis']['moving_averages'].get('short_term_trend')}",
                f"- Long-term trend: {results['trend_analysis']['moving_averages'].get('long_term_trend')}",
                "",
                "🔮 Forecasting:",
                f"- Recommended Model: {results['forecasting'].get('recommended_model')}",
                f"- First forecasted value: {results['forecasting']['best_forecast'][0]:.2f}",
                "",
                "📌 Anomaly Detection:",
                f"- Has anomalies: {results['anomaly_detection']['anomaly_summary'].get('has_anomalies')}",
                f"- Severity: {results['anomaly_detection']['anomaly_summary'].get('severity')}",
                "",
                "📈 Basic Statistics:",
                f"- Mean: {results['basic_statistics'].get('mean')}",
                f"- Std Dev: {results['basic_statistics'].get('std'):.2f}",
                f"- Min: {results['basic_statistics'].get('min')}",
                f"- Max: {results['basic_statistics'].get('max')}",
                f"- Coeff of Variation: {results['basic_statistics'].get('coefficient_of_variation'):.3f}",
                "",
                "🔗 Autocorrelation:",
                f"- Max autocorr: {results['autocorrelation'].get('max_autocorr'):.2f}",
                f"- Ljung-Box p-value: {results['autocorrelation']['ljung_box_test'].get('p_value'):.2e}",
                "",
                "💡 Insights:"
            ]
            for insight in results.get('insights', []):
                lines.append(f"✅ {insight}")
            return '\n'.join(lines)
        except Exception as e:
            return f"Error generating time series report: {str(e)}"
