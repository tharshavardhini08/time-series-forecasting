import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import math
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, SimpleRNN, Dense, Dropout,
                                          Bidirectional, Input, Add,
                                          BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1D9E75, #085041);
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        font-size: 24px;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        font-size: 13px;
        margin: 4px 0 0 0;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-label {
        font-size: 12px;
        color: #6c757d;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
    }
    .metric-green { color: #1D9E75; }
    .metric-blue  { color: #185FA5; }
    .metric-amber { color: #854F0B; }
    .winner-box {
        background: #E1F5EE;
        border: 1px solid #5DCAA5;
        border-radius: 10px;
        padding: 14px;
        margin-top: 10px;
    }
    .status-box {
        background: #E1F5EE;
        border: 1px solid #5DCAA5;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #085041;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 Sales Forecasting Dashboard</h1>
    <p>Time Series Forecasting — Gojan School of Business & Technology, Anna University</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_or_generate_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower().str.strip()
        if 'price' not in df.columns or 'expense' not in df.columns:
            st.error("CSV must have 'price' and 'expense' columns.")
            return None
        if 'revenue' not in df.columns:
            df['revenue'] = (df['price'] + df['expense']) * 1.6
    else:
        np.random.seed(42)
        n = 120
        base     = np.linspace(500, 1500, n)
        seasonal = 200 * np.sin(np.linspace(0, 6 * np.pi, n))
        noise    = np.random.normal(0, 50, n)
        prices   = (base + seasonal + noise).clip(100, 2000).astype(int)
        expenses = (prices * 0.35 + np.random.normal(0, 20, n)).clip(50, 800).astype(int)
        df = pd.DataFrame({
            'transactionid': range(1, n + 1),
            'customerid':    np.random.randint(1000, 2000, n),
            'productid':     np.random.randint(100, 200, n),
            'price':         prices,
            'expense':       expenses,
            'date':          pd.date_range('2015-01-01', periods=n, freq='ME')
        })
        df['revenue'] = (df['price'] + df['expense']) * 1.6

    if 'date' not in df.columns:
        df['date'] = pd.date_range('2015-01-01', periods=len(df), freq='ME')

    df['profit'] = df['revenue'] - df['price'] - df['expense']
    df['date']   = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

def create_sequences(data, step=12):
    data = np.array(data).flatten()
    X, y = [], []
    if len(data) <= step:
        return np.array([]), np.array([])
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

def compute_metrics(true, pred):
    true  = np.array(true).flatten()
    pred  = np.array(pred).flatten()
    mae   = mean_absolute_error(true, pred)
    rmse  = math.sqrt(mean_squared_error(true, pred))
    denom = np.where(np.abs(true) < 1e-10, 1e-10, true)
    mape  = np.mean(np.abs((true - pred) / denom)) * 100
    return round(mae, 2), round(rmse, 2), round(mape, 2)

def build_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x  = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x  = BatchNormalization()(x)
    x  = Dropout(0.3)(x)
    x2 = Bidirectional(LSTM(64, return_sequences=True))(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    x  = Add()([x, x2])
    x  = LSTM(64)(x)
    x  = Dropout(0.3)(x)
    out = Dense(1)(x)
    return Model(inputs, out)

def plot_line(actual, predicted, label1, label2, color1, color2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=actual, name=label1,
        line=dict(color=color1, width=2),
        mode='lines+markers', marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        y=predicted, name=label2,
        line=dict(color=color2, width=2, dash='dot'),
        mode='lines+markers', marker=dict(size=4)
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=1.1),
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee')
    )
    return fig

def show_metrics(mae, rmse, mape, color_class):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">MAE — Mean Absolute Error</div>
            <div class="metric-value {color_class}">{mae:,.2f}</div>
            <div style="font-size:11px;color:#999;">lower is better</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">RMSE — Root Mean Squared Error</div>
            <div class="metric-value {color_class}">{rmse:,.2f}</div>
            <div style="font-size:11px;color:#999;">lower is better</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">MAPE — Percentage Error</div>
            <div class="metric-value {color_class}">{mape:.2f}%</div>
            <div style="font-size:11px;color:#999;">lower is better</div>
        </div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📂 Data Input")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    use_sample    = st.button("▶ Use Sample Data", use_container_width=True)
    st.markdown("---")
    st.markdown("### 🔮 Choose Model")
    model_choice = st.selectbox("", ["ARIMA", "RNN", "LSTM", "Compare All"])
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    if model_choice == "ARIMA":
        st.info("Statistical model. Best for linear trends. Fast training.")
    elif model_choice == "RNN":
        st.info("Basic deep learning. Captures short-term patterns.")
    elif model_choice == "LSTM":
        st.info("Advanced deep learning. Best for long-term complex patterns.")
    else:
        st.info("Compare all three models side by side.")

if uploaded_file or use_sample:
    df = load_or_generate_data(uploaded_file if uploaded_file else None)

    if df is None:
        st.stop()

    monthly_profit = df['profit'].resample('ME').sum()
    yearly_profit  = df['profit'].resample('YE').sum()

    st.markdown("#### 📈 Monthly Profit Trend")
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=monthly_profit.index,
        y=monthly_profit.values,
        fill='tozeroy',
        line=dict(color='#1D9E75', width=2),
        fillcolor='rgba(29,158,117,0.1)',
        name='Monthly Profit'
    ))
    fig_monthly.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee'),
        showlegend=False
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    col_y, col_c = st.columns(2)

    with col_y:
        st.markdown("#### 📊 Yearly Profit Summary")
        fig_yearly = go.Figure(go.Bar(
            x=[str(d.year) for d in yearly_profit.index],
            y=yearly_profit.values,
            marker_color='rgba(55,138,221,0.7)',
            marker_line_width=0
        ))
        fig_yearly.update_layout(
            height=200, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#eee'),
            showlegend=False
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

    with col_c:
        st.markdown("#### ✅ Profit Classification")
        profitable = int((yearly_profit > 0).sum())
        loss_years = int((yearly_profit <= 0).sum())
        pc1, pc2 = st.columns(2)
        pc1.metric("Profitable Years", profitable)
        pc2.metric("Loss Years", loss_years)
        if loss_years == 0:
            st.success("All years are profitable!")
        else:
            st.warning(f"{loss_years} loss year(s) detected.")
        total = profitable + loss_years
        st.progress(profitable / total if total > 0 else 1.0,
                    text=f"{profitable}/{total} years profitable")

    st.markdown("---")

    forecast_data = monthly_profit.dropna().values.reshape(-1, 1)
    if len(forecast_data) < 15:
        st.error("Need at least 15 months of data.")
        st.stop()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaled_x = scaler_x.fit_transform(forecast_data)
    scaled_y = scaler_y.fit_transform(forecast_data)

    step  = min(12, len(forecast_data) // 3)
    X, y  = create_sequences(scaled_x, step=step)

    if len(X) == 0:
        st.error("Not enough data. Upload more rows.")
        st.stop()

    X     = X.reshape((X.shape[0], X.shape[1], 1))
    split = max(2, int(len(X) * 0.8))
    if split >= len(X):
        split = len(X) - 1

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_test) == 0:
        X_test = X[-1:]
        y_test = y[-1:]

    mae = rmse = mape = 0

    if model_choice == "ARIMA":
        st.markdown("#### 🤖 ARIMA — Actual vs Forecast")
        try:
            n_test       = min(12, max(3, len(monthly_profit) // 5))
            train_series = monthly_profit.iloc[:-n_test]
            test_series  = monthly_profit.iloc[-n_test:]

            with st.spinner("Training ARIMA..."):
                result   = ARIMA(train_series, order=(2, 1, 2)).fit()
                forecast = result.forecast(steps=n_test)

            mae, rmse, mape = compute_metrics(
                test_series.values, forecast.values
            )
            st.plotly_chart(
                plot_line(test_series.values, forecast.values,
                          "Actual", "Forecast", "#378ADD", "#D85A30"),
                use_container_width=True
            )
            show_metrics(mae, rmse, mape, "metric-blue")

        except Exception as e:
            st.error(f"ARIMA error: {e}")

    elif model_choice == "RNN":
        if not TF_AVAILABLE:
            st.error("TensorFlow not available. Check requirements.txt")
            st.stop()

        st.markdown("#### 🤖 RNN — Actual vs Predicted")
        try:
            with st.spinner("Training RNN... please wait"):
                rnn = Sequential([
                    SimpleRNN(64, return_sequences=True,
                              input_shape=(step, 1)),
                    SimpleRNN(32),
                    Dense(1)
                ])
                rnn.compile(optimizer='adam', loss='mse')
                rnn.fit(X_train, y_train, epochs=50,
                        batch_size=4, verbose=0)

            pred     = rnn.predict(X_test, verbose=0)
            inv_pred = scaler_y.inverse_transform(pred.reshape(-1, 1))
            inv_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            mae, rmse, mape = compute_metrics(inv_true, inv_pred)

            st.plotly_chart(
                plot_line(inv_true.flatten(), inv_pred.flatten(),
                          "Actual", "Predicted", "#378ADD", "#EF9F27"),
                use_container_width=True
            )
            show_metrics(mae, rmse, mape, "metric-amber")

        except Exception as e:
            st.error(f"RNN error: {e}")

    elif model_choice == "LSTM":
        if not TF_AVAILABLE:
            st.error("TensorFlow not available. Check requirements.txt")
            st.stop()

        st.markdown("#### 🤖 LSTM — Actual vs Predicted")
        try:
            with st.spinner("Training LSTM... this takes 1-2 minutes"):
                lstm_model = build_lstm((step, 1))
                lstm_model.compile(
                    optimizer=Adam(learning_rate=0.0005), loss='mse'
                )
                val_split = 0.2 if len(X_train) >= 10 else 0.0
                history   = lstm_model.fit(
                    X_train, y_train,
                    epochs=100, batch_size=4,
                    validation_split=val_split,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.5, patience=5)
                    ]
                )

            pred     = lstm_model.predict(X_test, verbose=0)
            inv_pred = scaler_y.inverse_transform(pred.reshape(-1, 1))
            inv_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            mae, rmse, mape = compute_metrics(inv_true, inv_pred)

            st.plotly_chart(
                plot_line(inv_true.flatten(), inv_pred.flatten(),
                          "Actual", "Predicted", "#378ADD", "#1D9E75"),
                use_container_width=True
            )

            st.markdown("#### 📉 Training Loss vs Validation Loss")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history.history['loss'],
                name='Train Loss',
                line=dict(color='#378ADD', width=2)
            ))
            if val_split > 0 and 'val_loss' in history.history:
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Val Loss',
                    line=dict(color='#D85A30', width=2, dash='dot')
                ))
            fig_loss.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#eee',
                           title='Epochs'),
                yaxis=dict(showgrid=True, gridcolor='#eee',
                           title='Loss'),
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            show_metrics(mae, rmse, mape, "metric-green")

            st.markdown("""<div class="winner-box">
                <b>LSTM architecture used:</b><br>
                Bidirectional LSTM layers + Batch Normalization +
                Dropout regularization + Residual connections +
                Early stopping + Learning rate scheduler
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"LSTM error: {e}")

    elif model_choice == "Compare All":
        if not TF_AVAILABLE:
            st.warning(
                "TensorFlow unavailable — showing ARIMA only. "
                "Fix requirements.txt for RNN and LSTM."
            )

        st.markdown("#### 📊 Running all models for comparison...")

        results = {}

        with st.spinner("Training ARIMA..."):
            try:
                n_test = min(12, max(3, len(monthly_profit) // 5))
                ts     = monthly_profit.iloc[-n_test:]
                fc     = ARIMA(monthly_profit.iloc[:-n_test],
                               order=(2,1,2)).fit().forecast(n_test)
                m, r, p        = compute_metrics(ts.values, fc.values)
                results['ARIMA'] = (m, r, p)
                st.success(f"ARIMA done — MAE: {m} | RMSE: {r} | MAPE: {p}%")
            except Exception as e:
                st.error(f"ARIMA: {e}")

        if TF_AVAILABLE:
            with st.spinner("Training RNN..."):
                try:
                    rnn = Sequential([
                        SimpleRNN(64, return_sequences=True,
                                  input_shape=(step, 1)),
                        SimpleRNN(32), Dense(1)
                    ])
                    rnn.compile(optimizer='adam', loss='mse')
                    rnn.fit(X_train, y_train, epochs=50,
                            batch_size=4, verbose=0)
                    pred = rnn.predict(X_test, verbose=0)
                    ip   = scaler_y.inverse_transform(pred.reshape(-1,1))
                    it   = scaler_y.inverse_transform(y_test.reshape(-1,1))
                    m, r, p         = compute_metrics(it, ip)
                    results['RNN']  = (m, r, p)
                    st.success(
                        f"RNN done — MAE: {m} | RMSE: {r} | MAPE: {p}%"
                    )
                except Exception as e:
                    st.error(f"RNN: {e}")

            with st.spinner("Training LSTM... (1-2 mins)"):
                try:
                    lm = build_lstm((step, 1))
                    lm.compile(optimizer=Adam(0.0005), loss='mse')
                    val_split = 0.2 if len(X_train) >= 10 else 0.0
                    lm.fit(X_train, y_train, epochs=100, batch_size=4,
                           validation_split=val_split, verbose=0,
                           callbacks=[
                               EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True),
                               ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5, patience=5)
                           ])
                    pred = lm.predict(X_test, verbose=0)
                    ip   = scaler_y.inverse_transform(pred.reshape(-1,1))
                    it   = scaler_y.inverse_transform(y_test.reshape(-1,1))
                    m, r, p          = compute_metrics(it, ip)
                    results['LSTM']  = (m, r, p)
                    st.success(
                        f"LSTM done — MAE: {m} | RMSE: {r} | MAPE: {p}%"
                    )
                except Exception as e:
                    st.error(f"LSTM: {e}")

        if results:
            st.markdown("#### 📊 Model Comparison (lower = better)")
            models = list(results.keys())
            maes   = [results[m][0] for m in models]
            rmses  = [results[m][1] for m in models]
            mapes  = [results[m][2] for m in models]
            colors = {'ARIMA':'rgba(55,138,221,0.7)',
                      'RNN':  'rgba(239,159,39,0.7)',
                      'LSTM': 'rgba(29,158,117,0.7)'}

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name='MAE', x=models, y=maes,
                marker_color=[colors[m] for m in models]
            ))
            fig_cmp.add_trace(go.Bar(
                name='RMSE', x=models, y=rmses,
                marker_color=[colors[m] for m in models],
                marker_pattern_shape='x'
            ))
            fig_cmp.update_layout(
                height=300, barmode='group',
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#eee'),
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            best = min(results, key=lambda m: results[m][2])
            st.markdown(f"""<div class="winner-box">
                <b>Best model: {best}</b> with lowest MAPE of
                {results[best][2]:.2f}% — most accurate for this dataset.
            </div>""", unsafe_allow_html=True)

    if mae > 0:
        st.markdown(
            f"""<div class="status-box">
            Done! &nbsp; MAE: {mae:,.2f} &nbsp;|&nbsp;
            RMSE: {rmse:,.2f} &nbsp;|&nbsp; MAPE: {mape:.2f}%
            </div>""",
            unsafe_allow_html=True
        )

    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    if mae > 0:
        st.session_state.model_metrics[model_choice] = {
            "MAE": mae, "RMSE": rmse, "MAPE": mape
        }

else:
    st.info(
        "Upload a CSV file from the sidebar OR click "
        "'Use Sample Data' to begin."
    )
    st.markdown("""
    ### How to use this app
    1. Click **Use Sample Data** in the sidebar
    2. View the monthly and yearly profit charts
    3. Select a model — ARIMA, RNN, or LSTM
    4. View actual vs predicted chart and accuracy metrics
    5. Select **Compare All** to see all models side by side

    ### What each model does
    - **ARIMA** — Statistical model, best for linear trends, very fast
    - **RNN** — Basic deep learning, captures short-term patterns
    - **LSTM** — Advanced deep learning with bidirectional layers,
    best accuracy for complex long-term patterns
    """)
