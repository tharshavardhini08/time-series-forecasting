import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import math
import warnings
warnings.filterwarnings('ignore')

# TensorFlow import with error handling
try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, SimpleRNN, Dense, Dropout,
                                          Bidirectional, Input, Add, BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📊 Sales Forecasting Dashboard")
st.sidebar.title("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your sales CSV file", type=["csv"])

# ── Data Generation / Loading ─────────────────────────────────────────────────
@st.cache_data
def load_or_generate_data(file=None):
    if file is not None:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.lower().str.strip()
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return None

        if 'price' not in df.columns or 'expense' not in df.columns:
            st.error("CSV must have 'price' and 'expense' columns.")
            return None

        if 'revenue' not in df.columns:
            df['revenue'] = (df['price'] + df['expense']) * 1.6

        if 'date' not in df.columns:
            df['date'] = pd.date_range(
                start='2015-01-01', periods=len(df), freq='ME'
            )
    else:
        np.random.seed(42)
        n_rows = 120
        base = np.linspace(500, 1500, n_rows)
        seasonal = 200 * np.sin(np.linspace(0, 6 * np.pi, n_rows))
        noise = np.random.normal(0, 50, n_rows)
        prices = (base + seasonal + noise).clip(100, 2000).astype(int)
        expenses = (prices * 0.35 + np.random.normal(0, 20, n_rows)).clip(50, 800).astype(int)

        df = pd.DataFrame({
            'transactionid': range(1, n_rows + 1),
            'customerid':    np.random.randint(1000, 2000, n_rows),
            'productid':     np.random.randint(100, 200, n_rows),
            'price':         prices,
            'expense':       expenses,
            'date':          pd.date_range(start='2015-01-01', periods=n_rows, freq='ME')
        })
        df['revenue'] = (df['price'] + df['expense']) * 1.6

    df['profit'] = df['revenue'] - df['price'] - df['expense']
    df['date']   = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

# ── Helper: Create Sequences ──────────────────────────────────────────────────
def create_sequences(data, step=12):
    data = np.array(data).flatten()
    X, y = [], []
    if len(data) <= step:
        return np.array([]), np.array([])
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# ── Helper: Compute Metrics ───────────────────────────────────────────────────
def compute_metrics(true, pred):
    true = np.array(true).flatten()
    pred = np.array(pred).flatten()
    mae  = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    denom = np.where(np.abs(true) < 1e-10, 1e-10, true)
    mape  = np.mean(np.abs((true - pred) / denom)) * 100
    return round(mae, 2), round(rmse, 2), round(mape, 2)

# ── Helper: Build LSTM ────────────────────────────────────────────────────────
def build_enhanced_lstm(input_shape):
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

# ── Load Data ─────────────────────────────────────────────────────────────────
use_sample = st.sidebar.button("▶ Use Sample Data")

if uploaded_file is not None or use_sample:

    df = load_or_generate_data(uploaded_file if uploaded_file else None)

    if df is None:
        st.stop()

    if len(df) < 20:
        st.error("Not enough data. Please upload at least 20 rows.")
        st.stop()

    # ── EDA Charts ───────────────────────────────────────────────────────────
    monthly_profit = df['profit'].resample('ME').sum()
    yearly_profit  = df['profit'].resample('YE').sum()

    st.subheader("📈 Monthly Profit Trend")
    st.line_chart(monthly_profit)

    st.subheader("📊 Yearly Profit Summary")
    st.bar_chart(yearly_profit)

    st.subheader("✅ Profit vs Loss Classification")
    yearly_class = yearly_profit.apply(lambda x: "Profit" if x > 0 else "Loss")
    col1, col2 = st.columns(2)
    col1.metric("Profitable Years", int((yearly_class == "Profit").sum()))
    col2.metric("Loss Years",       int((yearly_class == "Loss").sum()))

    # ── Model Selection ───────────────────────────────────────────────────────
    st.sidebar.title("🔮 Forecasting Model")
    model_choice = st.sidebar.selectbox("Choose Model", ["ARIMA", "RNN", "LSTM"])

    if not TF_AVAILABLE and model_choice in ["RNN", "LSTM"]:
        st.error("TensorFlow is not installed correctly on this server. Please select ARIMA or check your requirements.txt.")
        st.stop()

    # ── Preprocessing for DL ──────────────────────────────────────────────────
    forecast_data = monthly_profit.dropna().values.reshape(-1, 1)

    if len(forecast_data) < 15:
        st.error("Not enough monthly data points. Need at least 15 months.")
        st.stop()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaled_x = scaler_x.fit_transform(forecast_data)
    scaled_y = scaler_y.fit_transform(forecast_data)

    step = min(12, len(forecast_data) // 3)

    X, y = create_sequences(scaled_x, step=step)

    if len(X) == 0:
        st.error(f"Not enough data to create sequences with step={step}. Upload more data.")
        st.stop()

    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = max(2, int(len(X) * 0.8))
    if split >= len(X):
        split = len(X) - 1

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_test) == 0:
        X_test = X[-1:]
        y_test = y[-1:]

    st.subheader(f"🤖 Model: {model_choice}")
    mae = rmse = mape = 0

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    if model_choice == "ARIMA":
        try:
            series = monthly_profit.dropna()
            n_test = min(12, max(3, len(series) // 5))
            train_series = series.iloc[:-n_test]
            test_series  = series.iloc[-n_test:]

            with st.spinner("Training ARIMA model..."):
                arima_model = ARIMA(train_series, order=(2, 1, 2))
                result      = arima_model.fit()
                forecast    = result.forecast(steps=n_test)

            mae, rmse, mape = compute_metrics(test_series.values, forecast.values)

            chart_df = pd.DataFrame({
                'Actual':   test_series.values,
                'Forecast': forecast.values
            }, index=test_series.index)
            st.line_chart(chart_df)

        except Exception as e:
            st.error(f"ARIMA failed: {e}")

    # ── RNN ───────────────────────────────────────────────────────────────────
    elif model_choice == "RNN":
        try:
            with st.spinner("Training RNN model... please wait"):
                rnn_model = Sequential([
                    SimpleRNN(64, return_sequences=True,
                              input_shape=(step, 1)),
                    SimpleRNN(32),
                    Dense(1)
                ])
                rnn_model.compile(optimizer='adam', loss='mse')
                rnn_model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=4,
                    verbose=0
                )

            pred     = rnn_model.predict(X_test, verbose=0)
            inv_pred = scaler_y.inverse_transform(pred.reshape(-1, 1))
            inv_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            mae, rmse, mape = compute_metrics(inv_true, inv_pred)

            st.line_chart(pd.DataFrame({
                "Actual":    inv_true.flatten(),
                "Predicted": inv_pred.flatten()
            }))

        except Exception as e:
            st.error(f"RNN failed: {e}")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    elif model_choice == "LSTM":
        try:
            with st.spinner("Training LSTM model... this takes 1-2 minutes"):
                lstm_model = build_enhanced_lstm((step, 1))
                lstm_model.compile(
                    optimizer=Adam(learning_rate=0.0005),
                    loss='mse'
                )
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss', patience=10,
                        restore_best_weights=True
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=5
                    )
                ]

                val_split = 0.2 if len(X_train) >= 10 else 0.0

                history = lstm_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=4,
                    validation_split=val_split,
                    verbose=0,
                    callbacks=callbacks
                )

            pred     = lstm_model.predict(X_test, verbose=0)
            inv_pred = scaler_y.inverse_transform(pred.reshape(-1, 1))
            inv_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            mae, rmse, mape = compute_metrics(inv_true, inv_pred)

            st.line_chart(pd.DataFrame({
                "Actual":    inv_true.flatten(),
                "Predicted": inv_pred.flatten()
            }))

            st.subheader("📉 Training Loss vs Validation Loss")
            loss_df = {"Train Loss": history.history["loss"]}
            if val_split > 0 and "val_loss" in history.history:
                loss_df["Val Loss"] = history.history["val_loss"]
            st.line_chart(pd.DataFrame(loss_df))

            st.info(
                "LSTM used Bidirectional layers, Batch Normalization, "
                "Dropout regularization and Residual connections for "
                "improved accuracy."
            )

        except Exception as e:
            st.error(f"LSTM failed: {e}")

    # ── Metrics Display ───────────────────────────────────────────────────────
    if mae > 0:
        st.subheader("📐 Model Performance Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{mae:.2f}",  help="Mean Absolute Error — lower is better")
        m2.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error — lower is better")
        m3.metric("MAPE", f"{mape:.2f}%",help="Mean Absolute Percentage Error — lower is better")

        if 'model_metrics' not in st.session_state:
            st.session_state.model_metrics = {}
        st.session_state.model_metrics[model_choice] = {
            "MAE": mae, "RMSE": rmse, "MAPE": mape
        }

    # ── Model Comparison ──────────────────────────────────────────────────────
    if len(st.session_state.get('model_metrics', {})) > 1:
        if st.button("📊 Compare All Models"):
            st.subheader("Model Comparison (Lower = Better)")
            metric_df = pd.DataFrame(st.session_state.model_metrics).T
            st.bar_chart(metric_df)

    st.success(f"✅ Done! MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

else:
    st.info("👈 Upload a CSV file from the sidebar OR click 'Use Sample Data' to begin.")
    st.markdown("""
    ### How to Use This App
    1. Click **Use Sample Data** in the sidebar to run with built-in data
    2. OR upload your own CSV with columns: `price`, `expense` (and optionally `date`, `revenue`)
    3. Select a model: **ARIMA**, **RNN**, or **LSTM**
    4. View the predictions and accuracy metrics
    5. Run all 3 models then click **Compare All Models**
    
    ### Models Explained
    - **ARIMA** — Statistical model, best for linear trends
    - **RNN** — Basic deep learning, captures short-term patterns  
    - **LSTM** — Advanced deep learning, captures long-term complex patterns
    """)
