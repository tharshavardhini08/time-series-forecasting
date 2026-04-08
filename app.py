import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Bidirectional, Input, Add, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
import math

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.sidebar.title("📂 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your sales CSV file", type=["csv"])

@st.cache_data
def load_or_generate_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()
        required_cols = ['price', 'expense']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        if 'revenue' not in df.columns:
            st.warning("revenue column missing. Generating: (price + expense) x 1.6")
            df['revenue'] = (df['price'] + df['expense']) * 1.6
    else:
        np.random.seed(42)
        n_rows = 120
        df = pd.DataFrame({
            'transactionid': range(1, n_rows + 1),
            'customerid': np.random.randint(1000, 2000, n_rows),
            'productid': np.random.randint(100, 200, n_rows),
            'price': np.random.randint(100, 500, n_rows),
            'expense': np.random.randint(50, 200, n_rows)
        })
        df['revenue'] = (df['price'] + df['expense']) * 1.6

    if 'date' not in df.columns:
        max_periods = min(len(df), 120)
        df = df.head(max_periods).reset_index(drop=True)
        df['date'] = pd.date_range(start='2015-01-01', periods=max_periods, freq='ME')

    df['profit'] = df['revenue'] - df[['price', 'expense']].sum(axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def create_sequences(data, step=12):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

def compute_metrics(true, pred):
    true = np.array(true).flatten()
    pred = np.array(pred).flatten()
    mae  = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / (true + 1e-10))) * 100
    return mae, rmse, mape

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
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

if uploaded_file or st.sidebar.button("Use Sample Data"):
    df = load_or_generate_data(uploaded_file)
    if df is None:
        st.stop()

    st.title("📊 Sales Forecasting Dashboard")

    monthly_profit = df['profit'].resample('ME').sum()

    st.subheader("Monthly Profit Trend")
    st.line_chart(monthly_profit)

    st.subheader("Yearly Profit Summary")
    yearly_profit = df['profit'].resample('YE').sum()
    st.bar_chart(yearly_profit)

    st.subheader("Profit Classification")
    yearly_class = yearly_profit.apply(lambda x: "Profit" if x > 0 else "Loss")
    st.write(yearly_class.value_counts())

    st.sidebar.title("🔮 Forecasting Model")
    model_choice = st.sidebar.selectbox("Choose Model", ["ARIMA", "RNN", "LSTM"])

    forecast_data = monthly_profit.dropna()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaled_x = scaler_x.fit_transform(forecast_data.values.reshape(-1, 1))
    scaled_y = scaler_y.fit_transform(forecast_data.values.reshape(-1, 1))

    step = 12
    X, y = create_sequences(scaled_x, step=step)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    st.subheader(f"Model: {model_choice}")
    mae = rmse = mape = 0

    if model_choice == "ARIMA":
        try:
            train_data = forecast_data[:-12]
            actual     = forecast_data[-12:]
            model  = ARIMA(train_data, order=(2, 1, 2))
            result = model.fit()
            forecast = result.forecast(steps=12)
            mae, rmse, mape = compute_metrics(actual.values, forecast.values)
            chart_df = pd.DataFrame({'Actual': actual.values, 'Forecast': forecast.values})
            st.line_chart(chart_df)
        except Exception as e:
            st.error(f"ARIMA error: {e}")

    elif model_choice == "RNN":
        try:
            with st.spinner("Training RNN model... please wait"):
                rnn_model = Sequential([
                    SimpleRNN(50, return_sequences=True, input_shape=(step, 1)),
                    SimpleRNN(50),
                    Dense(1)
                ])
                rnn_model.compile(optimizer='adam', loss='mse')
                rnn_model.fit(X_train, y_train, epochs=30, verbose=0)
                pred     = rnn_model.predict(X_test)
                inv_pred = scaler_y.inverse_transform(pred)
                inv_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                mae, rmse, mape = compute_metrics(inv_true, inv_pred)
                df_pred = pd.DataFrame({
                    "Actual":    inv_true.flatten(),
                    "Predicted": inv_pred.flatten()
                })
                st.line_chart(df_pred)
        except Exception as e:
            st.error(f"RNN error: {e}")

    elif model_choice == "LSTM":
        try:
            with st.spinner("Training LSTM model... please wait (this takes 1-2 mins)"):
                lstm_model = build_enhanced_lstm((step, 1))
                lstm_model.compile(
                    optimizer=Adam(learning_rate=0.0005), loss='mse'
                )
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                ]
                history = lstm_model.fit(
                    X_train, y_train,
                    epochs=100,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=callbacks
                )
                pred     = lstm_model.predict(X_test)
                inv_pred = scaler_y.inverse_transform(pred)
                inv_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                mae, rmse, mape = compute_metrics(inv_true, inv_pred)
                df_pred = pd.DataFrame({
                    "Actual":    inv_true.flatten(),
                    "Predicted": inv_pred.flatten()
                })
                st.line_chart(df_pred)

                st.subheader("Training Loss vs Validation Loss")
                st.line_chart(pd.DataFrame({
                    "Train Loss": history.history["loss"],
                    "Val Loss":   history.history["val_loss"]
                }))

                st.subheader("Final Conclusion")
                st.markdown(f"""
                - LSTM used Bidirectional layers, Batch Normalization, Dropout and Residual connections.
                - **MAE  : {mae:.2f}**
                - **RMSE : {rmse:.2f}**
                - **MAPE : {mape:.2f}%**
                """)
        except Exception as e:
            st.error(f"LSTM error: {e}")

    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    st.session_state.model_metrics[model_choice] = {
        "MAE": mae, "RMSE": rmse, "MAPE": mape
    }

    if st.button("Compare Models"):
        st.subheader("Model Comparison (Lower is Better)")
        metric_df = pd.DataFrame(st.session_state.model_metrics).T
        st.bar_chart(metric_df)

    st.success(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

else:
    st.warning("Please upload a CSV file or click Use Sample Data.")
