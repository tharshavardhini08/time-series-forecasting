import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
import matplotlib.pyplot as plt

st.title("📊 Time Series Forecasting (RNN & LSTM)")

# Sample data button
if st.button("Use Sample Data"):
    np.random.seed(42)
    data = np.random.randint(100, 500, 120)
    df = pd.DataFrame({"value": data})
    df["date"] = pd.date_range(start="2015-01-01", periods=120, freq="ME")
    df.set_index("date", inplace=True)

    st.line_chart(df)

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create sequences
    def create_seq(data, step=10):
        X, y = [], []
        for i in range(len(data) - step):
            X.append(data[i:i+step])
            y.append(data[i+step])
        return np.array(X), np.array(y)

    X, y = create_seq(scaled_data)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model_type = st.selectbox("Choose Model", ["RNN", "LSTM"])

    if model_type == "RNN":
        st.write("Training RNN...")
        model = Sequential([
            SimpleRNN(50, input_shape=(X.shape[1],1)),
            Dense(1)
        ])
    else:
        st.write("Training LSTM...")
        model = Sequential([
            LSTM(50, input_shape=(X.shape[1],1)),
            Dense(1)
        ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, verbose=0)

    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    true = scaler.inverse_transform(y_test.reshape(-1,1))

    result = pd.DataFrame({
        "Actual": true.flatten(),
        "Predicted": pred.flatten()
    })

    st.subheader("Prediction")
    st.line_chart(result)

else:
    st.warning("Click 'Use Sample Data'")
