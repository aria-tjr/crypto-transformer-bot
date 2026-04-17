# 🚀 Pro-Level Transformer Trading Bot Upgrade

This update replaces the old TCN model with a state-of-the-art **Transformer Encoder** architecture, similar to the technology behind GPT, but optimized for time-series financial data.

## 🧠 New Model Architecture: `BTCTransformer`
- **Multi-Head Self-Attention**: Allows the model to look at the entire 6-hour history (72 candles) at once and understand complex relationships between different market events.
- **Positional Encoding**: Gives the model a sense of time and sequence order.
- **12-Factor Input Vector**: Instead of just price, the model now sees:
    1.  **Momentum**: 1h and 4h price velocity.
    2.  **Volatility**: Standard deviation and ATR (Average True Range).
    3.  **Oscillators**: RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence).
    4.  **Relative Position**: Bollinger Bands width/position and distance from EMA50.
    5.  **Volume**: Volume relative to 20-period average.

## 🛠️ Setup Instructions

### 1. Train the Model (Required!)
The bot code has been updated, but it needs the "brain" (the trained model file) to function.

1.  Open **Google Colab** (https://colab.research.google.com/).
2.  Upload the file `colab_btc_robust_training.ipynb` from your workspace.
3.  In Colab, go to **Runtime > Change runtime type** and select **T4 GPU**.
4.  Run all cells in the notebook.
    *   *Note: The training includes "Adversarial Training" (FGSM) to make the model robust against market noise.*
5.  When training finishes, the notebook will download a file named `btc_transformer_best.pt`.

### 2. Install the Model
1.  Take the downloaded `btc_transformer_best.pt` file.
2.  Place it in your workspace folder: `trading_bot/checkpoints/`.
    *   *Full Path:* `trading_bot/checkpoints/btc_transformer_best.pt`

### 3. Run the Bot
The `ai_sniper_bot.py` has already been updated to use the new model.

```bash
python ai_sniper_bot.py
```

## 📈 Expected Improvements
- **Better Context**: The Transformer understands "market regimes" (e.g., trending vs. ranging) better than the TCN.
- **Noise Filtering**: The adversarial training helps the bot ignore fake-outs and stop-hunts.
- **Smarter Entries**: With 12 distinct features, the bot can distinguish between a high-probability breakout and a trap.

## ⚠️ Disclaimer

This software is provided **for educational and research purposes only** and is **not** financial advice. The transformer model's predictions are algorithmic outputs, not investment recommendations. Trading leveraged perpetual futures can result in rapid liquidation and total loss of capital. Model performance in backtest does not guarantee live performance. The author assumes no liability for losses. Train and validate thoroughly on paper/testnet before any live deployment.
