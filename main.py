import functions
import yfinance as yf

def main():
    ticker = 'MSFT'  # interchangeable
    ## TRAINING DATA ##
    ticker = yf.Ticker(ticker) # initialize ticker
    train_df = ticker.history(start="2020-01-01", end="2024-12-31") # full DF of stock history
    train_df = functions.df_actions(train_df) # returns df as: high,ma5,ma20,madf,target
    train_df = train_df.dropna() # drop NaN's (lack of data)
    x_train = train_df['MADF'] # MADF
    y_train = train_df['Target'] # Tomorrow's Price

    ## TODAY'S DATA ##
    today_df = ticker.history(period="1mo")
    today_df = functions.df_actions(today_df)
    latest_row = today_df.iloc[-1]
    today_madf = latest_row['MADF']

    ## REGRESSION ##
    w = 0
    b = 0
    prev_cost = float('inf')
    x = x_train
    y = y_train
    alpha = 0.01
    epsilon = 1e-6
    max_iters = 2000

    for i in range(max_iters):
        cost = functions.cost_function(w, b, x, y)
        w, b = functions.gradient_descent(w, b, x, y, alpha)
        if abs(prev_cost - cost) < epsilon:
            print(f"Stopped in {i} iterations")
            break
        else:
             prev_cost = cost
             continue

    prediction = (w * today_madf) + b
    print(prediction)

if __name__ == "__main__":
    main()