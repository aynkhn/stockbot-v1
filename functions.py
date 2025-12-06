def df_actions(df):
    df = df[['High']].copy() # drop everything but high values
    df['MA5'] = df['High'].rolling(window=5).mean() # MA's
    df['MA20'] = df['High'].rolling(window=20).mean()
    df['MADF'] = df['MA5'] - df['MA20'] # MA difference
    df['Target'] = df['High'].shift(-1)
    return df

def cost_function(w, b, x, y):
    m = len(y)
    sigma = 0
    for i in range(m):
        sigma += (y.iloc[i] - ((w * x.iloc[i]) + b)) ** 2
    cost = sigma / (2 * m)
    return cost

def partial_derivatives(w, b, x, y):
    m = len(y)
    dw_sum = 0.0
    db_sum = 0.0
    for i in range(m):
            error = (y.iloc[i] - ((w * x.iloc[i]) + b))
            dw_sum += (y.iloc[i] - ((w * x.iloc[i]) + b)) * x.iloc[i]   # derivative w.r.t w
            db_sum += error          # derivative w.r.t b
    dw = -dw_sum / m
    db = -db_sum / m
    return dw, db

def gradient_descent(w, b, x, y, alpha):
    dw, db = partial_derivatives(w, b, x, y)
    tmp_w = w - (alpha * dw)
    tmp_b = b - (alpha * db)
    w = tmp_w
    b = tmp_b
    return w, b