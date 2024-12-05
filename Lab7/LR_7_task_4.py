import datetime 
import json 
import numpy as np 
from sklearn import covariance, cluster 
import pandas as pd
import yfinance as yf 
from sklearn.impute import SimpleImputer

# Load company symbols
input_file = 'company_symbol_mapping.json' 
with open(input_file, 'r') as f: 
    company_symbols_map = json.loads(f.read()) 
symbols = list(company_symbols_map.keys())

# Set date range
start_date = "2003-07-03"
end_date = "2007-05-04"

# Load stock data
quotes = {}
for symbol in symbols:
    try:
        print(f"Loading {symbol} ({company_symbols_map[symbol]})...", end='')
        q = yf.download(symbol, start=start_date, end=end_date)
        if not q.empty:
            quotes[symbol] = q
            print("done.")
        else:
            print("no data.")
    except Exception as e:
        print(f"error: {e}")

# Align all data to the same index (date range)
aligned_quotes = pd.concat(quotes.values(), keys=quotes.keys(), names=["Symbol", "Date"])
aligned_quotes = aligned_quotes.unstack(level=0)  # Organize by symbol

# Extract opening and closing quotes
opening_quotes = aligned_quotes['Open'].values  # Shape: (time, stocks)
closing_quotes = aligned_quotes['Close'].values  # Shape: (time, stocks)

# Compute differences
quotes_diff = closing_quotes - opening_quotes

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # Replace NaN with column mean
quotes_diff_imputed = imputer.fit_transform(quotes_diff)

# Normalize data
X = quotes_diff_imputed.copy()
X /= X.std(axis=0)  # Normalize each stock's data

# Graphical Lasso for covariance estimation
edge_model = covariance.GraphicalLassoCV(cv=3)

# Train model
with np.errstate(invalid='ignore'): 
    edge_model.fit(X)

# Perform clustering
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Output clustering results
names = np.array([company_symbols_map[symbol] for symbol in quotes.keys()])
for i in range(num_labels + 1): 
    print("Cluster", i+1, "==>", ', '.join(names[labels == i]))
