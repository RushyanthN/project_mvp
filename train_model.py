import json
import pickle

from utils.b2 import B2
from sklearn.linear_model import LinearRegression


with open('./config_vars.json') as f:
    config_vars = json.load(f)

b2 = B2(config_vars['B2_ENDPOINT'], 
        config_vars['B2_KEYID'],
        config_vars['B2_APPKEY'])

b2.set_bucket(config_vars['B2_BUCKETNAME'])

data = b2.to_df('seattle_home_prices/SeattleHomePrices_2.csv')
data = data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
clean_data = data.dropna(subset=['PRICE', 'SQUARE FEET', 'BEDS'])

X = clean_data[['SQUARE FEET', 'BEDS']]
y = clean_data['PRICE']

lm = LinearRegression()

lm.fit(X, y)

with open('model.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(lm, f, pickle.HIGHEST_PROTOCOL)