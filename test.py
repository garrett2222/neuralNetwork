from pandas_datareader import data
import pandas as pd
import urllib2
from bs4 import BeautifulSoup as bs


ticker=['AAPL']
dataSource='google'
start="2017-10-18"
end="2017-10-20"

panel_data=data.DataReader(ticker,dataSource,start,end)
close=panel_data.ix['Close']
all_weekdays = pd.date_range(start=start, end=end, freq='B')
close = close.reindex(all_weekdays)

print close[ticker[0]][0]


#print panel_data



def get_short_interest(symbol):
    url = "http://www.nasdaq.com/symbol/" + symbol + "/short-interest"
    res = urllib2.urlopen(url)
    res = res.read()
    soup = bs(res,"lxml")
    si = soup.find("div", {"id": "quotes_content_left_ShortInterest1_ContentPanel"})
    si = si.find("div", {"class": "genTable floatL"})
    df = pd.read_html(str(si.find("table")))[0]
    df.index = pd.to_datetime(df['Settlement Date'])
    del df['Settlement Date']
    df.columns = ['ShortInterest', 'ADV', 'D2C']
    return df['ADV'][0], df['ShortInterest'][0]

#ADV,ShortInt = get_short_interest("aapl")