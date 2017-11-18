import pandas as pd
import datetime
import time

bikes_df = pd.read_csv('Bikes.csv')
weather_df = pd.read_csv('Weather.csv')
holidays_df = pd.read_csv('Holidays.csv')

# Convert the holidays_df date format to %d/%m/%Y
holidays_df['Date'] = holidays_df.Date.apply(lambda x:
        time.strftime("%d/%m/%Y", time.localtime(x)))

# Weather_df has the most complete data.
cleaned_df = weather_df
cleaned_df = pd.merge(cleaned_df, bikes_df, on=['Date'])
cleaned_df = pd.merge(cleaned_df, holidays_df, on=['Date'], how='outer')

# Fill holiday NaNs with "No"
cleaned_df.Holiday = cleaned_df.Holiday.fillna("No")
cleaned_df.index.name = "ID"

# Get month/day of week cols
cleaned_df['Month'] = cleaned_df.Date.apply(lambda x: int(x.split("/")[1]))
cleaned_df['Weekday'] = cleaned_df.Date.apply(lambda x:
        (int(datetime.date(int(x.split("/")[2]),
            int(x.split("/")[1]),
            int(x.split("/")[0])).strftime("%w"))+6) % 7)

cleaned_df.to_csv("train.csv")
