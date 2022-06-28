import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def render_predict(data: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame):
    data = data.sort_values(by= ['Date'], ascending = True)
    y_true = y_true.sort_values(by= ['Date'], ascending = True)
    y_pred = y_pred.sort_values(by= ['Date'], ascending = True)

    plt.plot(data['Date'][-500:], data['Close'][-500:], color = "blue", label="data")
    plt.plot(y_true['Date'], y_true['Close'], color = "green", label = "true")
    plt.plot(y_pred['Date'], y_pred['Close'], color = "red", label = "pred")
    plt.legend()
    plt.show()

def render_col(df: pd.DataFrame, code: int, col_name: List[str], num_days: int = None):
    if num_days is None:
        num_days = len(df)
    sample = df[df['SecuritiesCode'] == code]
    for col in col_name:
        plt.plot(
            sample['Date'][-num_days :], 
            sample[col_name][-num_days :],
            # label= col
        )

    plt.legend(col_name)
    plt.xlabel("Date")
    plt.ylabel(col_name)
    plt.title('SecuritiesCode: ' + str(code))
    plt.show()
    print(len(sample))

def render_feature_important(model_xgboost, df_trained):
  feature_important = model_xgboost.feature_importances_
  sorted_idx = np.argsort(feature_important)[::-1]

  keys = df_trained.columns[sorted_idx]
  values = feature_important[sorted_idx]

  data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
  data.plot(kind='barh', figsize = (20,10))

def fillInterpolate(_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    assert all([x in _df.columns for x in cols])
    
    df = _df.copy()
    df = df.set_index(['Date'])
    df[cols] = df[cols].interpolate('time', limit_direction='both')
    return df.reset_index()




def DataPreprocessing_for_official_test(_df: pd.DataFrame, df_train: pd.DataFrame):
    df = _df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    df_return = df[['Date', 'SecuritiesCode', 'Volume', 'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag']]
    df_return['ExpectedDividend'] = df_return['ExpectedDividend'].fillna(-1)

    sorted_tmp = pd.DataFrame()
    for code in df['SecuritiesCode'].unique():
        df_tmp = df_train[['SecuritiesCode', 'Date', 'Open', 'High', 'Low', 'Close']][df_train['SecuritiesCode'] == code].sort_values(by=['Date'], ascending= True).iloc[-15:].append(
            df[['SecuritiesCode', 'Date', 'Open', 'High', 'Low', 'Close']][df['SecuritiesCode'] == code])
        df_tmp = df_tmp.sort_values(by=['Date'], ascending= True)
        df_tmp = fillInterpolate(df_tmp, ['Open', 'High', 'Low', 'Close'])
    
        sorted_tmp = sorted_tmp.append(df_tmp)

    df_return = df_return.merge(
        sorted_tmp,
        how='left', on=['SecuritiesCode', 'Date']
    )

    return df_return

def FeatureEgineering_for_official_test(_df: pd.DataFrame, df_train: pd.DataFrame, N: int = 3):
    df = _df.copy()
    df['Range_HL'] = df['High'] - df['Low']
    df['Range_OC'] = df['Open'] - df['Close']
    df.drop(['High', 'Low', 'Open'], axis = 1, inplace= True)

    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['week'] = df['Date'].dt.isocalendar().week
    df['is_month_end'] = df['Date'].dt.is_month_end
    df['is_month_start'] = df['Date'].dt.is_month_start
    df['is_year_end'] = df['Date'].dt.is_year_end
    df['is_year_start'] = df['Date'].dt.is_year_start   
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    df['Volume'] = np.log(df['Volume'])

    info_cols = ['Close', 'Range_HL', 'Range_OC', 'Volume']

    df_train = df_train[['SecuritiesCode', 'Date'] + info_cols]

    df_root = pd.DataFrame()
    for code in df_train['SecuritiesCode'].unique():
        df_root = df_root.append(
            df_train[
                df_train['SecuritiesCode'] == code
            ].sort_values(by=['Date'], ascending= True)[-N:]
        )

    for i in range(N):
        tmp = df_root.append(df[['SecuritiesCode', 'Date'] + info_cols])
        tmp.columns = ['SecuritiesCode', 'Date'] + [col_name + '_' + str(i + 1) +'_before' for col_name in info_cols]

        sorted_tmp = pd.DataFrame()
        for code in tmp['SecuritiesCode'].unique():

            df_tmp = tmp[tmp['SecuritiesCode'] == code]
            df_tmp = df_tmp.sort_values(by=['Date'], ascending= True)

            df_tmp['Date'] = df_tmp['Date'].shift(-(i + 1))
            sorted_tmp = sorted_tmp.append(df_tmp)

        df = df.merge(
            sorted_tmp,
            how='left', on=['SecuritiesCode', 'Date']
        )

        df = fillInterpolate(df, [col_name + '_' + str(i + 1) +'_before' for col_name in info_cols])
    
    df = df.reset_index(drop= True)
    df['Close' + '_mean'] = df[['Close' + '_' + str(i + 1) +'_before' for i in range(N)]].mean(axis= 1)
    df['Close' + '_std'] = df[['Close' + '_' + str(i + 1) +'_before' for i in range(N)]].std(axis= 1).replace(0, 1e-3)
    
    # Scale
    df['Close'] = (df['Close'] - df['Close_mean'])/df['Close_std']
    for i in range(N):
        name = 'Close' + '_' + str(i + 1) +'_before'
        df[name] = (df[name] - df['Close_mean'])/df['Close_std']
    for col_name in ['Range_HL', 'Range_OC']:
        df[col_name] = df[col_name]/df['Close_std']
        for i in range(N):
            name = col_name + '_' + str(i + 1) +'_before'
            df[name] = df[name]/df['Close_std']
    
    
    return df

