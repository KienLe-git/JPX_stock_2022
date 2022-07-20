import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def train_valid_test_split(df: pd.DataFrame):
    df_train = df[df['Date'] <= '2021-05-27'].reset_index(drop= True)

    df_valid = df[
        (df['Date'] > '2021-05-27') & (df['Date'] <= '2021-12-03')
    ].reset_index(drop= True)
    
    df_test = df[df['Date'] >= '2021-12-06'].reset_index(drop= True)
    return df_train, df_valid, df_test

def render_col(df: pd.DataFrame, code: int, col_name: str, num_days: int = None):
    if num_days is None:
        num_days = len(df)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sample = df[df['SecuritiesCode'] == code]
    ax.plot(
        sample['Date'][-num_days :], 
        sample[col_name][-num_days :],
    )

    ax.legend(col_name)
    ax.set_xlabel("Date")
    ax.set_title('SecuritiesCode: ' + str(code))
    ax.set_xticks([sample['Date'].iloc[0], sample['Date'].iloc[-1]])

    plt.show()
    print(len(sample))

def render_feature_important(model_xgboost, df_trained):
  feature_important = model_xgboost.feature_importances_
  sorted_idx = np.argsort(feature_important)[::-1]

  keys = df_trained.columns[sorted_idx]
  values = feature_important[sorted_idx]

  data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
  data.plot(kind='barh', figsize = (20,10))

def fillInterpolate(_df: pd.DataFrame, cols: List[str], limit_direction='both', **kwargs) -> pd.DataFrame:
    assert all([x in _df.columns for x in cols])
    
    df = _df.copy()
    df = df.set_index(['Date'])
    df[cols] = df[cols].interpolate('time', limit_direction= limit_direction, **kwargs)
    return df.reset_index()


def DataPreprocessing_for_HiddenTest(_df: pd.DataFrame, df_base: pd.DataFrame):
    df = _df.copy()

    last_base_date = df_base['Date'].max()

    df = df.append(df_base, ignore_index= True)
    df['ExpectedDividend'].fillna(-1, inplace= True)

    sorted_tmp = df.reset_index().groupby(by=['SecuritiesCode']).apply(
        lambda df_code: fillInterpolate(df_code[
            ['index', 'Date', 'Open', 'High', 'Low', 'Close']
        ], ['Open', 'High', 'Low', 'Close'], limit_direction= 'both')
    ).reset_index(drop= True).set_index('index')[['Open', 'High', 'Low', 'Close']]

    df[['Open', 'High', 'Low', 'Close']] = sorted_tmp[['Open', 'High', 'Low', 'Close']]

    return df[df['Date'] > last_base_date].reset_index(drop= True).drop(['Target', 'CloseT1', 'CloseT2', 'RowId'], axis = 1)


def FeatureEngineering_for_HiddenTest(_df: pd.DataFrame, df_train: pd.DataFrame, N: int = 3):
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

def render_predict(data: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame):
    data = data.sort_values(by= ['Date'], ascending = True)
    y_true = y_true.sort_values(by= ['Date'], ascending = True)
    y_pred = y_pred.sort_values(by= ['Date'], ascending = True)

    plt.plot(data['Date'][-500:], data['Close'][-500:], color = "blue", label="data")
    plt.plot(y_true['Date'], y_true['Close'], color = "green", label = "true")
    plt.plot(y_pred['Date'], y_pred['Close'], color = "red", label = "pred")
    plt.legend()
    plt.show()

def calc_spread_return_per_day(df, portfolio_size: int = 200, toprank_weight_ratio: float = 2):
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): spread return
    """
    assert df['Rank'].min() == 0
    assert df['Rank'].max() == len(df['Rank']) - 1
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
    short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
    return purchase - short

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    buf = df.groupby('Date').apply(calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio, buf

def add_rank(df: pd.DataFrame, y_pred):
    df["Pred"] = y_pred
    df["Rank"] = df.groupby("Date")["Pred"].rank(ascending=False, method="first") - 1 
    df = df.drop("Pred", axis= 1)
    return df

def calc_score(df: pd.DataFrame, y_pred: pd.DataFrame, y_true: pd.DataFrame, render_info= True):
    feature_df = df.copy()
    feature_df = add_rank(feature_df, y_pred)
    feature_df['Target'] = y_true
    score, buf = calc_spread_return_sharpe(feature_df)
    if render_info:
        print(f'score -> {score}\nmean -> {buf.mean()}\nstd -> {buf.std()}')
    return score