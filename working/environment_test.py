import pandas as pd 
import numpy as np 
class jpx_tokyo_market_prediction():
    def __init__(self,test_folder =None):
        if test_folder is None:
            self.df = pd.read_csv('../__input__/jpx-tokyo-stock-exchange-prediction/example_test_files/stock_prices.csv')
            self.options_df = pd.read_csv('../__input__/jpx-tokyo-stock-exchange-prediction/example_test_files/options.csv')
            self.secondary_df = pd.read_csv('../__input__/jpx-tokyo-stock-exchange-prediction/example_test_files/secondary_stock_prices.csv')
            self.trades_df =  pd.read_csv('../__input__/jpx-tokyo-stock-exchange-prediction/example_test_files/trades.csv')
            self.financials_df = pd.read_csv('../__input__/jpx-tokyo-stock-exchange-prediction/example_test_files/financials.csv')
            self.pred_df = pd.read_csv('../__input__/jpx-tokyo-stock-exchange-prediction/example_test_files/sample_submission.csv')
        else:
            self.df = pd.read_csv(test_folder+'stock_prices.csv')
            self.options_df = pd.read_csv(test_folder+'options.csv')
            self.secondary_df = pd.read_csv(test_folder +'secondary_stock_prices.csv')
            self.trades_df = pd.read_csv(test_folder+ 'trades.csv')
            self.pred_df = pd.read_csv(test_folder+'sample_submission')
            self.financials_df = pd.read_csv(test_folder+'financials.csv')
        self.dates = self.df['Date'].unique()
        self.iter_index = 0
        self.predict_call = 0 
    def iter_test(self):
        for date in self.dates:
            test_df = self.df[self.df['Date'] == date]
            options = self.options_df[self.options_df['Date'] == date]
            financials = self.financials_df[self.financials_df['DisclosedDate'] == date]
            trades = self.trades_df[self.trades_df['Date'] == date]
            secondary_prices = self.secondary_df[self.secondary_df['Date'] == date]
            df_pred = self.pred_df[self.pred_df['Date'] == date]
            if (self.iter_index == self.predict_call) :
                self.iter_index += 1 
                yield test_df,options,financials,trades,secondary_prices,df_pred
            else:
                print('You must call `predict()` successfully before you can continue with `iter_test()`')
                break
        
    def predict(self,submission):
        self.predict_call += 1
        submission.to_csv('submission.csv', mode='a', header=False)