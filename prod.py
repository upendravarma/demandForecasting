import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

#read data

train_df = pd.read_csv(r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Data\train.csv')
test_df = pd.read_csv(r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Data\test.csv')

holidaysEvents_df = pd.read_csv(r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Data\holidays_events.csv')
items_df = pd.read_csv(r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Data\items.csv')
oil_df = pd.read_csv(r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Data\oil.csv')
stores_df = pd.read_csv(r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Data\stores.csv')
transactions_df = pd.read_csv(r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Data\transactions.csv')

#Data Cleaning




#Feature Construction

#Classifier functions

class FeatureConstructor:

    def __init__(self, train_df, holidayEvents_df, items_df, oil_df, transactions_df, stores_df):
        """This function initializes the training & current dataframes to be processed
        Training df is used to calculate feature values
        Current df is the dataframe which needs to be featurized
        """
        self.train_df = train_df
        self.holidayEvents_df  = holidayEvents_df
        self.items_df = items_df
        self.oil_df = oil_df
        self.transactions_df = transactions_df
        self.stores_df = stores_df

    def dataCleanup(self, inputData):
        #cleans any type of input data based on specified parameters
        #suggested to use this before using anytype of input data
        return

    def OverallDayWiseTemporal(self):
        """Probabilty that any transaction will occur on a particular day of week
        Based on all of training data with all transactions over all products & places
        """

        #Using transactions to calculate this
        currentTransactions_df = self.transactions_df
        currentTransactions_df['date'] = pd.to_datetime(currentTransactions_df['date'])
        currentTransactions_df['dayOfWeek'] = currentTransactions_df['date'].dt.dayofweek
        overallDayWiseTemporalDist = currentTransactions_df.groupby(by=['dayOfWeek']).mean()

        #MaxAbs Scaler
        self.overallDayWiseTemporalDist_AbsNormal = overallDayWiseTemporalDist.loc[:,['transactions']]
        self.overallDayWiseTemporalDist_AbsNormal['transactions'] = pp.maxabs_scale(self.overallDayWiseTemporalDist_AbsNormal['transactions'])

        #MinMax Scaler
        self.overallDayWiseTemporalDist_MinMaxNormal = overallDayWiseTemporalDist.loc[:, ['transactions']]
        self.overallDayWiseTemporalDist_MinMaxNormal['trnsactions'] = pp.minmax_scale(self.overallDayWiseTemporalDist_MinMaxNormal['transactions'])
        return

    def OverallWeekWiseTemporal(self):

        #using transactions to calculate this
        currentTransactions_df = self.transactions_df
        currentTransactions_df['date'] = pd.to_datetime(currentTransactions_df['date'])
        currentTransactions_df['weekOfMonth'] = currentTransactions_df['date'].dt.day.apply(lambda x: (int)((x-1)/7 +1))
        overallWeekWiseTemporalDist = currentTransactions_df.groupby(by=['weekOfMonth']).mean()

        #MaxAbsScaler
        self.overallWeekWiseTemporalDist_AbsNormal = overallWeekWiseTemporalDist.loc[:, ['transactions']]
        self.overallWeekWiseTemporalDist_AbsNormal['transactions'] = pp.maxabs_scale(self.overallWeekWiseTemporalDist_AbsNormal['transactions'])

        #MinMaxScaler
        self.overallWeekWiseTemporalDist_MinMaxNormal = overallWeekWiseTemporalDist.loc[:, ['transactions']]
        self.overallWeekWiseTemporalDist_MinMaxNormal['transactions'] = pp.minmax_scale(self.overallWeekWiseTemporalDist_MinMaxNormal['transactions'])
        return

    def OverallMonthWiseTemporal(self):

        #Using transactions to calculate this
        currentTransactions_df = self.transactions_df
        currentTransactions_df['date'] = pd.to_datetime(currentTransactions_df['date'])
        currentTransactions_df['month'] = currentTransactions_df['date'].dt.month
        overallMonthWiseTransactions = currentTransactions_df.groupby(by=['month']).mean()

        #MaxAbsScaler
        self.overallMonthWiseTransactions_AbsNormal = overallMonthWiseTransactions.loc[:, ['transactions']]
        self.overallMonthWiseTransactions_AbsNormal['transactions'] = pp.maxabs_scale(self.overallMonthWiseTransactions_AbsNormal['transactions'])

        #MinMaxScaler
        self.overallMonthWiseTransactions_MinMaxNormal = overallMonthWiseTransactions.loc[:, ['transactions']]
        self.overallMonthWiseTransactions_MinMaxNormal['transactions'] = pp.minmax_scale(self.overallMonthWiseTransactions_MinMaxNormal['transactions'])

        return

    def OverallQuarterWiseTemporal(self):
        #Using transactions to calculate this
        currentTransactions_df = self.transactions_df
        currentTransactions_df['date'] = pd.to_datetime(currentTransactions_df['date'])
        currentTransactions_df['quarter'] = currentTransactions_df['date'].dt.quarter
        overallQuarterWiseTransactions = currentTransactions_df.groupby(by=['quarter']).mean()

        #MaxAbsScaler
        self.overallQuarterWiseTransactions_AbsNormal = overallQuarterWiseTransactions.loc[:, ['transactions']]
        self.overallQuarterWiseTransactions_AbsNormal['transactions'] = pp.maxabs_scale(self.overallQuarterWiseTransactions_AbsNormal['transactions'])

        #MinMaxScaler
        self.overallQuarterWiseTransactions_MinMaxNormal = overallQuarterWiseTransactions.loc[:, ['transactions']]
        self.overallQuarterWiseTransactions_MinMaxNormal['transactions'] = pp.minmax_scale(self.overallQuarterWiseTransactions_MinMaxNormal['transactions'])

        return


    def ProductDayWiseTemporal(self):

        #Using sales data from training data to construct the features
        currentTrain_df = self.train_df.loc[:,['date', 'item_nbr', 'unit_sales']]
        currentTrain_df['date'] = pd.to_datetime(currentTrain_df['date'])
        currentTrain_df['dayOfWeek'] = currentTrain_df['date'].dt.dayofweek
        productDayWiseTransactions = currentTrain_df.groupby(by=['item_nbr','dayOfWeek'])['unit_sales'].mean()

        #Template to deal with missing data
        templateDayWiseTransactions = pd.DataFrame(index= pd.MultiIndex.from_product([currentTrain_df['item_nbr'].unique(),np.linspace(0, 6, 7).astype(int)], names=['item_nbr', 'dayOfWeek']))
        productDayWiseTransactions = templateDayWiseTransactions.join(productDayWiseTransactions, how='outer')
        productDayWiseTransactions = productDayWiseTransactions.fillna(value= 0)

        #MaxAbsScaler
        self.productDayWiseTransactions_AbsNormal = productDayWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.maxabs_scale(x))
        #MinMaxScaler
        self.productDayWiseTransactions_MinMaxNormal = productDayWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.minmax_scale(x))

        return

    def ProductWeekWiseTemporal(self):

        #Using sales data from training data to construct the features
        currentTrain_df = self.train_df.loc[:,['date', 'item_nbr', 'unit_sales']]
        currentTrain_df['date'] = pd.to_datetime(currentTrain_df['date'])
        currentTrain_df['weekOfMonth'] = currentTrain_df['date'].dt.day.apply(lambda x: (int)((x-1)/7 +1))
        productWeekWiseTransactions = currentTrain_df.groupby(by=['item_nbr','weekOfMonth'])['unit_sales'].mean()

        #Template to deal with missing data
        templateWeekWiseTransactions = pd.DataFrame(index= pd.MultiIndex.from_product([currentTrain_df['item_nbr'].unique(),np.linspace(1, 5, 5).astype(int)], names=['item_nbr', 'weekOfMonth']))
        productWeekWiseTransactions = templateWeekWiseTransactions.join(productWeekWiseTransactions, how='outer')
        productWeekWiseTransactions = productWeekWiseTransactions.fillna(value= 0)

        #MaxAbsScaler
        self.productWeekWiseTransactions_AbsNormal = productWeekWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.maxabs_scale(x))
        #MinMaxScaler
        self.productWeekWiseTransactions_MinMaxNormal = productWeekWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.minmax_scale(x))

        return

    def ProductMonthWiseTemporal(self):

        #Using sales data from training data to construct the features
        currentTrain_df = self.train_df.loc[:,['date', 'item_nbr', 'unit_sales']]
        currentTrain_df['date'] = pd.to_datetime(currentTrain_df['date'])
        currentTrain_df['monthOfYear'] = currentTrain_df['date'].dt.month
        productMonthWiseTransactions = currentTrain_df.groupby(by=['item_nbr','monthOfYear'])['unit_sales'].mean()

        #Template to deal with missing data
        templateMonthWiseTransactions = pd.DataFrame(index= pd.MultiIndex.from_product([currentTrain_df['item_nbr'].unique(),np.linspace(1, 12, 12).astype(int)], names=['item_nbr', 'monthOfyear']))
        productMonthWiseTransactions = templateMonthWiseTransactions.join(productMonthWiseTransactions, how='outer')
        productMonthWiseTransactions = productMonthWiseTransactions.fillna(value= 0)


        #MaxAbsScaler
        self.productMonthWiseTransactions_AbsNormal = productMonthWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.maxabs_scale(x))
        #MinMaxScaler
        self.productMonthWiseTransactions_MinMaxNormal = productMonthWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.minmax_scale(x))
        return

    def ProductQuarterWiseTemporal(self):
        #Using sales data from training data to construct the features
        currentTrain_df = self.train_df.loc[:,['date', 'item_nbr', 'unit_sales']]
        currentTrain_df['date'] = pd.to_datetime(currentTrain_df['date'])
        currentTrain_df['quarterOfYear'] = currentTrain_df['date'].dt.quarter
        productQuarterWiseTransactions = currentTrain_df.groupby(by=['item_nbr','quarterOfYear'])['unit_sales'].mean()

        #Template to deal with missing data
        templateQuarterWiseTransactions = pd.DataFrame(index= pd.MultiIndex.from_product([currentTrain_df['item_nbr'].unique(),np.linspace(1, 4, 4).astype(int)], names=['item_nbr', 'quarterOfYear']))
        productQuarterWiseTransactions = templateQuarterWiseTransactions.join(productQuarterWiseTransactions, how='outer')
        productQuarterWiseTransactions = productQuarterWiseTransactions.fillna(value= 0)


        #MaxAbsScaler
        self.productQuarterWiseTransactions_AbsNormal = productQuarterWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.maxabs_scale(x))
        #MinMaxScaler
        self.productQuarterWiseTransactions_MinMaxNormal = productQuarterWiseTransactions.groupby(level=['item_nbr']).transform(lambda x: pp.minmax_scale(x))
        
        return

    def FamilyDayWiseTemporal(self):

        #Using sales data from training data
        #Need to normalize within a family for projecting fair number of transactions
        return

    def FamilyWeekWiseTemporal(self):

        return

    def FamilyMonthWiseTemporal(self):

        return

    def FamilyQuarterWiseTemporal(self):

        return

    def ClassDayWiseTemporal(self):

        return

    def ClassWeekWiseTemporal(self):

        return

    def ClassMonthWiseTemporal(self):

        return

    def ClassQuarterWiseTemporal(self):

        return

    #Spatial Features


    #more such feature functions


    def GenerateFeatureConstructors(self):
        #update all the graph variables

        return

    def GenerateFeatureMatrix(self, current_df):
        #takes current_df(train or test) & returns a feature matrix

        #Need to take care of cases where test data has values out of train data graph variables eg. ProductDayWiseTransactions with new product id

        return



#Testing various feature constructors(temporary)
sampleTrain_df = train_df.sample(frac= 0.001)
fc = FeatureConstructor(sampleTrain_df, holidaysEvents_df, items_df, oil_df, transactions_df, stores_df)
fc.ProductDayWiseTemporal()

outputPath = r'C:\Users\upvarma\Documents\ProjectF\DemandForecasting\CorporationFavoritaKaggle\Code\output.txt'
fc.productDayWiseTransactions_AbsNormal.to_csv(outputPath, sep='\t')

#Training the model

