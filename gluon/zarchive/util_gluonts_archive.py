#https://github.com/arita37/mlmodels/blob/dev/mlmodels/model_gluon/gluonts_model.py
# -*- coding: utf-8 -*-

#### New version
new = True



"""
Advanded GlutonTS models

"""
import os, copy
import pandas as pd, numpy as np

import importlib
import math
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
from pathlib import Path


import mdn

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import l2
from keras.optimizers import Adam

from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.seq2seq import Seq2SeqEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.simple_feedforward import  SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator, WaveNetSampler, WaveNet



from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset,load_datasets
from gluonts.dataset.repository.datasets import get_dataset as get_dataset_gluon

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from tqdm.autonotebook import tqdm



#############################################################################################
##################### CREATE SYNTHETIC TIMESERIES############################################
#############################################################################################
def create_timeseries1d(start_date = '1/1/2000' ,end_date = None ,periods = 1000, freq = 'D' ,weight = 1 , fun=None ):
  
    date_rng   = pd.date_range(start=start_date, end=end_date, freq='D',periods=periods)
    df        = pd.DataFrame(date_rng, columns=['date'])
    data      =[]
    for index in range(0,len(df)):
        data.append(fun(index))  
    df['data']         = data
    df['datetime']     = pd.to_datetime(df['date'])
    df                = df.set_index('datetime')
    df.drop(['date'], axis=1, inplace=True)
    return df


def create_timeseries_2d(start_date = '1/1/2000' ,end_date = None ,periods = 660, freq = 'D' ,weight = 1 , fun=None ):
   pass
   
   
   
def create_timeseries_kd(start_date = '1/1/2000' ,end_date = None ,periods = 660, freq = 'D' ,weight = 1 , params=None ):
    date_rng   = pd.date_range(start=start_date, end=end_date, freq='D',periods=periods)
    df        = pd.DataFrame(date_rng, columns=['date'])
    data      =[]
    Ncols   = len(params)
    for key in params:
      data  = [] 
      fun = params[key]
      for index in range(0,len(df)):
          data.append(fun(index))  
      df[key]     = data
    df['datetime']     = pd.to_datetime(df['date'])
    df                = df.set_index('datetime')
    df.drop(['date'], axis=1, inplace=True)
    return df
   ### k dumsnio







###################################################################################################################
###############PREPARE GLUONTS DATA################################################################################
###################################################################################################################

def gluonts_create_dynamic(df_dynamic, submission=True, single_pred_length=28, submission_pred_length=10, n_timeseries=1, transpose=1) :
    """
        N_cat x N-timseries
    """
    for col in df_dynamic.columns:
        df_dynamic[col]=df_dynamic[col].apply(lambda x: 0 if str(x) == "nan" else 1)
    v   = df_dynamic.values.T if transpose else df_dynamic.values
    
    if submission==True:
      train_cal_feat    = v[:,:-submission_pred_length]
      test_cal_feat     = v
    else:
      train_cal_feat     = v[:,:-submission_pred_length-single_pred_length]
      test_cal_feat     = v[:,:-submission_pred_length]

    #### List of individual time series   Nb Series x Lenght_time_series
    test_list   = [test_cal_feat] * n_timeseries
    train_list  = [train_cal_feat] * n_timeseries
    
    return train_list, test_list


def gluonts_create_static(df_static, submission=1, single_pred_length=28, submission_pred_length=10, n_timeseries=1, transpose=1) :
    """
        N_cat x N-timseries
    """
    static_cat_list=[]
    static_cat_cardinalities=[]
    ####### Static Features 
    for col in df_static :
      
      v_col  = df_static[col].astype('category').cat.codes.values
      static_cat_list.append(v_col)

      _un ,_counts   = np.unique(v_col, return_counts=True)
      static_cat_cardinalities.append(len(_un))

   
    static_cat               = np.concatenate(static_cat_list)
   
    static_cat               = static_cat.reshape(len(static_cat_list), len(df_static.index)).T
    #print(static_cat.shape)
    static_cat_cardinalities=np.array(static_cat_cardinalities)
    #static_cat_cardinalities = [len(df_static[col].unique()) for col in df_static]

    #### Train, Test, cardinalities
    return static_cat, static_cat,static_cat_cardinalities



def gluonts_static_cardinalities(df_static, submission=1, single_pred_length=28, submission_pred_length=10, n_timeseries=1, transpose=1) :
    """
        N_cat x N-timseries
    """
    # static_cat_list=[]
    static_cat_cardinalities=[]
    ####### Static Features 
    for col in df_static :      
      v_col  = df_static[col].astype('category').cat.codes.values
      # static_cat_list.append(v_col)
      _un ,_counts   = np.unique(v_col, return_counts=True)
      static_cat_cardinalities.append(len(_un))

    #static_cat               = np.concatenate(static_cat_list)
    #static_cat               = static_cat.reshape(len(static_cat_list), len(df_static.index)).T
    #print(static_cat.shape)
    static_cat_cardinalities=np.array(static_cat_cardinalities)
    #static_cat_cardinalities = [len(df_static[col].unique()) for col in df_static]

    #### Train, Test, cardinalities
    return static_cat_cardinalities



def gluonts_create_timeseries(df_timeseries, submission=1, single_pred_length=28, submission_pred_length=10, n_timeseries=1, transpose=1) :
    """
        N_cat x N-timseries
    """
    #### Remove Categories colum
    train_target_values = df_timeseries.values

    if submission == True:
        test_target_values = [np.append(ts, np.ones(submission_pred_length) * np.nan) for ts in df_timeseries.values]

    else:
        #### List of individual timeseries
        test_target_values  = train_target_values.copy()
        train_target_values = [ts[:-single_pred_length] for ts in df_timeseries.values]
  
    return train_target_values, test_target_values


#### Start Dates for each time series
def create_startdate(date="2011-01-29", freq="1D", n_timeseries=1):
   start_dates_list = [date for _ in range(n_timeseries)]
   return start_dates_list


def gluonts_create_dataset(train_timeseries_list, start_dates_list, train_dynamic_list,  train_static_list, freq="D" ) :
    from gluonts.dataset.common import load_datasets, ListDataset
    from gluonts.dataset.field_names import FieldName
    
    train_ds = [
        {
            FieldName.TARGET            : target.tolist(),
            FieldName.START             : start,
            FieldName.FEAT_DYNAMIC_REAL : fdr.tolist(),
            FieldName.FEAT_STATIC_CAT   : fsc.tolist()
        } for (target, start, fdr, fsc) in zip(train_timeseries_list,   # list of individual time series
                                               start_dates_list,              # list of start dates
                                               train_dynamic_list,   # List of Dynamic Features
                                               train_static_list)              # List of Static Features 
        ]
    return train_ds



def gluonts_save_to_file(path:Path, data: List[Dict]):
    import os
    print(f"saving time-series into {path}")
    path=os.path.join(path ,"data.json")
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))


def pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static,
                                  pars={'submission':True,'single_pred_length':28,'submission_pred_length':10, 'n_timeseries':1,'start_date':"2011-01-29",'freq':"D"},
                                  path_save=None , return_df=False) :
    ###         NEW CODE    ######################
    submission             = pars['submission']
    single_pred_length     = pars['single_pred_length']
    submission_pred_length = pars['submission_pred_length']
    n_timeseries           = pars['n_timeseries']
    start_date             = pars['start_date']
    freq                   = pars['freq']
    #start_date             = "2011-01-29"
    ##########################################
    train_dynamic_list = [np.array([None])]*n_timeseries
    test_dynamic_list= [np.array([None])]*n_timeseries
    train_static_list= [np.array([None])]*n_timeseries
    test_static_list = [np.array([None])]*n_timeseries
    cardinalities=  np.array([None]) 
  
    
    if len(df_dynamic)>1:
        train_dynamic_list, test_dynamic_list       = gluonts_create_dynamic(df_dynamic, submission=submission, single_pred_length=single_pred_length, 
                                                                         submission_pred_length=submission_pred_length, n_timeseries=n_timeseries, transpose=1)

    #print(train_dynamic_list[1])
    if len(df_static)>1:
        train_static_list, test_static_list,cardinalities   = gluonts_create_static(df_static , submission=submission, single_pred_length=single_pred_length, 
                                                                         submission_pred_length=submission_pred_length, n_timeseries=n_timeseries, transpose=0)
    #print(train_static_list[0])
    
    train_timeseries_list, test_timeseries_list = gluonts_create_timeseries(df_timeseries, submission=submission, single_pred_length=single_pred_length, 
                                                                            submission_pred_length=submission_pred_length, n_timeseries=n_timeseries, transpose=0)
    #print(train_timeseries_list[0])
    start_dates_list = create_startdate(date=start_date, freq=freq, n_timeseries=n_timeseries)

    train_ds = gluonts_create_dataset(train_timeseries_list, start_dates_list, train_dynamic_list, train_static_list, freq=freq ) 
    test_ds  = gluonts_create_dataset(test_timeseries_list,  start_dates_list, test_dynamic_list,  test_static_list,  freq=freq ) 
    
    if path_save :
        gluonts_save_to_file(path_save + "/train/", train_ds)
        gluonts_save_to_file(path_save + "/test/", test_ds)
        with open(path_save+'/metadata.json', 'w') as f:
          f.write(
              json.dumps(
                  {"cardinality":cardinalities.tolist(),
                    "freq":freq,
                    "prediction_length":single_pred_length,       
                  }
                  )
              )
        


    if return_df :
        return train_ds, test_ds, cardinalities


def cols_remove(col1, col_remove):
  return [   r for r in col1 if not r in col_remove ]

###################################################################################
###########Funtions ROLLING FORCAST AND SUPERVISED CLASSIFICATION OF TIMESERIES####
###################################################################################

# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/



# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size=1, nb_epoch=5, lstm_neurons=1,timesteps=1,dense_neurons=1,mdn_output=1,mdn_Nmixes=1):
  X, y = train[:, 0:-1], train[:, -1]
  #print(X.shape)
  X = X.reshape(X.shape[0], timesteps, X.shape[1])
  model = Sequential()
  model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
  model.add(Dense(dense_neurons))
  model.add(mdn.MDN(mdn_output,mdn_Nmixes))

  adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08,   decay=0.0)
  model.compile(loss=mdn.get_mixture_loss_func(mdn_output,mdn_Nmixes),
             optimizer=adam )
  print(model.summary())
  for i in range(nb_epoch):
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    model.reset_states()
  return model

# make a one-step forecast
def forecast_lstm(model, batch_size,timesteps, X):
	X = X.reshape(1, timesteps, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

  
 


####################################################

def test_supervised() :
    pass
      ######################################
    # Test supervised timeseries classification with armdn
    
    

'''
#################################################################################################################
##### Generate Gluonts Data using functions  #####################################################################
#########################################################################################################
def myfun(t):
       return 20*np.sin(2*np.pi *2* t/100)
def myfun1(t):
       return 20*np.sin(2*np.pi *2* t/100)
params={"t1":myfun,
"t2":myfun1
}
df=create_timeseries_kd(params=params)
print(df.head(5))
df_dynamic=pd.DataFrame() 
df_static=pd.DataFrame() 


single_pred_length     = 28
submission_pred_length = single_pred_length * 2
startdate              = "2011-01-29"
freq                   = "D"
n_timeseries           = len(df)
gluonts_datafolder='gluonts_data/kd_timeseries'
submission=False
pars_data              = {'submission':submission,'single_pred_length':single_pred_length, 'submission_pred_length':submission_pred_length,
                         'n_timeseries':n_timeseries   ,'start_date':startdate ,'freq':freq}



df_timeseries = df


##### Generate the Gluonts format dataset
pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static,
                              pars=pars_data,
                              path_save=gluonts_datafolder , return_df=False) 


#### For Model definition
cardinalities = gluonts_static_cardinalities(df_static, submission=1, single_pred_length=28, submission_pred_length=10, n_timeseries=1, transpose=1) 




#################################################################
########EXAMPLE CODE TO RUN ARMDN#######################
#############################################################################


timesteps=1
interval=1
test_values=30
batch_size, nb_epoch, neurons  = 1,10,1
data_folder = "kaggle_data/synthetic_data"
def myfun(t):
       return 20*np.sin(2*np.pi *2* t/100)
df=create_timeseries1d(fun= myfun)
df.to_csv(data_folder+'/cos_timeseries.csv')
series = pd.read_csv(data_folder+'/cos_timeseries.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values,interval)
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, timesteps)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-test_values], supervised_values[-test_values:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, batch_size, nb_epoch,neurons,timesteps)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), timesteps,1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
  # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, batch_size,timesteps, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+interval-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i +1]
    print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))


# report performance
rmse = math.sqrt(mean_squared_error(raw_values[-test_values:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[-test_values:])
plt.plot(predictions)
plt.show()

'''

if __name__ == '__main__':
   VERBOSE = False
   import numpy as np
   def myfun(t):
       return 20*np.sin(2*np.pi *2* index/100)
       
   df=create_timeseries1d(fun= myfun)
   print(df.head(5))


    #test()
