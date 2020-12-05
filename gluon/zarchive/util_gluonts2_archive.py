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


from time import time
import os, sys, copy,gc, random, json, glob
from collections import OrderedDict


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


import sklearn as sk
from sklearn import manifold, datasets

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_iris


from sklearn.linear_model import LinearRegression,ElasticNet, RidgeCV, RANSACRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (confusion_matrix,roc_curve, roc_auc_score, classification_report,
                             accuracy_score)


 
##### util #####################################################################################
def log(*s) :
    print(*s, flush=True)

def tag_create(pars, sep=";")  :
    s = [  str(k) + "-"+ str(v) for k,v in pars.items()   ]
    s =  ";".join(s)
    return s


##### Conversion ################################################################################
def generate_pivot_unit(df, keyref=None,  **kw) :
    """item_id X Date  : Unit
       generate__file(    )    
    """
    col_merge = [ "shop_id", "item_id"  ]

    df['time_key'] = df['time_key'].astype("int32")                    
    dfp            = df.pivot_table(values='units', index= col_merge  , 
                             columns='time_key', aggfunc='sum').reset_index().fillna(0)    
    
    ########### Cols reference
    colref  = [  "shop_id", "dept_id", "l1_genre_id", "l2_genre_id", 'item_id'   ]    
    key_all = df[ colref].drop_duplicates(colref) 
    del df ; gc.collect()
        
    ########### Cols   ##############################################################
    dfp         = dfp.join( key_all.set_index(col_merge),  on= col_merge, how="left"   )
    coldate     = [ x for x in dfp.columns if x not in colref ]
    dfp         = dfp[ colref + coldate ]
    dfp.columns = [ str(x) for x in dfp.columns ]
    # dfp = dfp[dfp['l1_genre_id'] != -1 ] 
    return dfp


def generate_pivot_gluonts(path_input="/data/pos/", path_export=None, folder_list=None, cols=None,  
                           prefix_col="", prefix_file="pivot-gluonts", shop_list=[16,17], verbose=1,  **kw) :
    
    def isint(t):
        try : int(t)
        except : return False
        return True
        
    dfp2 = None
    for ii, (date0) in enumerate(folder_list) :    
        pos_dir  =  path_input + f"/{date0}"
        log("Process", ii, pos_dir)
        
        dfpi    = pd_read_file2( pos_dir , n_pool=1)
        log_pd( dfpi)
        keys =['shop_id',  "l1_genre_id", "l2_genre_id", 'item_id', ]
        dfp2 = dfp2.join( dfpi.set_index(keys), on=keys, how="outer", rsuffix="b" )  if dfp2 is not None else dfpi
        log_pd( dfp2)
        del dfpi; gc.collect()


    #### df_date ###########################################################
    date_list        = sorted([ int(t)  for t in dfp2.columns  if  isint(t)  ])
    cols_timeseries  = [ t  for t in date_list   ]
    dfdate           = pd.DataFrame({"time_key" : date_list})
    dfdate['order_date'] = [ from_timekey(t)  for t in date_list]
    log_pd( dfdate)
    dfdate = generate_X_date(dfdate, keyref= 'time_key', coldate =  'order_date', prefix_col ="" )
    log_pd( dfdate)
    
    
    ### Split by shop #####################################################
    for shop_id in shop_list :           
        dfp  = dfp2[dfp2.shop_id == shop_id]
        dfp  = dfp.reset_index(drop=True)
        path = f"{path_export}/{prefix_file}_{shop_id}"

    
        # cols_timeseries =  cols_remove( dfp.columns , colref )  
        df_timeseries = dfp[ cols_timeseries].fillna(0.0)
        log_pd(df_timeseries)
        pd_to_file(df_timeseries, f"{path}/df_timeseries.parquet", "none"  )
        
        cols_calendar   =  [ 'day', 'month',  'quarter', 'weekday', 'weekmonth',  'weekyeariso', 'isholiday' ]  
        df_dynamic      = dfdate[ cols_calendar]
        log_pd(df_dynamic)
        pd_to_file(df_dynamic, f"{path}/df_dynamic.parquet", "none"  )
    
        cols_static   = [  "shop_id",  "l1_genre_id", "l2_genre_id", 'item_id'   ]   
        df_static     = dfp[ cols_static]
        pd_to_file(df_static, f"{path}/df_static.parquet", "none"  )
        del dfp; gc.collect()
        
        submission = False
        start_date             = str(dfdate.date.min())
        pars_data              = {'submission'         : submission,
                                  'single_pred_length' : 28,   'submission_pred_length': 2*28,
                                  'n_timeseries': len(df_timeseries)   , 'start_date':  start_date,
                                  'freq': "D"}
        log(pars_data)
        
        ##### Generate the Gluonts format dataset
        pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static,
                                      pars = pars_data,
                                      path_save= f"{path}/"  , return_df=False) 

    


#### Model metrics ######################################################################
def model_eval(estimator=None, TD=None, cardinalities=None,
               istrain=True, ismetric=True, isplot=True, pars=None) :
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.trainer import Trainer
    p = pars
    
    if estimator is None :
        estimator = DeepAREstimator(
            prediction_length     = p.get("single_pred_length", 28),
            freq                  = "D",
            distr_output          = p.get("distr_output", None),
            use_feat_static_cat   = True,
            use_feat_dynamic_real = True,
            cardinality           = p.get("cardinality", None),
            trainer               = Trainer(
            learning_rate         = p.get("lr",   1e-4),  # 1e-4,  #1e-3,
            epochs                = p.get("epoch",   None),
            num_batches_per_epoch = p.get("num_batches_per_epoch",   10),
            batch_size            = p.get("batch_size",   8),
            )
        )
    if istrain : estimator = estimator.train(TD.train)
        
    #### Evaluate  ########################################################################
    from gluonts.evaluation.backtest import make_evaluation_predictions
    forecast_it, ts_it = make_evaluation_predictions(dataset=TD.test, predictor=estimator,  
                                                     num_samples= p.get("num_samples", 5))
    forecasts,tss = list(forecast_it), list(ts_it) 

    if isplot :   
        forecast_graph( forecasts, tss, p.get("ii_series",0) )
      
    ####### Metrics ######################################################################
    agg_metrics, item_metrics = None, None
    if ismetric :
         agg_metrics, item_metrics = forecast_metrics(tss, forecasts, TD, 
                                     quantiles=[0.1, 0.5, 0.9], show=True, dir_save=None) 

    return  estimator,forecasts, tss, agg_metrics, item_metrics



def model_eval_all(dataset_name, estimator):
    dataset = get_dataset(dataset_name)
    estimator = estimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        use_feat_static_cat=True,
        cardinality=[
            feat_static_cat.cardinality
            for feat_static_cat in dataset.metadata.feat_static_cat
        ],
    )

    log(f"evaluating {estimator} on {dataset}")

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    pprint.plog(agg_metrics)

    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    return eval_dict



def forecast_metrics(tss, forecasts, quantiles=[0.1, 0.5, 0.9], show=True, dir_save=None) :
    from gluonts.evaluation import Evaluator
    evaluator = Evaluator(quantiles=quantiles)
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(forecasts ))

    if show :  log(json.dumps(agg_metrics, indent=4))        
    if dir_save :
      json.dump(agg_metrics, indent=4)
      
    return agg_metrics, item_metrics



def forecast_graph( forecasts, tss, ii_series ):
    #ii_series = 0
    forecast_entry = forecasts[ii_series]  
    ts_entry       = tss[ii_series]
    log(f"Nb sample paths: {forecast_entry.num_samples}")
    log(f"Dim of samples: {forecast_entry.samples.shape}")
    log(f"Start date  forecast window: {forecast_entry.start_date}")
    log(f"Time SeriesFrequency : {forecast_entry.freq}")
    plot_prob_forecasts(ts_entry, forecast_entry)


def plot_prob_forecasts(ts_entry, forecast_entry, plot_length = 90, prediction_intervals =  (0.10,) ):
    # plot_length = 90
    # prediction_intervals =  (0.10,)  # (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def cols_remove(col1, col_remove):
  return [   r for r in col1 if not r in col_remove ]


def plot_util():
    import tqdm
    plot_log_path = "./plots/"
    directory = os.path.dirname(plot_log_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    def plot_prob_forecasts(ts_entry, forecast_entry, path, sample_id, inline=True):
        plot_length = 150
        prediction_intervals = (50, 67, 95, 99)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    
        _, ax = plt.subplots(1, 1, figsize=(10, 7))
        ts_entry[-plot_length:].plot(ax=ax)
        forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
        ax.axvline(ts_entry.index[-prediction_length], color='r')
        plt.legend(legend, loc="upper left")
        if inline:
            plt.show()
            plt.clf()
        else:
            plt.savefig('{}forecast_{}.pdf'.format(path, sample_id))
            plt.close()
    
    print("Plotting time series predictions ...")
    for i in tqdm(range(5)):
        ts_entry = tss[i]
        forecast_entry = forecasts[i]
        plot_prob_forecasts(ts_entry, forecast_entry, plot_log_path, i)


######################################################################################
#### Gluonts dataset##################################################################
def gluonts_get_cardinality(dataset_path="", TD=None):
  if TD is None :   
    cc = json.load(open( dataset_path + "/metadata.json", mode="r" ) )['cardinality']
  else :
    cc = [feat_static_cat.cardinality for feat_static_cat in TD.metadata.feat_static_cat ]
  return cc     
      


def gluonts_dataset_check(TD, nline=3) :
  path  = TD.train.path
  flist = glob.glob( path + "/*.json")
  fpath = flist[0]

  sw = "["
  with open(fpath, mode='r') as fp :
    for ii in range(nline) :
       w  = fp.readline()
       sw = sw + w.replace("\n", "") + "," 
       
  sw = sw[:-1] + "]"     
  #return sw     
  dd = json.loads(sw)       
  return dd

        
def gluonts_create_dynamic(df, submission=True, single_pred_length=28, submission_pred_length=10, 
                           n_timeseries=1, transpose=1) :
    """
        N_cat x N-timseries
        
        size of Dynamic == Length of time series.
        
        File "C:\D\anaconda3\envs\py36\lib\site-packages\numpy\core\shape_base.py", line 283, in vstack
            return _nx.concatenate([atleast_2d(_m) for _m in tup], 0)    
        ValueError: all the input array dimensions except for the concatenation axis must match exactly    
        https://github.com/awslabs/gluon-ts/issues/468        
        
        I think the issue is that when you run on the test in the way above, 
        the predict method would expect that your target has some length n and the dynamic features n + prediction length.

        
    """
    df_dynamic = df
    
    """
    def to_flag(x) :
        try :
            a = str(x)
            if  "nan" in a : return 0
            return 1
        except : 
            return 0    
        
    dd = dict(df_dynamic.dtypes)
    
    for col in df_dynamic.columns:
        if "object" in str(dd[col]) :
            print("to One Hot", col)
            df_dynamic[col]=df_dynamic[col].apply(lambda x: to_flag(x) )
    """
    for col in df_dynamic.columns:    
       df_dynamic[col] = df_dynamic[col].factorize()[0]   ### Encode the columns
    
    v = df_dynamic.values.T if transpose else df_dynamic.values
    
    if submission==True:
      train_cal_feat = v[:,:-submission_pred_length]
      test_cal_feat  = v
    
    else:
      # train_cal_feat = v[:,:-submission_pred_length-single_pred_length]
      # test_cal_feat  = v[:,:-submission_pred_length]

      ### Dynamic and time series should be SAME !!
      train_cal_feat = v[:,:-single_pred_length]
      test_cal_feat  = v[:,:]

    
    #### List of individual time series   Nb Series x Lenght_time_series
    test_list  = [test_cal_feat] * n_timeseries
    train_list = [train_cal_feat] * n_timeseries
    
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



def gluonts_static_cardinalities(df_static, submission=1, single_pred_length=28, submission_pred_length=10, 
                                 n_timeseries=1, transpose=1) :
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
    return list(static_cat_cardinalities)



def gluonts_create_timeseries(df_timeseries, submission=1, single_pred_length=28, submission_pred_length=10, 
                              n_timeseries=1, transpose=1) :
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



def gluonts_create_startdate(date="2011-01-29", freq="1D", n_timeseries=1):
   #### Start Dates for each time series 
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



def gluonts_save_to_file(path="", data=None):
    import os
    print(f"saving time-series into {path}")
    path=os.path.join(path ,"data.json")
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))


#### Gluonts Converter ######################################################################
def pandas_to_gluonts(df_timeseries, df_dynamic, df_static,
                                  pars={'submission':True,'single_pred_length':28,'submission_pred_length':10, 
                                        'n_timeseries':1,'start_date':"2011-01-29",'freq':"D"},
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

    train_dynamic_list, test_dynamic_list       = gluonts_create_dynamic(df_dynamic, submission=submission, single_pred_length=single_pred_length, 
                                                                         submission_pred_length=submission_pred_length, n_timeseries=n_timeseries, transpose=1)

    #print(train_dynamic_list[1])
    train_static_list, test_static_list,cardinalities   = gluonts_create_static(df_static , submission=submission, single_pred_length=single_pred_length, 
                                                                         submission_pred_length=submission_pred_length, n_timeseries=n_timeseries, transpose=0)
    #print(train_static_list[0])

    train_timeseries_list, test_timeseries_list = gluonts_create_timeseries(df_timeseries, submission=submission, single_pred_length=single_pred_length, 
                                                                            submission_pred_length=submission_pred_length, n_timeseries=n_timeseries, transpose=0)
    #print(train_timeseries_list[0])
    start_dates_list = gluonts_create_startdate(date=start_date, freq=freq, n_timeseries=n_timeseries)

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


            
def gluonts_to_pandas(dataset_path=None):
  from gluonts.dataset.common import ListDataset,load_datasets  
  all_targets = []
  all_dynamic = []
  all_static  = []
  start       = []
  TD             =load_datasets(   metadata=dataset_path,
                                    train=dataset_path / "train", test=dataset_path / "test")
  instance=next(iter(TD.test))
  #### load decode pars ########
  decode_pars = json.load(open(dataset_path / "decode.json", mode='r'),object_hook=lambda d: {int(k) 
                         if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
  df_dynamic_labels   = decode_pars["df_dynamic_labels"]
  df_dynamic_cols     = decode_pars["df_dynamic_cols"]
  df_static_labels    = decode_pars["df_static_labels"]
  df_static_cols      = decode_pars["df_static_cols"]
  df_timeseries_cols  = decode_pars["df_timeseries_cols"]
  df_timeseries_dtype = decode_pars["df_timeseries_dtype"]
  df_dynamic_dtype    = decode_pars["df_dynamic_dtype"]
  df_static_dtype     = decode_pars["df_static_dtype"]

  #################################################
 
  dynamic_features=np.transpose(instance["feat_dynamic_real"])
  for items in TD.test:
    #print(items)
    target=np.transpose(items["target"]).tolist() 
    static= np.transpose(items["feat_static_cat"]).tolist()
  
    all_static.append(static)
    all_targets.append(target)
  del TD
  df_timeseries =pd.DataFrame(all_targets)
  del all_targets
  df_dynamic =pd.DataFrame(dynamic_features)
  df_static =pd.DataFrame(all_static)

  ################ decode  df_dynamic #####
  if  df_dynamic_labels is not None:
    for key in  df_dynamic_labels:
      col = key
      labels= df_dynamic_labels[key]
      for l in labels:
        v = labels[l]
        df_dynamic[col]=df_dynamic[col].apply(lambda x: v if x == l else x)
  for col in df_dynamic.columns:       
    df_dynamic[col]=df_dynamic[col].apply(lambda x: np.NAN if x == -l else x)  
  if  df_dynamic_cols is not None:
    df_dynamic.rename(columns =  df_dynamic_cols, inplace = True) 


  ##### decode df_timeseries#####
  if  df_timeseries_cols is not None:
    df_timeseries.rename(columns = df_timeseries_cols , inplace = True) 
  del all_dynamic

  ####### decode df staic################
  if df_static_labels  is not None:
       for key in df_static_labels: 
         d =  df_static_labels[key]
         df_static[key] = df_static[key].map(d)
  if  df_static_cols is not None:
      df_static.rename(columns = df_static_cols, inplace = True) 

  #####################
  del all_static
  df_timeseries = df_timeseries.astype(df_timeseries_dtype)
  df_dynamic    = df_dynamic.astype(df_dynamic_dtype)
  df_static     = df_static.astype(df_static_dtype)
  
  return df_timeseries,df_dynamic,df_static




def pd_difference(df1, df2):
  """Identify differences between two pandas DataFrames
    
  """
  if (df1.columns != df2.columns).any(0):
    print("DataFrame column names are different")
    return None
  if any(df1.dtypes != df2.dtypes):
        print("Data Types are different, trying to convert")
        df2 = df2.astype(df1.dtypes)
  if df1.equals(df2):
        print("Exactly Same")
        return None
  else:
        df1=df1.fillna(-1)
        df2=df2.fillna(-1)
        # need to account for np.nan != np.nan returning True
        diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
                            index=changed.index)



def load_datasset_m5():
    #load data  from SHARED FOLDER
    data_folder = "kaggle_data/m5_dataset"
    gluonts_datafolder='gluonts_data/m5_dataset_new'

    calendar               = pd.read_csv(data_folder+'/calendar.csv')[0:100]
    sales_train_val        = pd.read_csv(data_folder+'/sales_train_validation.csv.zip')[0:100]
    sample_submission      = pd.read_csv(data_folder+'/sample_submission.csv.zip')
    sell_prices            = pd.read_csv(data_folder+'/sell_prices.csv.zip')[0:100]

    ##### Generate Data using functions  ############################################################################
    submission = False
    single_pred_length     = 28
    submission_pred_length = single_pred_length * 2
    startdate              = "2011-01-29"
    freq                   = "D"
    n_timeseries           = len(sales_train_val)
    pars_data              = {'submission':submission,'single_pred_length':single_pred_length, 'submission_pred_length':submission_pred_length,
                             'n_timeseries':n_timeseries   ,'start_date':startdate ,'freq':freq}

    cols_timeseries =  cols_remove(sales_train_val.columns, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])   ### Add mamnually  id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2,d_3,d_4,d_5,d_6,d_7,d_8,d_9,d_10,d_11,d_12,d_13,d_14,d_15,d_16,d_17,d_18,d_19,d_20,d_21,d_22,d_23,d_24,d_25,d_26,d_27,d_28,d_29,d_30,d_31,d_32,d_33,d_34,d_35,d_36,d_37,d_38,d_39,d_40,d_41,d_42,d_43,d_44,d_45,d_46,d_47,d_48,d_49,d_50,d_51,d_52,d_53,d_54,d_55,d_56,d_57,d_58,d_59,d_60,d_61,d_62,d_63,d_64,d_65,d_66,d_67,d_68,d_69,d_70,d_71,d_72,d_73,d_74,d_75,d_76,d_77,d_78,d_79,d_80,d_81,d_82,d_83,d_84,d_85,d_86,d_87,d_88,d_89,d_90,d_91,d_92,d_93,d_94,d_95,d_96,d_97,d_98,d_99,d_100,d_101,d_102,d_103,d_104,d_105,d_106,d_107,d_108,d_109,d_110,d_111,d_112,d_113,d_114,d_115,d_116,d_117,d_118,d_119,d_120,d_121,d_122,d_123,d_124,d_125,d_126,d_127,d_128,d_129,d_130,d_131,d_132,d_133,d_134,d_135,d_136,d_137,d_138,d_139,d_140,d_141,d_142,d_143,d_144,d_145,d_146,d_147,d_148,d_149,d_150,d_151,d_152,d_153,d_154,d_155,d_156,d_157,d_158,d_159,d_160,d_161,d_162,d_163,d_164,d_165,d_166,d_167,d_168,d_169,d_170,d_171,d_172,d_173,d_174,d_175,d_176,d_177,d_178,d_179,d_180,d_181,d_182,d_183,d_184,d_185,d_186,d_187,d_188,d_189,d_190,d_191,d_192,d_193,d_194,d_195,d_196,d_197,d_198,d_199,d_200,d_201,d_202,d_203,d_204,d_205,d_206,d_207,d_208,d_209,d_210,d_211,d_212,d_213,d_214,d_215,d_216,d_217,d_218,d_219,d_220,d_221,d_222,d_223,d_224,d_225,d_226,d_227,d_228,d_229,d_230,d_231,d_232,d_233,d_234,d_235,d_236,d_237,d_238,d_239,d_240,d_241,d_242,d_243,d_244,d_245,d_246,d_247,d_248,d_249,d_250,d_251,d_252,d_253,d_254,d_255,d_256,d_257,d_258,d_259,d_260,d_261,d_262,d_263,d_264,d_265,d_266,d_267,d_268,d_269,d_270,d_271,d_272,d_273,d_274,d_275,d_276,d_277,d_278,d_279,d_280,d_281,d_282,d_283,d_284,d_285,d_286,d_287,d_288,d_289,d_290,d_291,d_292,d_293,d_294,d_295,d_296,d_297,d_298,d_299,d_300,d_301,d_302,d_303,d_304,d_305,d_306,d_307,d_308,d_309,d_310,d_311,d_312,d_313,d_314,d_315,d_316,d_317,d_318,d_319,d_320,d_321,d_322,d_323,d_324,d_325,d_326,d_327,d_328,d_329,d_330,d_331,d_332,d_333,d_334,d_335,d_336,d_337,d_338,d_339,d_340,d_341,d_342,d_343,d_344,d_345,d_346,d_347,d_348,d_349,d_350,d_351,d_352,d_353,d_354,d_355,d_356,d_357,d_358,d_359,d_360,d_361,d_362,d_363,d_364,d_365,d_366,d_367,d_368,d_369,d_370,d_371,d_372,d_373,d_374,d_375,d_376,d_377,d_378,d_379,d_380,d_381,d_382,d_383,d_384,d_385,d_386,d_387,d_388,d_389,d_390,d_391,d_392,d_393,d_394,d_395,d_396,d_397,d_398,d_399,d_400,d_401,d_402,d_403,d_404,d_405,d_406,d_407,d_408,d_409,d_410,d_411,d_412,d_413,d_414,d_415,d_416,d_417,d_418,d_419,d_420,d_421,d_422,d_423,d_424,d_425,d_426,d_427,d_428,d_429,d_430,d_431,d_432,d_433,d_434,d_435,d_436,d_437,d_438,d_439,d_440,d_441,d_442,d_443,d_444,d_445,d_446,d_447,d_448,d_449,d_450,d_451,d_452,d_453,d_454,d_455,d_456,d_457,d_458,d_459,d_460,d_461,d_462,d_463,d_464,d_465,d_466,d_467,d_468,d_469,d_470,d_471,d_472,d_473,d_474,d_475,d_476,d_477,d_478,d_479,d_480,d_481,d_482,d_483,d_484,d_485,d_486,d_487,d_488,d_489,d_490,d_491,d_492,d_493,d_494,d_495,d_496,d_497,d_498,d_499,d_500,d_501,d_502,d_503,d_504,d_505,d_506,d_507,d_508,d_509,d_510,d_511,d_512,d_513,d_514,d_515,d_516,d_517,d_518,d_519,d_520,d_521,d_522,d_523,d_524,d_525,d_526,d_527,d_528,d_529,d_530,d_531,d_532,d_533,d_534,d_535,d_536,d_537,d_538,d_539,d_540,d_541,d_542,d_543,d_544,d_545,d_546,d_547,d_548,d_549,d_550,d_551,d_552,d_553,d_554,d_555,d_556,d_557,d_558,d_559,d_560,d_561,d_562,d_563,d_564,d_565,d_566,d_567,d_568,d_569,d_570,d_571,d_572,d_573,d_574,d_575,d_576,d_577,d_578,d_579,d_580,d_581,d_582,d_583,d_584,d_585,d_586,d_587,d_588,d_589,d_590,d_591,d_592,d_593,d_594,d_595,d_596,d_597,d_598,d_599,d_600,d_601,d_602,d_603,d_604,d_605,d_606,d_607,d_608,d_609,d_610,d_611,d_612,d_613,d_614,d_615,d_616,d_617,d_618,d_619,d_620,d_621,d_622,d_623,d_624,d_625,d_626,d_627,d_628,d_629,d_630,d_631,d_632,d_633,d_634,d_635,d_636,d_637,d_638,d_639,d_640,d_641,d_642,d_643,d_644,d_645,d_646,d_647,d_648,d_649,d_650,d_651,d_652,d_653,d_654,d_655,d_656,d_657,d_658,d_659,d_660,d_661,d_662,d_663,d_664,d_665,d_666,d_667,d_668,d_669,d_670,d_671,d_672,d_673,d_674,d_675,d_676,d_677,d_678,d_679,d_680,d_681,d_682,d_683,d_684,d_685,d_686,d_687,d_688,d_689,d_690,d_691,d_692,d_693,d_694,d_695,d_696,d_697,d_698,d_699,d_700,d_701,d_702,d_703,d_704,d_705,d_706,d_707,d_708,d_709,d_710,d_711,d_712,d_713,d_714,d_715,d_716,d_717,d_718,d_719,d_720,d_721,d_722,d_723,d_724,d_725,d_726,d_727,d_728,d_729,d_730,d_731,d_732,d_733,d_734,d_735,d_736,d_737,d_738,d_739,d_740,d_741,d_742,d_743,d_744,d_745,d_746,d_747,d_748,d_749,d_750,d_751,d_752,d_753,d_754,d_755,d_756,d_757,d_758,d_759,d_760,d_761,d_762,d_763,d_764,d_765,d_766,d_767,d_768,d_769,d_770,d_771,d_772,d_773,d_774,d_775,d_776,d_777,d_778,d_779,d_780,d_781,d_782,d_783,d_784,d_785,d_786,d_787,d_788,d_789,d_790,d_791,d_792,d_793,d_794,d_795,d_796,d_797,d_798,d_799,d_800,d_801,d_802,d_803,d_804,d_805,d_806,d_807,d_808,d_809,d_810,d_811,d_812,d_813,d_814,d_815,d_816,d_817,d_818,d_819,d_820,d_821,d_822,d_823,d_824,d_825,d_826,d_827,d_828,d_829,d_830,d_831,d_832,d_833,d_834,d_835,d_836,d_837,d_838,d_839,d_840,d_841,d_842,d_843,d_844,d_845,d_846,d_847,d_848,d_849,d_850,d_851,d_852,d_853,d_854,d_855,d_856,d_857,d_858,d_859,d_860,d_861,d_862,d_863,d_864,d_865,d_866,d_867,d_868,d_869,d_870,d_871,d_872,d_873,d_874,d_875,d_876,d_877,d_878,d_879,d_880,d_881,d_882,d_883,d_884,d_885,d_886,d_887,d_888,d_889,d_890,d_891,d_892,d_893,d_894,d_895,d_896,d_897,d_898,d_899,d_900,d_901,d_902,d_903,d_904,d_905,d_906,d_907,d_908,d_909,d_910,d_911,d_912,d_913,d_914,d_915,d_916,d_917,d_918,d_919,d_920,d_921,d_922,d_923,d_924,d_925,d_926,d_927,d_928,d_929,d_930,d_931,d_932,d_933,d_934,d_935,d_936,d_937,d_938,d_939,d_940,d_941,d_942,d_943,d_944,d_945,d_946,d_947,d_948,d_949,d_950,d_951,d_952,d_953,d_954,d_955,d_956,d_957,d_958,d_959,d_960,d_961,d_962,d_963,d_964,d_965,d_966,d_967,d_968,d_969,d_970,d_971,d_972,d_973,d_974,d_975,d_976,d_977,d_978,d_979,d_980,d_981,d_982,d_983,d_984,d_985,d_986,d_987,d_988,d_989,d_990,d_991,d_992,d_993,d_994,d_995,d_996,d_997,d_998,d_999,d_1000,d_1001,d_1002,d_1003,d_1004,d_1005,d_1006,d_1007,d_1008,d_1009,d_1010,d_1011,d_1012,d_1013,d_1014,d_1015,d_1016,d_1017,d_1018,d_1019,d_1020,d_1021,d_1022,d_1023,d_1024,d_1025,d_1026,d_1027,d_1028,d_1029,d_1030,d_1031,d_1032,d_1033,d_1034,d_1035,d_1036,d_1037,d_1038,d_1039,d_1040,d_1041,d_1042,d_1043,d_1044,d_1045,d_1046,d_1047,d_1048,d_1049,d_1050,d_1051,d_1052,d_1053,d_1054,d_1055,d_1056,d_1057,d_1058,d_1059,d_1060,d_1061,d_1062,d_1063,d_1064,d_1065,d_1066,d_1067,d_1068,d_1069,d_1070,d_1071,d_1072,d_1073,d_1074,d_1075,d_1076,d_1077,d_1078,d_1079,d_1080,d_1081,d_1082,d_1083,d_1084,d_1085,d_1086,d_1087,d_1088,d_1089,d_1090,d_1091,d_1092,d_1093,d_1094,d_1095,d_1096,d_1097,d_1098,d_1099,d_1100,d_1101,d_1102,d_1103,d_1104,d_1105,d_1106,d_1107,d_1108,d_1109,d_1110,d_1111,d_1112,d_1113,d_1114,d_1115,d_1116,d_1117,d_1118,d_1119,d_1120,d_1121,d_1122,d_1123,d_1124,d_1125,d_1126,d_1127,d_1128,d_1129,d_1130,d_1131,d_1132,d_1133,d_1134,d_1135,d_1136,d_1137,d_1138,d_1139,d_1140,d_1141,d_1142,d_1143,d_1144,d_1145,d_1146,d_1147,d_1148,d_1149,d_1150,d_1151,d_1152,d_1153,d_1154,d_1155,d_1156,d_1157,d_1158,d_1159,d_1160,d_1161,d_1162,d_1163,d_1164,d_1165,d_1166,d_1167,d_1168,d_1169,d_1170,d_1171,d_1172,d_1173,d_1174,d_1175,d_1176,d_1177,d_1178,d_1179,d_1180,d_1181,d_1182,d_1183,d_1184,d_1185,d_1186,d_1187,d_1188,d_1189,d_1190,d_1191,d_1192,d_1193,d_1194,d_1195,d_1196,d_1197,d_1198,d_1199,d_1200,d_1201,d_1202,d_1203,d_1204,d_1205,d_1206,d_1207,d_1208,d_1209,d_1210,d_1211,d_1212,d_1213,d_1214,d_1215,d_1216,d_1217,d_1218,d_1219,d_1220,d_1221,d_1222,d_1223,d_1224,d_1225,d_1226,d_1227,d_1228,d_1229,d_1230,d_1231,d_1232,d_1233,d_1234,d_1235,d_1236,d_1237,d_1238,d_1239,d_1240,d_1241,d_1242,d_1243,d_1244,d_1245,d_1246,d_1247,d_1248,d_1249,d_1250,d_1251,d_1252,d_1253,d_1254,d_1255,d_1256,d_1257,d_1258,d_1259,d_1260,d_1261,d_1262,d_1263,d_1264,d_1265,d_1266,d_1267,d_1268,d_1269,d_1270,d_1271,d_1272,d_1273,d_1274,d_1275,d_1276,d_1277,d_1278,d_1279,d_1280,d_1281,d_1282,d_1283,d_1284,d_1285,d_1286,d_1287,d_1288,d_1289,d_1290,d_1291,d_1292,d_1293,d_1294,d_1295,d_1296,d_1297,d_1298,d_1299,d_1300,d_1301,d_1302,d_1303,d_1304,d_1305,d_1306,d_1307,d_1308,d_1309,d_1310,d_1311,d_1312,d_1313,d_1314,d_1315,d_1316,d_1317,d_1318,d_1319,d_1320,d_1321,d_1322,d_1323,d_1324,d_1325,d_1326,d_1327,d_1328,d_1329,d_1330,d_1331,d_1332,d_1333,d_1334,d_1335,d_1336,d_1337,d_1338,d_1339,d_1340,d_1341,d_1342,d_1343,d_1344,d_1345,d_1346,d_1347,d_1348,d_1349,d_1350,d_1351,d_1352,d_1353,d_1354,d_1355,d_1356,d_1357,d_1358,d_1359,d_1360,d_1361,d_1362,d_1363,d_1364,d_1365,d_1366,d_1367,d_1368,d_1369,d_1370,d_1371,d_1372,d_1373,d_1374,d_1375,d_1376,d_1377,d_1378,d_1379,d_1380,d_1381,d_1382,d_1383,d_1384,d_1385,d_1386,d_1387,d_1388,d_1389,d_1390,d_1391,d_1392,d_1393,d_1394,d_1395,d_1396,d_1397,d_1398,d_1399,d_1400,d_1401,d_1402,d_1403,d_1404,d_1405,d_1406,d_1407,d_1408,d_1409,d_1410,d_1411,d_1412,d_1413,d_1414,d_1415,d_1416,d_1417,d_1418,d_1419,d_1420,d_1421,d_1422,d_1423,d_1424,d_1425,d_1426,d_1427,d_1428,d_1429,d_1430,d_1431,d_1432,d_1433,d_1434,d_1435,d_1436,d_1437,d_1438,d_1439,d_1440,d_1441,d_1442,d_1443,d_1444,d_1445,d_1446,d_1447,d_1448,d_1449,d_1450,d_1451,d_1452,d_1453,d_1454,d_1455,d_1456,d_1457,d_1458,d_1459,d_1460,d_1461,d_1462,d_1463,d_1464,d_1465,d_1466,d_1467,d_1468,d_1469,d_1470,d_1471,d_1472,d_1473,d_1474,d_1475,d_1476,d_1477,d_1478,d_1479,d_1480,d_1481,d_1482,d_1483,d_1484,d_1485,d_1486,d_1487,d_1488,d_1489,d_1490,d_1491,d_1492,d_1493,d_1494,d_1495,d_1496,d_1497,d_1498,d_1499,d_1500,d_1501,d_1502,d_1503,d_1504,d_1505,d_1506,d_1507,d_1508,d_1509,d_1510,d_1511,d_1512,d_1513,d_1514,d_1515,d_1516,d_1517,d_1518,d_1519,d_1520,d_1521,d_1522,d_1523,d_1524,d_1525,d_1526,d_1527,d_1528,d_1529,d_1530,d_1531,d_1532,d_1533,d_1534,d_1535,d_1536,d_1537,d_1538,d_1539,d_1540,d_1541,d_1542,d_1543,d_1544,d_1545,d_1546,d_1547,d_1548,d_1549,d_1550,d_1551,d_1552,d_1553,d_1554,d_1555,d_1556,d_1557,d_1558,d_1559,d_1560,d_1561,d_1562,d_1563,d_1564,d_1565,d_1566,d_1567,d_1568,d_1569,d_1570,d_1571,d_1572,d_1573,d_1574,d_1575,d_1576,d_1577,d_1578,d_1579,d_1580,d_1581,d_1582,d_1583,d_1584,d_1585,d_1586,d_1587,d_1588,d_1589,d_1590,d_1591,d_1592,d_1593,d_1594,d_1595,d_1596,d_1597,d_1598,d_1599,d_1600,d_1601,d_1602,d_1603,d_1604,d_1605,d_1606,d_1607,d_1608,d_1609,d_1610,d_1611,d_1612,d_1613,d_1614,d_1615,d_1616,d_1617,d_1618,d_1619,d_1620,d_1621,d_1622,d_1623,d_1624,d_1625,d_1626,d_1627,d_1628,d_1629,d_1630,d_1631,d_1632,d_1633,d_1634,d_1635,d_1636,d_1637,d_1638,d_1639,d_1640,d_1641,d_1642,d_1643,d_1644,d_1645,d_1646,d_1647,d_1648,d_1649,d_1650,d_1651,d_1652,d_1653,d_1654,d_1655,d_1656,d_1657,d_1658,d_1659,d_1660,d_1661,d_1662,d_1663,d_1664,d_1665,d_1666,d_1667,d_1668,d_1669,d_1670,d_1671,d_1672,d_1673,d_1674,d_1675,d_1676,d_1677,d_1678,d_1679,d_1680,d_1681,d_1682,d_1683,d_1684,d_1685,d_1686,d_1687,d_1688,d_1689,d_1690,d_1691,d_1692,d_1693,d_1694,d_1695,d_1696,d_1697,d_1698,d_1699,d_1700,d_1701,d_1702,d_1703,d_1704,d_1705,d_1706,d_1707,d_1708,d_1709,d_1710,d_1711,d_1712,d_1713,d_1714,d_1715,d_1716,d_1717,d_1718,d_1719,d_1720,d_1721,d_1722,d_1723,d_1724,d_1725,d_1726,d_1727,d_1728,d_1729,d_1730,d_1731,d_1732,d_1733,d_1734,d_1735,d_1736,d_1737,d_1738,d_1739,d_1740,d_1741,d_1742,d_1743,d_1744,d_1745,d_1746,d_1747,d_1748,d_1749,d_1750,d_1751,d_1752,d_1753,d_1754,d_1755,d_1756,d_1757,d_1758,d_1759,d_1760,d_1761,d_1762,d_1763,d_1764,d_1765,d_1766,d_1767,d_1768,d_1769,d_1770,d_1771,d_1772,d_1773,d_1774,d_1775,d_1776,d_1777,d_1778,d_1779,d_1780,d_1781,d_1782,d_1783,d_1784,d_1785,d_1786,d_1787,d_1788,d_1789,d_1790,d_1791,d_1792,d_1793,d_1794,d_1795,d_1796,d_1797,d_1798,d_1799,d_1800,d_1801,d_1802,d_1803,d_1804,d_1805,d_1806,d_1807,d_1808,d_1809,d_1810,d_1811,d_1812,d_1813,d_1814,d_1815,d_1816,d_1817,d_1818,d_1819,d_1820,d_1821,d_1822,d_1823,d_1824,d_1825,d_1826,d_1827,d_1828,d_1829,d_1830,d_1831,d_1832,d_1833,d_1834,d_1835,d_1836,d_1837,d_1838,d_1839,d_1840,d_1841,d_1842,d_1843,d_1844,d_1845,d_1846,d_1847,d_1848,d_1849,d_1850,d_1851,d_1852,d_1853,d_1854,d_1855,d_1856,d_1857,d_1858,d_1859,d_1860,d_1861,d_1862,d_1863,d_1864,d_1865,d_1866,d_1867,d_1868,d_1869,d_1870,d_1871,d_1872,d_1873,d_1874,d_1875,d_1876,d_1877,d_1878,d_1879,d_1880,d_1881,d_1882,d_1883,d_1884,d_1885,d_1886,d_1887,d_1888,d_1889,d_1890,d_1891,d_1892,d_1893,d_1894,d_1895,d_1896,d_1897,d_1898,d_1899,d_1900,d_1901,d_1902,d_1903,d_1904,d_1905,d_1906,d_1907,d_1908,d_1909,d_1910,d_1911,d_1912,d_1913
    cols_calendar   = cols_remove( calendar.columns       , ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "event_name_1", "event_name_2", "d" ])    ### Add manuallly  date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,event_name_2,event_type_2,snap_CA,snap_TX,snap_WI
    cols_static     = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]  


    df_timeseries0 = sales_train_val[ cols_timeseries]
    df_dynamic0    = calendar[ cols_calendar]
    df_static0     = sales_train_val[ cols_static]
    #df_timeseries.to_csv('original df_timeseries')
    #df_dynamic.to_csv('original df_dynamic')
    #df_static.to_csv('original df_static')
    print('shape of original df_dynamic'+str(df_dynamic0.shape))

    print(df_timeseries0.head(10))
    print(df_dynamic0.head(10))
    print(df_static0.head(10))
    print(df_timeseries0.shape,df_dynamic0.shape,df_static0.shape )

    #### For Model definition

    return df_timeseries0, df_dynamic0, df_static0, pars_data 
  

def test_gluonts_to_pandas() :
    df_timeseries0, df_dynamic0, df_static0, pars_data  = load_datasset_m5()
    cardinalities = gluonts_static_cardinalities(df_static0, submission=1, single_pred_length=28, submission_pred_length=10, n_timeseries=1, transpose=1) 

    ##### Generate the Gluonts format dataset
    gluonts_datafolder='time/data/gluonts_m5_02'
    pandas_to_gluonts(df_timeseries0, df_dynamic0, df_static0,
                                 pars=pars_data,
                                  path_save=gluonts_datafolder , return_df=False) 

    #################################################################################################################    
    from pathlib import Path
    gluonts_datafolder='time/data/gluonts_m5_02'
    dataset_path=Path(gluonts_datafolder)
    df_timeseries1,df_dynamic1,df_static1 = gluonts_to_pandas(dataset_path)
    print(df_timeseries1.shape,df_dynamic1.shape,df_static1.shape  )

    print(pd_difference(df_timeseries0,df_timeseries1))
    print(pd_difference(df_dynamic0,df_dynamic1))
    print(pd_difference(df_static0,df_static1))


    ### test above function #####
    firstProductSet = {'Product1': ['Computer','Phone','Printer','Desk'],
                       'Price1': [1200,800,200,350]
                       }
    df1 = pd.DataFrame(firstProductSet,columns= ['Product1', 'Price1'])
    print(df1)

    secondProductSet = {'Product1': ['Computer','Phone','Printer','Desk'],
                        'Price1': [900,800,300,350]
                        }
    df2 = pd.DataFrame(secondProductSet,columns= ['Product1', 'Price1'])
    print(df2)
    pd_difference(df1,df2)



def test_pandas_to_gluonts():
  #################################################################################################################
  ##### Generate Data using functions  ############################################################################
  from zlocal import root
  dir0         = root + "/zsample/m5_dataset/" 
  gluonts_path = dir0 + "/json/"
  

  df_timeseries, df_dynamic, df_static, pars_data  = load_datasset_m5()

  ####### Generate the Gluonts format dataset
  pandas_to_gluonts(df_timeseries, df_dynamic, df_static,
                                pars=pars_data,
                                path_save=gluonts_path , return_df=False) 

  ####### For Model definition
  cardinalities = gluonts_static_cardinalities(df_static, submission=1, single_pred_length=28, submission_pred_length=10, n_timeseries=1, transpose=1) 


  test_ds  = None
  train_ds = None
  # test gluonts data
  from gluonts.dataset.common import ListDataset,load_datasets
  dataset_path  = gluonts_path
      
  TD  = load_datasets(  metadata=dataset_path,
                         train=dataset_path + "/train",
                         test=dataset_path  + "/test",)


if __name__ == '__main__':
  print("loaded")



"""


a,b = gluonts_create_dynamic(df_dynamic, submission=True, single_pred_length=28, submission_pred_length=10, 
                           n_timeseries=1, transpose=1) 

col = "event_type_1"
df_dynamic[col].apply(lambda x: to_flag(x) )


 to_flag('ok')


cal_features = calendar.drop(
    ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'event_name_1', 'event_name_2', 'd'], 
    axis=1
)
cal_features['event_type_1'] = cal_features['event_type_1'].apply(lambda x: 0 if str(x)=="nan" else 1)
cal_features['event_type_2'] = cal_features['event_type_2'].apply(lambda x: 0 if str(x)=="nan" else 1)

test_cal_features = cal_features.values.T
if submission:
    train_cal_features = test_cal_features[:,:-submission_prediction_length]
else:
    train_cal_features = test_cal_features[:,:-submission_prediction_length-single_prediction_length]
    test_cal_features = test_cal_features[:,:-submission_prediction_length]

test_cal_features_list = [test_cal_features] * len(sales_train_validation)
train_cal_features_list = [train_cal_features] * len(sales_train_validation)

"""













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
