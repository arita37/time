import os, sys
import pandas as pd, numpy as np


#############################################################################################
##################### CREATE SYNTHETIC TIMESERIES############################################
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

    
    
    

def model_eval(clf0, df, colX, coly="y", test_size=0.5, istrain=1, use_eval=0 ) :
  clf = copy.deepcopy(clf0)  
  yy  = df[coly].values
  X   = df[colX].values  
  X   = X.reshape(-1,1) if len(colX) == 1  else X
  print("X", X.shape )

  if istrain : 
     X_train, X_test, y_train, y_test = train_test_split( X, yy, test_size=test_size, 
                                                          random_state=42)
     del X, yy
     gc.collect()
     if use_eval  :       
       try :
         clf.fit(X_train, y_train , eval_set= (X_test0, y_test0) )
       except :
         print("Using local", flush=True)  
         clf.fit(X_train, y_train , eval_set= (X_test, y_test) )  
           
     else :
       clf.fit(X_train, y_train )


     ytest_proba   = clf.predict_proba(X_test)[:, 1]
     ytest_pred    = clf.predict(X_test)
     sk_showmetrics(y_test, ytest_pred, ytest_proba)
     return clf

  else :
     y_proba   = clf.predict_proba(X)[:, 1]
     y_pred    = clf.predict(X)
     sk_showmetrics(yy, y_pred, y_proba)





def model_fit(df2, cols_train,   col_target = "y", save_suffix="area_gms_201909",
                 modelname= "RandomForestClassifier", dosave=1 , coldate="dateint", test_size=0.9,
                 coldate_limit = 201801, dfeval=None,
                 dirmodel ="",
                 **kw) :
    #cols_train = [  'ALL_ROOM', 'travel_gms_total_1yr', 'travel_gms_total_6mth',
    #               'travel_gms_total_1mth', 'travel_cnt_total_1yr',
    #               'travel_cnt_total_6mth',  'gms_6mth_diff', 'gms_1yr_diff',   ]
    # col_target = "y"
    print( df2[ cols_train ].head(3) )
    print( "coltarget",  sum(df2[ col_target] ) )
    imax =len(df2)


    ########### Train  ############################################################
    clf, use_eval = model_get(name= modelname, **kw)
    #cols_train = RandomForestClassifier(max_depth= kw["max_depth"], n_estimators= kw["n_estimators"],
    #                                  class_weight="balanced"  )  # random_state=0,

    clf = model_eval( clf, df2,
                      colX = cols_train, coly= col_target, test_size=test_size, istrain = 1,
                      use_eval= use_eval )

    clf_features = feature_impt_rf(clf , cols_train)
    

    ###########  Prediction Check   ##############################################
    if dfeval is not None :
       print("Using Eval") 
    else :     
       dfeval =  df2
       print("using Full")
    
    dfeval[ col_target + "pred" ]      = clf.predict( dfeval[cols_train].values )
    dfeval[ col_target + "pred_proba"] = clf.predict_proba( dfeval[cols_train].values )[:,1]

    dfstat = metric_accuracy_check(dfeval, col_target= col_target, ypred = col_target + "pred",
                                   ypred_proba = col_target + "pred_proba", coldate = coldate )
    print(dfstat)
    
    if dosave :
      ###########  Export Model  ##################################################
      dirmodel2 = dirmodel + "/" +save_suffix +"/"
      os.makedirs( dirmodel2, exist_ok=True )
      save_model(clf, cols_train, df2, dirmodel2 , f"clf" )
      # df2.to_csv(  dirmodel + f"/travel_{save_suffix}_.csv")
      dfstat.to_csv(  dirmodel2 + f"/clf_stats.csv")
      save_session(folder= dirsession + f"/{save_suffix}_train" , glob=globals() )

    return clf, dfstat, clf_features




"""
clf = clf_h
cols_train = cols_h_train





####################################################################################################
############  Train  h #########################################################################
df    = df[ -df.n_user.isnull( )]
df    = df[ -df.travel_gms_total_3mth.isnull( )]
df    = df[ -df.travel_gms_total_6mth.isnull( )]
df    = df[ -df.travel_gms_total_1mth_area_log_area.isnull() ]





cols_h = list( df.columns)
cols_h_train = col_remove(cols_h, [ "date", "y", "zip3",  "shi_int", 'travel_gms_total_3yr_h_log',
                                           'travel_cnt_total_3yr_h_log',
                                           
                     'y2', 'y3', 'dateint', 'ypred_area', 'zipcode', 'C_TIKU_ID', 'y2',
                    'travel_cnt_total_3yr',   'travel_gms_total_1yr_total', 
                    'C_TIKU_ID', 'h_name',
                    'zipcode',
                    'cat1', 'area_name', 'ken', 'shi', 'size', 'cat2',
                    'h_gms_score', 'amt_sum', 'amt_max', 'year', 'h_id',
                    
                    'gms_3yr_diff', 'cnt_3yr_diff', 'gms_3yr_diff_area',
                    'travel_gms_total_3yr',
                    
                    
                    
                    'travel_cnt_total_3yr_area_log_area',  'travel_gms_total_3yr_area_log_area',
                    'travel_cnt_total_3yr_log',  'travel_gms_total_3yr_log',
                    'gms_6mth_yoy',  'ken_int_area',
 'shi_int_area',  'ypred',
 'gms_1yr_yoy',
 'gms_3mth_yoy'
                                           
            ]   )
print(len( cols_h_train))
    
col_target = 'y'

datec      = 201801
datemin    = 201706
datemax    = 201905



#### Check
for x in cols_h_train :
    print(x, df[ (df.date >= datemin ) & (df.date < datemax ) ][x].isnull().sum() )
df[ (df.date >= datemin ) & (df.date < datemax )   ]['y'].hist()






###### Details
ii = 0
dirmodel   = "C:/Users/kevin.noel/Box/Data Science Department/Personal/znono/a/model/h/"
dirsession =  dirmodel + "/session/"


#### Eval Dataset 
X_test0 =  df[ (df.date >= datec ) & (df.date < datemax )   ][ cols_h_train].values
y_test0 =  df[ (df.date >= datec ) & (df.date < datemax ) ][ col_target].values
print(X_test0.shape)



dfeval = None


ii = ii + 1
clf_h, dfstat_h,clf_features_h = model_fit( df[ (df.date >= datemin) & (df.date <= datemax ) ], 
              cols_train  = cols_h_train,   col_target = "y",
              coldate     = 'date',
              # dfeval      = df[ (df.date > 201806)  & (df.date < 201905 ) ],
              
      save_suffix  = "area_gms_202004_OK_v"+ str(ii) , 
      dirmodel = dirmodel ,
                    
      dosave = 1 ,
      test_size = 0.40,

      modelname = "LGBMClassifier",
      
      num_leaves= 300, 
      max_depth=40, 
      
      learning_rate=0.01, 
      num_iterations= 200,
      max_bin = 800,       

      n_estimators= 300,
      boosting_type='gbdt',  
      
       bagging_fraction=0.3,
       
        subsample_for_bin=200000, objective="binary",
        class_weight="balanced", min_split_gain=0.0, min_child_weight=0.001,
        min_child_samples=5, subsample=1.0, subsample_freq=0,
        colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
        n_jobs=-1, silent=False, importance_type='split' )




##### Evaluation Over a period :
df["ypred"]       =  clf_h.predict( df[ cols_h_train ].values )
df["ypred_proba"] =  clf_h.predict_proba( df[ cols_h_train ] )[:,1]


dfeval2 =  df[ (df.date >= datec ) & (df.date < datemax  )  ]
dfstat  = metric_accuracy_check(dfeval2, col_target= col_target, ypred = col_target + "pred",
                                   ypred_proba = col_target + "pred_proba", coldate = 'date' )


dfstat.to_csv(  dirmodel + f"/clf_stats.csv" )




#####################################
gluonts_model_eval( clf_h, df[ (df.date >= datec )  & (df.date < datemax ) ],
            colX = cols_h_train, coly= col_target, test_size=0.99, istrain = 0,
            use_eval= 1 )


print(len(df[ (df.date >= datec )  ]))
df.groupby('date').agg({ 'h_id' : 'count' })
df.columns


"""

