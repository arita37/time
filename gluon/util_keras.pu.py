from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
import mdn

from sklearn.preprocessing import MinMaxScaler



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




##### Conversion ################################################################################
def zgenerate_pivot_unit(df, keyref=None,  **kw) :
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


def zgenerate_pivot_gluonts(path_input="/data/pos/", path_export=None, folder_list=None, cols=None,
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

