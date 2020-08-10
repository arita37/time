from util_feat import *
FIRST_DAY = 1500
df_demand = create_dt(is_train=True, first_day= FIRST_DAY)
df_rolling = create_dt(is_train=True, first_day= FIRST_DAY)
rolling_sales(df_rolling)
rolling_demand(df_demand)

df_demand.dropna(inplace=True)
df_rolling.dropna(inplace=True)
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols_d = df_demand.columns[~df_demand.columns.isin(useless_cols)]
X_train_d = df_demand[train_cols_d]
y_train_d = df_demand["sales"]
train_cols_r = df_rolling.columns[~df_rolling.columns.isin(useless_cols)]
X_train_r = df_rolling[train_cols_r]
y_train_r = df_rolling["sales"]
train_data_r = lgb.Dataset(X_train_r, label = y_train_r, categorical_feature=cat_feats, free_raw_data=False)
train_data_d = lgb.Dataset(X_train_d, label = y_train_r, categorical_feature=cat_feats, free_raw_data=False)
fake_valid_inds_r = np.random.choice(30,4000)
fake_valid_inds_d = np.random.choice(10,400)
fake_valid_data_r = lgb.Dataset(X_train_r.iloc[fake_valid_inds_r], label = y_train_r.iloc[fake_valid_inds_r],categorical_feature=cat_feats,
                             free_raw_data=False)   
fake_valid_data_d = lgb.Dataset(X_train_d.iloc[fake_valid_inds_d], label = y_train_d.iloc[fake_valid_inds_d],categorical_feature=cat_feats,
                             free_raw_data=False)   
params = {
        "objective" : "poisson",
        "metric" :"rmse",
     #   'min_data_in_leaf': 2**8-1, 
        #'num_leaves': 2**7-1, #model only for fast check
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
    #     'device' : 'gpu',
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    #'early_stopping_rounds': 30,

#     'num_iterations' : 200,
    'num_iterations' : 250,
}
valid_sets_r = [fake_valid_data_r]
valid_sets_d = [fake_valid_data_d]
m_lgb_r = lgb.train(params, train_data_r, valid_sets = [fake_valid_data_r], verbose_eval=50) 
m_lgb_d = lgb.train(params, train_data_d, valid_sets = [fake_valid_data_d], verbose_eval=50) 
alphas = [1.035, 1.03, 1.025, 1.02]
weights = [1/len(alphas)]*len(alphas)
sub = 0.
for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_dt(False)
    cols = [f"F{i}" for i in range(1,5)]

    for tdelta in range(0, 5):
#     for tdelta in range(0, 1):
        day = fday + timedelta(days=tdelta)
        print(icount, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        rolling_demand(tst)
        tst = tst.loc[tst.date == day , train_cols_d]
        te.loc[te.date == day, "sales"] = alpha*m_lgb_d.predict(tst) # magic multiplier by kyakovlev



    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 
#                                                                           "id"].str.replace("validation$", "evaluation")
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0., inplace = True)
    te_sub.sort_values("id", inplace = True)
    te_sub.reset_index(drop=True, inplace = True)
#     te_sub.to_csv("submission.csv",index=False)
    if icount == 0 :
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    print(icount, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_d.csv",index=False)


for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_dt(False)
    cols = [f"F{i}" for i in range(1,29)]

    for tdelta in range(0, 5):
#     for tdelta in range(0, 1):
        day = fday + timedelta(days=tdelta)
        print(icount, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        rolling_sales(tst)
        tst = tst.loc[tst.date == day , train_cols_r]
        te.loc[te.date == day, "sales"] = alpha*m_lgb_r.predict(tst) # magic multiplier by kyakovlev



    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 
#                                                                           "id"].str.replace("validation$", "evaluation")
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0., inplace = True)
    te_sub.sort_values("id", inplace = True)
    te_sub.reset_index(drop=True, inplace = True)
#     te_sub.to_csv("submission.csv",index=False)
    if icount == 0 :
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    print(icount, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_r.csv",index=False)

