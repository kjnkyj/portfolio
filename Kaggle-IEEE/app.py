
import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
import gc

from .memory_red import cat_dict
import pickle
import os

params = {'colsample_by_tree' : 0.2,
                           'early_stopping_rounds' : 100,
                           'learning_rate' : 0.05,
                           'min_child_samples' : 20,
                           'tree_method' : 'gpu_hist',
         'min_child_weight' : 5,
          'num_tree_leaves' : 40000,
          'objective' : 'binary',
          'subsample_for_bin' : 50000,
          'max_depth' : 7,
          'n_estimators' : 4000,
          'subsample' : 1,
          'reg_alpha' : 0,
          'reg_lambda':0
         }



def impute(df,object_columns):
    df[object_columns] = df[object_columns].fillna(df[object_columns].mode())
    df = df.fillna(df.mean())



def kfold_xgb(data,y,params,test, limit = -1,saved = False, model_name = None):
    folds = KFold(n_splits=5, shuffle=True, random_state=123)
    oof_preds = np.zeros(data.shape[0])
    sub_preds = np.zeros(test.shape[0])
    n = -1
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
        if limit != -1:
            n+=1
            if n == limit:
                break

        trn_x, trn_y = data.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = data.iloc[val_idx], y.iloc[val_idx]
        if saved:
            clf = pickle.load(open(model_name,"rb"))
        else:
            clf = XGBClassifier(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                num_leaves=params['num_tree_leaves'],
                colsample_bytree=params['colsample_by_tree'],
                subsample=params['subsample'],
                max_depth=params['max_depth'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                #min_split_gain=.01,
                min_child_weight=params['min_child_weight'],
                #tree_method = params['tree_method']
            )

            clf.fit(trn_x, trn_y,
                    eval_set= [(trn_x, trn_y), (val_x, val_y)],
                    eval_metric='auc', verbose=50, early_stopping_rounds=params['early_stopping_rounds']
                   )

            pickle.dump(clf, open("cv_{}_xgb.pickle.dat".format(n_fold), "wb"))
        prob = clf.predict_proba(val_x)
        oof_preds[val_idx] = list(zip(*pred))[1]
        sub_preds += clf.predict_proba(test) / folds.n_splits
        sub_preds_nocv = clf.predict_proba(test)
        sub_preds_nocv_bi = clf.predict(test)
        prb_predictions = pd.DataFrame([sub_preds_nocv,sub_preds_nocv_bi], columns = ['isFraud_prb','isFraud'])
        prb_predictions.to_csv('cv_{}_submission.csv'.format(n_fold),index = False)




        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))


def main(transformed = True):


    cwd = os.getcwd()
    train = pd.read_csv(cwd + '/Kaggle-IEEE/train.csv')
    test = pd.read_csv(cwd + '/Kaggle-IEEE/test.csv')
    print ("Data Loaded")
    object_columns = train.select_dtypes(include=['object']).columns
    us_emails, emails, labels = cat_dict()

    train['nulls1'] = train.isna().sum(axis=1)
    test['nulls1'] = test.isna().sum(axis=1)

    for c in ['P_emaildomain', 'R_emaildomain']:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)

        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    for c1, c2 in train.dtypes.reset_index().values:
        if c2 == 'O':
            train[c1] = train[c1].map(lambda x: labels[str(x).lower()])
            test[c1] = test[c1].map(lambda x: labels[str(x).lower()])

    impute(train, object_columns)
    impute(test, object_columns)
    print ("data transformed")
    train.to_csv('train_trans.csv',index = False)
    test.to_csv('test_trans.csv',index = False)
    #kfold_xgb(train.loc[:, train.columns != 'isFraud'], train.isFraud, params, test, limit =1, saved = True, model_name= 'cv_0_xgb.pickle.dat')







if __name__ == "__main__":

    main()