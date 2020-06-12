import os
import joblib
from abc import abstractmethod
from kernel import utils


def catboost_preprocessing(df_fea, **kwargs):
    if kwargs.get('cat_cols', False):
        cat_cols = kwargs['cat_cols']
        num_cols = list(set(df_fea.columns) - set(cat_cols))
    else:
        num_cols, cat_cols = utils.get_feature_types(df_fea)
    for c in cat_cols:
        df_fea[c] = df_fea[c].astype(str)
    return df_fea, cat_cols, num_cols


class GenericDataProcessor:
    def __init__(self, df, label_col, model_name=None, normalize=False, max_onehot=10, **kwargs):
        self.id_col = df.index.name
        if kwargs.get('cat_cols', False):
            self.cat_cols = kwargs['cat_cols']
            self.num_cols = list(set(df_fea.columns) - set(self.cat_cols + [label_col]))
        else:
            self.num_cols, self.cat_cols = utils.get_feature_types(df.drop(label_col, axis=1))
        self.save_path = utils.init_model_dir(model_name)
        if self.save_path is not None:
            self.data = self.__process_features(df_fea, normalize, max_onehot)

    def __process_features(self, df_fea, normalize, max_onehot):
        data, self.cat_info = utils.cat_processing(df_fea, self.cat_cols, max_onehot)
        data, self.num_info = utils.num_processing(data, self.num_cols, normalize)
        self.feature_cols = self.num_info['columns'] + self.cat_info['columns']
        return data[self.feature_cols].copy()
    
    def __convert_num_cols(self, df_test):
        for col in self.num_info['columns']:
            df_test[col] = df_test[col].fillna(self.num_info['medians'][col])
        scaler = self.num_info['scaler']
        if scaler:
            res = scaler.transform(df_test[self.num_info['columns']].values).T
        for i, n in enumerate(self.num_info['columns']):
            df_test[n] = res[i]
        return df_test
    
    def __conver_cat_cols(self, df_test):
        for col in self.cat_info['label']:
            df_test[col] = df_test[col].fillna('UNK').astype(str)
            enc = self.cat_info['encoders'][col]
            df_test.loc[~df_test[col].isin(enc.classes_), col] = enc.classes_[0]
            df_test[col] = enc.transform(df_test[col])
        for col in self.cat_info['one_hot']:
            df_test[col] = df_test[col].fillna('UNK').astype(str)
            enc = self.cat_info['encoders'][col]
            res = enc.transform(df_test[col].values.reshape(-1,1))
            feature_names = enc.get_feature_names([col])
            for n, v in zip(feature_names, res.toarray().T):
                df_test[n] = v
        return df_test

    def transform(self, df_test_in):
        df_test = df_test_in.copy()
        if self.id_col:
            df_test.set_index(self.id_col, inplace=True)
        df_test = self.__convert_num_cols(df_test).pipe(self.__conver_cat_cols)
        return df_test[self.feature_cols]
    
    def save(self):
        if hasattr(self, 'data'):
            del self.data
        joblib.dump(self, os.path.join(self.save_path, 'data_processor.pkl'))