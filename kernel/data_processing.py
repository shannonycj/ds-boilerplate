import os
import joblib
from kernel import utils


class DataProcessor:
    def __init__(self, X, model_name=None, normalize=False, max_onehot=10):
        self.id_col = X.index.name
        self.num_cols, self.cat_cols = utils.get_feature_types(X)
        self.save_path = utils.init_model_dir(model_name)
        if self.save_path is not None:
            self.data = self.__process_features(X, normalize, max_onehot)

    def __process_features(self, X, normalize, max_onehot):
        data, self.cat_info = utils.cat_processing(X, self.cat_cols, self.save_path, max_onehot)
        data, self.num_info = utils.num_processing(data, self.num_cols, self.save_path, normalize)
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
            df_test[col] = df_test[col].fillna('UNK')
            enc = joblib.load(os.path.join(self.save_path, f'encoders/{col}_enc.pkl'))
            df_test.loc[~df_test[col].isin(enc.classes_), col] = 'UNK'
            df_test[col] = enc.transform(df_test[col])
        for col in self.cat_info['one_hot']:
            df_test[col] = df_test[col].fillna('UNK')
            enc = joblib.load(os.path.join(self.save_path, f'encoders/{col}_enc.pkl'))
            res = enc.transform(df_test[col].values.reshape(-1,1))
            feature_names = enc.get_feature_names([col])
            for n, v in zip(feature_names, res.toarray().T):
                df_test[n] = v
        return df_test

    def transform(self, df_test):
        if self.id_col:
            df_test.set_index(self.id_col, inplace=True)
        df_test = self.__convert_num_cols(df_test).pipe(self.__conver_cat_cols)
        return df_test[self.feature_cols]
    
    def save(self):
        joblib.dump(self, os.path.join(self.save_path, 'data_processor.pkl'))