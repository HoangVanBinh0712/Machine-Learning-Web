#%%
# Lớp FeatureUnion dùng để hợp nhất các feature khi xử lý dữ liệu từ nhiều pipline
from sklearn.pipeline import FeatureUnion
#Hàm mean dùng để tính giá trị trung bình
from statistics import mean
# Thư viện dùng để đọc dữ liệu
import pandas as pd
# Thư viện tính toán số học, ma trận,...
import numpy as np
# Hàm Pipeline dùng để xử lý dữ liệu
from sklearn.pipeline import Pipeline
# Hàm BaseEstimator, TransformerMixin hỗ trợ xử lý dữ liệu
from sklearn.base import BaseEstimator, TransformerMixin
# Hàm SimpleImputer dùng để điền các giá trị bị thiếu của feature
from sklearn.impute import SimpleImputer  
# Hàm train_test_split dùng để chia dữ liệu thành tập train và tập test
from sklearn.model_selection import train_test_split
# Hàm StandardScaler dùng để scaling dữ liệu
from sklearn.preprocessing import StandardScaler
# Hàm OneHotEncoder dùng để đưa dữ liệu (1- nhiều cột) thành dạng mã hóa one-hot
from sklearn.preprocessing import OneHotEncoder
# Lớp GridSearchCV dùng để tìm hyperparameter
from sklearn.model_selection import GridSearchCV 
# Hàm PolynomialFeatures dùng để biến các feature thành feature bậc cao (mặc định là bậc 2)
from sklearn.preprocessing import PolynomialFeatures
# Nạp model LinearRegression
from sklearn.linear_model import LinearRegression
# Thư viện dùng để đánh giá model
from sklearn.model_selection import KFold
# Thư viện dùng để phân dữ liệu thành nhiều bộ giúp đánh giá được model tốt hơn
import joblib 

#%% 1. Xử lý dữ liệu

# 1.1. Đọc csv 
df_main = pd.read_csv('./Car-details-v3.csv')
df_main.head()

df_main.shape

df_main.info()

df_main.describe()

df_main.drop('torque',axis=1,inplace = True)

df_main['age'] = 2022 - df_main['year']
df_main.drop('year',axis=1,inplace = True)

#%% Chuan Hoa
#Engine to engine_cc
engine_CC = df_main['engine'].astype(str).apply(lambda x : x.split(' ')[0])
df_main.insert(10,"engine_CC",engine_CC)
df_main['engine_CC'] = df_main['engine_CC'].astype(float)
df_main.drop('engine',axis=1,inplace = True)

#MAX POWER to max_power_bhb
max_power_bhb = df_main['max_power'].astype(str).apply(lambda x : x.split(' ')[0])
df_main.insert(11,"max_power_bhb",max_power_bhb)
df_main['max_power_bhb'] = df_main['max_power_bhb'].astype(float)
df_main.drop('max_power',axis=1,inplace = True)

# mileage to mileage_kmpl
mileage_kmpl = df_main['mileage'].astype(str).apply(lambda x : x.split(' ')[0])
df_main.insert(9,"mileage_kmpl",mileage_kmpl)
df_main['mileage_kmpl'] = df_main['mileage_kmpl'].astype(float)
df_main.drop('mileage',axis=1,inplace = True)


## Chuan Hoa Xong
# %% Thêm cột Brand
brand = df_main['name'].astype(str).str.split()
for i in range(0, df_main.shape[0]):
    brand[i] = brand[i][0]
df_main.insert(1,"brand",brand)
df_main.drop('name',axis=1,inplace = True)

df_main.info()


# X : Các feature khác selling_price, Y : selling_price
X = df_main.drop(columns="selling_price")
y = df_main["selling_price"].copy()


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values        

num_feat_names = ['km_driven', 'mileage_kmpl', 'age', 'engine_CC', 'max_power_bhb', 'seats'] # =list(train_set.select_dtypes(include=[np.number]))
cat_feat_names = ['fuel', 'seller_type', 'transmission', 'owner', 'brand'] # =list(train_set.select_dtypes(exclude=[np.number])) 

# 4.4.2 Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
    ])    
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="mean", copy=True)), # copy=False: imputation will be done in-place
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
    ])  

#pipeline
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  
#joblib.dump(full_pipeline, r'models/full_pipeline.pkl')
#joblib.dump(ColumnSelector,r'models/ColumnSelector.pkl')
processed_train_set_val = full_pipeline.fit_transform(X)
joblib.dump(X, r'models/data_to_fit_transform.pkl')

# tách tập train và test
X_train, X_test, y_train, y_test = train_test_split(processed_train_set_val, y, test_size=0.2, random_state=42)


#%% 3. Xử lý các hàm cần thiết

#store model and load model
def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_model.pkl')
def load_model(model_name):
    model = joblib.load('models/' + model_name + '_model.pkl')
    return model

# r2score, root mean square error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse    



#%% 4. Training các models
# 4.1. RANDOM FOREST MODEL
run_new = False
# Ý nghĩa các hyperparameter được chọn
# n_estimators: Chỉ định số tree trong forest
# max_depth: Chỉ định độ sâu tối đa của decision tree.
# min_samples_split: Chỉ định số sample tối thiểu cho phép để tạo một node mới (decision boundary mới).
# min_samples_leaf: Chỉ định số sample tối thiểu trên một node lá.
# max_leaf_nodes: Chỉ định số node lá tối đa của decision tree.
# max_features: Chỉ định số lượng feature để xét tách node mới tốt nhất (auto: max_features=n_features; sqrt: max_features=sqrt(n_features); log2: max_features=log2(n_features) với n_features là số lượng feature).
from sklearn.ensemble import RandomForestRegressor
#n_jobs: Sử dụng tất cả bộ xử lý để chạy thuật toán
# param_grid: một list danh sách chứa các tham số với các giá trị khác nhau
# cv: thực hiện cross validation với số lần truyền vào
# return_train_score=True: trả về score của tập training
# best_estimator_ : tìm được các tham só cho score cao nhất

if run_new:
    randforest_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = [
            {'n_estimators':[10, 20, 50, 100, 150],
            'max_depth': [None, 3, 5, 10, 15],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 5, 10, 15],
            'max_leaf_nodes': [None, 10, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2']}
        ]
    grid_search = GridSearchCV(randforest_regressor, param_grid, cv=3, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    randforest_regressor = grid_search.best_estimator_
    joblib.dump(grid_search,'./saved_var/grid_search_randforestRegressor')
else:
    randforest_regressor = joblib.load('saved_var/grid_search_randforestRegressor').best_estimator_

# 4.2. Polynomial Model
if run_new:
    polynomial_regressor = Pipeline([ ('poly_feat_adder', PolynomialFeatures()), # add high-degree features
                       ('lin_reg', LinearRegression()) ]) 
    param_grid = [
        # try 3 values of degree
        {'poly_feat_adder__degree': [1, 2, 3]} ] # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
        # Train across 5 folds, hence a total of 3*5=15 rounds of training 
    grid_search = GridSearchCV(polynomial_regressor, param_grid, cv=3, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    polynomial_regressor = grid_search.best_estimator_
    joblib.dump(grid_search,'saved_var/grid_search_PolynomialRegression')   
else:
    polynomial_regressor = joblib.load('saved_var/grid_search_PolynomialRegression').best_estimator_


# 4.3. Decision tree model
# Ý nghĩa các hyperparameter được chọn
# max_depth: Chỉ định độ sâu tối đa của decision tree.
# min_samples_split: Chỉ định số sample tối thiểu cho phép để thuật toán tạo một node mới (decision boundary mới).
# min_samples_leaf: Chỉ định số sample tối thiểu trên một node lá.
# max_leaf_nodes: Chỉ định số node lá tối đa của decision tree.
# max_features: Chỉ định số lượng feature để xét tách node mới tốt nhất (None: max_features=n_features; sqrt: max_features=sqrt(n_features); log2: max_features=log2(n_features)).

#Nạp model Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
if run_new:
    model = DecisionTreeRegressor(random_state=42)
    param_grid = [
            {'max_depth': [None, 3, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 5, 10, 15, 20],
            'max_leaf_nodes': [None, 10, 20, 50, 100],
            'max_features': [None, 'sqrt', 'log2']}
        ]
    grid_search = GridSearchCV(model, param_grid, cv=3, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    decisiontree_regressor = grid_search.best_estimator_
    joblib.dump(grid_search,'saved_var/grid_search_decisiontree')
else:
    decisiontree_regressor = joblib.load('saved_var/grid_search_decisiontree').best_estimator_

def load_model(model_name):
    model = joblib.load('models/' + model_name + '_model.pkl') #load model với tham số tốt nhất
    return model



#%% 5. So sánh Polinomial, Decision tree và Random forest

def r2score_func(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    return r2score     

from sklearn.model_selection import cross_val_score
#cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
#cv2 = StratifiedKFold(n_splits=10, random_state=42); 
#cv3 = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
print('\n____________ K-fold cross validation ____________')

run_evaluation = 1
if run_evaluation:
    from sklearn.model_selection import KFold
    # NOTE: 
    #   + If data labels are float, cross_val_score use KFold() to split cv data.
    #   + KFold randomly splits data, hence does NOT ensure data splits are the same (only StratifiedKFold may ensure that)
    cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator: just a try to persist data splits (hopefully)
    # Evaluate Polinomial regression
    model_name = "PolinomialRegression" 
    model = joblib.load('saved_var/grid_search_PolynomialRegression').best_estimator_

    # khi dùng degree lớn hơn 1 thì sẽ cần add feature bậc cao
    # poly_feat_adder = PolynomialFeatures(degree = 1)
    # train_set_poly_added = poly_feat_adder.fit_transform(X_train)
    nmse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    print('On train:')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)))
    r2score = r2score_func(model, X_train, y_train)
    print('R2 score (on training data, best=1):', r2score, '\n')
    print('On test:')
    nmse_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)))
    r2score = r2score_func(model, X_test, y_test)
    print('R2 score (on test data, best=1):', r2score, '\n')

    model_name = "RandomForestRegressor" 
    model = joblib.load('./saved_var/grid_search_randforestRegressor').best_estimator_
    nmse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    print('On train:')
    print("Random forest regressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)))
    r2score = r2score_func(model, X_train, y_train)
    print('R2 score (on training data, best=1):', r2score, '\n')
    print('On test:')
    nmse_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    print("Random forest regressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)))
    r2score = r2score_func(model, X_test, y_test)
    print('R2 score (on test data, best=1):', r2score, '\n')


    model_name = "DecisionTreeRegressor" 
    model = joblib.load('saved_var/grid_search_decisiontree').best_estimator_  #lấy model đã lưu cùng với tham số tốt nhất
    #đưa vào model,feature,label, cv đã thực hiện KFold, scoring là chiến lược đánh giá
    nmse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    # tính root mean square error, vì negative âm nên ta lấy sqrt(-nmse_score)
    rmse_scores = np.sqrt(-nmse_scores)
    print('On train:')
    print("Decision tree regressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)))
    r2score = r2score_func(model, X_train, y_train)
    print('R2 score (on training data, best=1):', r2score, '\n')
    print('On test:')
    #đưa vào model,feature,label, cv đã thực hiện KFold, scoring là chiến lược đánh giá
    nmse_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring='neg_mean_squared_error')
    # tính root mean square error, vì negative âm nên ta lấy sqrt(-nmse_score)
    rmse_scores = np.sqrt(-nmse_scores)
    print("Decision tree regressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)))
    r2score = r2score_func(model, X_test, y_test)
    print('R2 score (on test data, best=1):', r2score, '\n')

# %%
