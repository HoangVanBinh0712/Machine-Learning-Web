#%%
from sklearn.pipeline import FeatureUnion
from statistics import mean
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder  
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import joblib 

#%% 1.Read data
df_main = pd.read_csv('./Car-details-v3.csv') # Load dữ liệu từ file Car-details-v3.csv
df_main.head()

df_main.shape

df_main.info()

df_main.describe()

df_main.drop('torque',axis=1,inplace = True) # bỏ những column không sử dụng trong bài toán này



#%% 2. Chuan Hoa
# trong data của nhóm em sẽ có những cột chưa được đưa về dạng số thuần mà còn các đơn vị khác trong đó VD: 120 CC, 72 bhp,..
# vì vậy nhóm em sẽ tiến hành chuẩn hóa lại các cột đó để phục vụ cho việc giải quyết bài toán.
# chuyển year sang age
df_main['age'] = 2022 - df_main['year']  # ở đây nhóm em sẽ chuyển cột year thành cột age bằng cách lấy năm hiện tại (2022) trừ đi giá trị trong cột year
df_main.drop('year',axis=1,inplace = True)  # sau khi đã có cột age thì loại bỏ cột year ban đầu.

#Engine to engine_cc
# cột engine có định dạng '120 CC' vì vậy ta tiến hành tách những phần nằm trước dấu ' ', astype dùng để đưa dữ liệu cột về dạng string và split dùng để tách và lấy những phần trước dấu ' ' 
engine_CC = df_main['engine'].astype(str).apply(lambda x : x.split(' ')[0]) 
df_main.insert(10,"engine_CC",engine_CC) # ta insert cột engine_CC vừa tách vào trong dataframe  
# sau khi tách thì cột engine có dạng '120' nên ta dùng astype để ép về kiểu float.
df_main['engine_CC'] = df_main['engine_CC'].astype(float)
df_main.drop('engine',axis=1,inplace = True) #sau khi đã có cột engine được chuẩn hóa thành cột engine_CC thì ta bỏ cột engine ban đầu đi


# tương tự như cột engine thì 2 cột max_power và mileage cũng làm tương tự
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

# %% Thêm cột Brand
# cột name trong data của nhóm em là tên các loại xe nhưng vì có quá nhiều tên khác nhau nên khi xử lý sẽ rất mất thời gian nên nhóm em sẽ thêm cột brand thay thế cho cột name vì tên xe có thể nhiêu nhưng hãng xe sẽ ít hơn vì 1 hãng xe có thể có hơn 10 chiếc xe với tên khác nhau.
brand = df_main['name'].astype(str).str.split()  # tách tên của xe dựa vào dấu ' '
# sau khi xem dataset thì nhóm em thấy rằng mỗi tên đều bắt đầu từ tên hãng nên ta sẽ cho vòng lặp và add vào cột mới là brand
for i in range(0, df_main.shape[0]):  
    brand[i] = brand[i][0]
df_main.insert(1,"brand",brand) # insert cột brand vào dataframe
df_main.drop('name',axis=1,inplace = True)  # bỏ cột name ban đầu

df_main.info()



#%% 3. Tạo Pipeline và chia tập train, tập test
# X : Các feature khác selling_price, Y : selling_price

#ta sẽ lấy cột selling_price làm label và những feature còn lại sẽ nằm trong tập dữ liệu để sử dụng.
X = df_main.drop(columns="selling_price")  # bỏ cột selling_price trong tập data
y = df_main["selling_price"].copy() # lấy cột selling_price làm label.


class ColumnSelector(BaseEstimator, TransformerMixin):   # define một hàm dùng để lựa chọn các feature tùy ý.
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values        

# ta chia các feature trong tập dữ liệu thành 2 dàng:  dạng feature số và feature chữ.
# num_feat_names: đây là list các feature dạng số
num_feat_names = ['km_driven', 'mileage_kmpl', 'age', 'engine_CC', 'max_power_bhb', 'seats'] # =list(train_set.select_dtypes(include=[np.number]))
#cat_feat_names: list các feature dạng chữ.
cat_feat_names = ['fuel', 'seller_type', 'transmission', 'owner', 'brand'] # =list(train_set.select_dtypes(exclude=[np.number])) 

# 4.4.2 Pipeline for categorical features
# ở phần này chúng ta sẽ tạo pipeline để xử lý 2 dạng feature ở trên.
# cat_pipeline sẽ dùng để xử lý feature dạng chữ.
cat_pipeline = Pipeline([  
    ('selector', ColumnSelector(cat_feat_names)),  # dùng hàm ColumnSelector đã define để lấy ra giá trị của feature đượct truyền vào
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # hàm OneHotEncoder này sẽ biến những giá trị khá nhau trong 1 feature thành những feature mới dựa theo số lượng
                                    # VD: cột name có 3 tên khác nhau: A, B, C thì khi hàm OneEncoder hoạt động nó sẽ biến A,B,C thành 3 feature mới => từ feature name -> 3 feature: A, B, C
    ])    
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),   # lấy giá trị từ feature được truyền vào
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="mean", copy=True)), # ở hàm SimpleImputer sẽ biến những giá trị missing values thành giá trị trung bình của feature đó ở đây strategy=mean có nghĩa là lấy giá trị trung bình của toàn cột
                                                                                # VD: cột km có 4 giá trị lần lượt [1,2,3,4] và 1 giá trị missing values khi này hàm SimpleImputer sẽ thay thế giá trị missing values đó bàng giá trị trung bình của 4 giá trị kia cộng lại miss_val = (1+2+3+4)/4
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Standard Scaler sẽ lấy giá trị trong cột chia cho giá trị lớn nhất trong cột đó nhằm mục đích biến đổi sử chênh lẹch giá trị quá lớn giữa các feature
                                                                            # # khi làm như vậy thì khoảng giá trị nó sẽ nằm trong khoảng từ 0-1.
    ])  

#pipeline
# khi đã có 2 pipeline dùng để xử lý feature số và chữ thì ta kết hợp lại thành một full_pipline
full_pipeline = FeatureUnion(transformer_list=[  
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  
joblib.dump(full_pipeline, r'models/full_pipeline.pkl')
# dùng full_pipeline vừa tạo để xử lý data chuẩn bị cho việc training
processed_train_set_val = full_pipeline.fit_transform(X)
joblib.dump(X, r'models/data_to_fit_transform.pkl')  # lưu lại tập dữ liệu sau khi xử lý xuống thư mục models


# tách tập train và test
# sau khi đã có được dữ liệu đã được xứ lý ta chia thành 2 tập train_set và test_set để bắt đầu việc training
X_train, X_test, y_train, y_test = train_test_split(processed_train_set_val, y, test_size=0.2, random_state=42) # test_size =0.2 nghĩa là ta sẽ chia data ban đầu thành 80% cho tập train va 20% cho tập test


# %% thực hiện train model 
# store model and load model
def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_model.pkl')
def load_model(model_name):
    model = joblib.load('models/' + model_name + '_model.pkl')
    return model

new_run = True
if new_run == True :
    # khai báo model Random forest
    model = RandomForestRegressor(random_state=42) 
    # Train model
    model.fit(X_train, y_train)
    # Lưu model đã train
    store_model(model,"randforest_regressor")  # lưu model
else: 
    model = load_model("randforest_regressor") # nếu đã train rồi thì load model lên


# predict trên 9 samples đầu tập train
print("-------------------------Train set----------------------------")
print("\nPredictions: ", model.predict(X_train[0:9]).round(decimals=1))
print("Labels:      ", list(y_train[0:9]))

# predict trên 9 samples đầu tập test
print("-------------------------Test set----------------------------")
print("\nPredictions: ", model.predict(X_test[0:9]).round(decimals=1))
print("Labels:      ", list(y_test[0:9]))


# r2score, root mean square error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)  # dùng để tính r2score 
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)  # thực hiện dự đoán trên toàn bộ tập train đưa vào
    mse = mean_squared_error(labels, prediction) # dựa trên toàn bộ dự đoán cùng với labels --> tính mean square error
    rmse = np.sqrt(mse)  # lấy căn của mse --> ra rmse
    return r2score, rmse      


# Tính r2score , root mean square error cho 9 samples đầu tiên 
r2score, rmse = r2score_and_rmse(model, X_train, y_train)
print('\n____________ Randomforest regression ____________')
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
print("\nPredictions: ", model.predict(X_train[0:9]).round(decimals=1))
print("Labels:      ", list(y_train[0:9]))

#test
r2score, rmse = r2score_and_rmse(model, X_test, y_test)
print('\n____________ Randomforest regression ____________')
print('\nR2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
print("\nPredictions: ", model.predict(X_test[0:9]).round(decimals=1))
print("Labels:      ", list(y_test[0:9]))


# %% K-FOLD
#khai báo thư viện sử dụng cross_val_score từ sklearn
from sklearn.model_selection import cross_val_score
print('\n____________ K-fold cross validation ____________')

run_evaluation = 0 #Tham số chạy chương trình
if run_evaluation:
    # NOTE: 
    #   + If data labels are float, cross_val_score use KFold() to split cv data.
    #   + KFold randomly splits data, hence does NOT ensure data splits are the same (only StratifiedKFold may ensure that)
    #   chia dữ liệu đã xáo trộn thành n split 5 fold
    cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator: just a try to persist data splits (hopefully)
    # Evaluate RandomForest regression
    model_name = "RandomForestRegression" #Tên model
    model = joblib.load('models/best_randforest_regressor_model.pkl') #load model
    #Negative mean square error
    #đưa vào model,feature,label, cv đã thực hiện KFold, scoring là chiến lược đánh giá
    nmse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    #vì negative âm nên sqrt tính root mean square error 
    rmse_scores = np.sqrt(-nmse_scores)
    #lưu model xuông
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForest regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
else:
    rmse_scores = joblib.load('saved_objects/RandomForestRegression_rmse.pkl')


# %% ANALYZE   
# Chạy model Random forest đã fine tune bên model_selection
best_model = joblib.load('saved_var/grid_search_randforestRegressor')

print('\n____________ ANALYZE AND TEST YOUR SOLUTION ____________')
print('SOLUTION: ' , best_model.best_estimator_)


#%% Đánh giá lại best model
model = best_model.best_estimator_

r2score, rmse = r2score_and_rmse(model, X_train, y_train)
print('\n____________ Randomforest regression ____________')
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 5.4.3 Predict labels for some training instances
print("\nPredictions: ", model.predict(X_train[0:9]).round(decimals=1))
print("Labels:      ", list(y_train[0:9]))

#test
r2score, rmse = r2score_and_rmse(model, X_test, y_test)
print('\n____________ Randomforest regression ____________')
print('\nR2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 5.4.3 Predict labels for some training instances
print("\nPredictions: ", model.predict(X_test[0:9]).round(decimals=1))
print("Labels:      ", list(y_test[0:9]))


# %%
