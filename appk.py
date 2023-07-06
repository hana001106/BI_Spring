# 사용된 라이브러리들
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import calendar
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# import optuna

# 한글로 나오게
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['font.size'] = 15 
matplotlib.rcParams['axes.unicode_minus'] = False

# 앱의 타이틀 설정
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("기상 정보를 바탕으로 한 전력 수요 예측")

#train_data (전력 수요)
data = pd.read_csv("전기.csv", encoding = "UTF-8")
data["기준일시"] = pd.to_datetime(data["기준일시"])
data = data[['기준일시', '현재수요(MW)']]

df = pd.read_csv("전기2.csv")
df["기준일시"] = pd.to_datetime(df["기준일시"])
df = df[['기준일시', '현재수요(MW)']]
df = df[df['기준일시'] >= '2022-04-08']
supply = pd.concat([data,df],axis=0)

#test_data(전력 수요)
supply_want = pd.read_csv("elec.csv")
supply_want = supply_want.rename(columns ={"현재부하(MW)":"현재수요(MW)"})
supply_want = supply_want.loc[:, ["일시", "현재수요(MW)"]]
supply_want["일시"] = pd.to_datetime(supply_want["일시"])
for i in range(len(supply_want)):
    supply_want.loc[i, "현재수요(MW)"] = int(supply_want["현재수요(MW)"].str.split(",")[i][0]+supply_want["현재수요(MW)"].str.split(",")[i][1])
supply_want["현재수요(MW)"] = supply_want["현재수요(MW)"].astype(int)

#train_data(날씨)
weather = pd.read_csv("서울기상자료_5분단위.csv")
weather["일시"] = pd.to_datetime(weather["일시"])

#test_data(날씨)
weather_want = pd.read_csv("기상.csv", encoding = "UTF-8")
weather_want["일시"] = pd.to_datetime(weather_want["일시"])
weather_want = weather_want[weather_want["일시"].between("2023-03-13", "2023-03-19 23:59")]
weather_want['요일'] = weather_want["일시"].apply(lambda x: x.weekday())
weather_want['월'] = weather_want["일시"].apply(lambda x: x.month)
weather_want.index = weather_want['일시']
weather_want.drop(columns=['일시'], inplace=True)
weather_want = weather_want.resample('5T').mean()

#날씨 전력 데이터 하나로 합치기(train)
df = pd.merge(weather, supply, left_on='일시', right_on='기준일시')
df.drop(columns=['기준일시'], inplace=True)


#날씨 전력 데이터 하나로 합치기(test)
df_want = pd.merge(weather_want, supply_want, left_on='일시', right_on='일시')

#요일, 월, 주말 변수 추가
df['요일'] = df["일시"].apply(lambda x: x.weekday())
df['월'] = df["일시"].apply(lambda x: x.month)

df["주말"] = 0
for i in range(len(df["요일"])):
    if (df["요일"][i]==5 or df["요일"][i]==6):
        df.iloc[i, -1] = 1

df_want["주말"] = 0
for i in range(len(df_want["요일"])):
    if (df_want["요일"][i]==5 or df_want["요일"][i]==6):
        df_want.iloc[i, -1] = 1

# 이슬점 변수 추가
def calculate_dew_point(temperature, relative_humidity):
    a = 17.27
    b = 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(relative_humidity/100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

df["이슬점"] = calculate_dew_point(df['기온(°C)'], df['습도(%)'])
df_want["이슬점"] = calculate_dew_point(df_want['기온(°C)'], df_want['습도(%)'])

# 낮, 밤 변수 추가
night = pd.read_csv("sun_df.csv", encoding = "UTF-8")
night["일시"] = pd.to_datetime(night["일시"])+ pd.to_timedelta('00:00:00')
night.index = night["일시"]
night = night.iloc[:, 1:]
night = night[night["일시"].between("2020-02-01", "2023-03-12")]

night['일출'] = pd.to_datetime(night['일출'], format='%H:%M').dt.time
night['일몰'] = pd.to_datetime(night['일몰'], format='%H:%M').dt.time
df['낮밤'] = df['일시'].apply(lambda x: 1 if night['일출'].max() < x.time() < night['일몰'].min() else 0)

sun = pd.read_csv("sun_df.csv", encoding = "UTF-8")
sun["일시"] = pd.to_datetime(sun["일시"])+ pd.to_timedelta('00:00:00')
sun = sun.iloc[:, 1:]
sun = sun[sun["일시"].between("2023-03-13", "2023-03-19")]

sun['일출'] = pd.to_datetime(sun['일출'], format='%H:%M').dt.time
sun['일몰'] = pd.to_datetime(sun['일몰'], format='%H:%M').dt.time
df_want['낮밤'] = df_want['일시'].apply(lambda x: 1 if sun['일출'].max() < x.time() < sun['일몰'].min() else 0)

#전처리 
df2 = df.copy()
df2.index = df2['일시']
df2.drop(columns=['일시'], inplace=True)
df3 = df2[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)','풍향(deg)','풍속(m/s)', '습도(%)', '일사(MJ/m^2)', '일조(Sec)', '현재수요(MW)','이슬점','낮밤','요일','월','주말']].interpolate(method = "nearest")

df4 = df_want.copy()
df4.index = df4['일시']
df4.drop(columns=['일시'], inplace=True)
df5 = df4[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)','풍향(deg)','풍속(m/s)', '습도(%)', '일사(MJ/m^2)', '일조(Sec)', '현재수요(MW)','이슬점','낮밤','요일','월','주말']].interpolate(method = "nearest")

#정상성 검정
#기온
df3["기온"] = df3["기온(°C)"].diff(12)
df5["기온"] = df5["기온(°C)"].diff(12)

#현지기압+해면기압
df3["현지기압"] = np.log(df3["현지기압(hPa)"])
df3["현지기압"] = df3["현지기압"].diff(12)
df3["해면기압"] = np.log(df3["해면기압(hPa)"])
df3["해면기압"] = df3["해면기압"].diff(12)

df5["현지기압"] = np.log(df5["현지기압(hPa)"])
df5["현지기압"] = df5["현지기압"].diff(12)
df5["해면기압"] = np.log(df5["해면기압(hPa)"])
df5["해면기압"] = df5["해면기압"].diff(12)

#풍향+풍속
df3["풍향"] = df3["풍향(deg)"].diff(1)
df5["풍향"] = df5["풍향(deg)"].diff(1)
df3["풍속"] = df3["풍속(m/s)"].diff(1)
df5["풍속"] = df5["풍속(m/s)"].diff(1)

#일사
df3["일사"] = df3['일사(MJ/m^2)'].diff(12)
df5["일사"] = df5['일사(MJ/m^2)'].diff(12)

#이슬점
df3["이슬점"] = df3['이슬점'].diff(12)
df5["이슬점"] = df5['이슬점'].diff(12)

#습도
df3["습도"] = df3['습도(%)'].diff(12)
df5["습도"] = df5['습도(%)'].diff(12)

#피쳐 선택
df6 = df3[['기온', '습도', '현지기압', '해면기압', '풍향', '풍속', '일사', '일조(Sec)', '현재수요(MW)','이슬점', '주말', '월','낮밤']]
df7 = df5[['기온', '습도', '현지기압', '해면기압', '풍향', '풍속', '일사', '일조(Sec)', '현재수요(MW)','이슬점', '주말', '월','낮밤']]

#테스트, 훈련 데이터 분리
train_data = df6
test_data = df7
X_train=train_data.drop('현재수요(MW)', axis=1)
y_train=train_data[['현재수요(MW)']]
X_test=test_data.drop('현재수요(MW)', axis=1)
y_test=test_data[['현재수요(MW)']]

# def objective(trial: optuna.Trial):
#     param = {
#         'iterations': trial.suggest_int('iterations', 100, 1000),
#         'depth': trial.suggest_int('depth', 3, 8),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
#         'random_state': 0,
#         'cat_features': cat_cols
#     }

#     model = CatBoostRegressor(**param)
#     cat_model = model.fit(X_train, y_train, verbose=False)

#     pred = cat_model.predict(X_test)
#     mse = mean_squared_error(y_test, pred)

#     return mse

# # Convert categorical columns to string type
# cat_cols = ['기온', '현지기압', '해면기압', '풍향', '풍속', '일사', '일조(Sec)', '이슬점', '요일', '월']
# X_train[cat_cols] = X_train[cat_cols].astype(str)
# X_test[cat_cols] = X_test[cat_cols].astype(str)

# # Create an Optuna study
# study = optuna.create_study(direction='minimize')

# # Run the optimization process
# study.optimize(objective, n_trials=5)

# # Print the best score and the best parameters found
# best_score = study.best_value
# best_params = study.best_params

date_string = st.slider("날짜를 선택하세요",
              min_value = datetime(2023,3,13,0,0,0), 
              max_value = datetime(2023,3,19,0,0,0),
              step = timedelta(minutes = 5),
              format = "MM-DD-YY - HH:mm"
)

X_test = X_test.loc[date_string:date_string+timedelta(hours=6)]
y_test = y_test.loc[date_string:date_string+timedelta(hours=6)]

#모델링
# 데이터 정규화
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LGBMRegressor 모델 생성
lgbm_model = LGBMRegressor(n_estimators =1000)

# CatBoostRegressor 모델 생성
cat_model = CatBoostRegressor(iterations=719, learning_rate= 0.08491471282079409, depth=4)

# 앙상블 모델 생성
ensemble_model = VotingRegressor([('lgbm', lgbm_model), ('cat', cat_model)],verbose = 0)

ensemble_model.fit(X_train_scaled, y_train)
ensemble_predictions = ensemble_model.predict(X_test_scaled)
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_predictions)

y_pred = y_test.copy()
y_pred["현재수요(MW)"] = ensemble_predictions

# model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=4)
# model.fit(X_train, y_train)
# y_pred = y_test.copy()
# y_pred['현재수요(MW)'] = 0
# y_pred['현재수요(MW)'] = model.predict(X_test)

button = st.button("Mape 수치를 확인해보세요")
if button:
    st.markdown(mean_absolute_percentage_error(y_test, ensemble_predictions))

# 시각화
legend = ["observations", "median prediction"]

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
y_test.plot(ax=ax)
y_pred.plot(ax=ax)
plt.grid(which="both")
plt.legend(legend, loc="upper left")

plt.xlabel("Timestamp")
plt.xticks(rotation=10)
plt.ylabel("Electricity Consumption")
plt.title("CatBoostRegressor Electricity Consumption Forecast", fontsize=15)
st.pyplot(fig)