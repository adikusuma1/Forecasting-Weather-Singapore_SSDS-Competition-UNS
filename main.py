import os
import cv2
import shutil
import re
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import optuna
from xgboost import XGBRegressor, plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

src = "Train"
src_test = 'Test'

from sklearn.impute import KNNImputer

class RainfallImputer:
    def __init__(self, strategy=None, n_neighbors=3):
        self.strategy = strategy or {
            'Highest_30min_Rainfall_mm': 'median',
            'Highest_60min_Rainfall_mm': 'median',
            'Highest_120min_Rainfall_mm': 'median',
            'Mean_Temperature_C': 'mean',
            'Max_Temperature_C': 'mean',
            'Min_Temperature_C': 'mean',
            'Mean_Wind_Speed_kmh': 'median',
            'Max_Wind_Speed_kmh': 'median'
        }
        self.n_neighbors = n_neighbors
        self.impute_values = {}
        self.global_values = {}
        self.fitted = False
    
    def fit(self, df):
        df = df.copy()
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        self.min_year = df['year'].min()
        self.max_year = df['year'].max()

        for col in self.strategy.keys():
            print(f"Preparing historical data for KNN imputation: {col}")
            col_impute_values = {}

            for y in range(self.min_year, self.max_year + 1):
                for w in range(1, 6):
                    hist_year = y - w
                    if hist_year < self.min_year:
                        continue
                    
                    df_hist = df[df['year'] == hist_year][['city', 'month', col]].dropna()
                    if df_hist.empty:
                        continue

                    key = f"{hist_year}-{y}"
                    if key not in col_impute_values:
                        col_impute_values[key] = df_hist.copy()
                    else:
                        col_impute_values[key] = pd.concat([col_impute_values[key], df_hist], ignore_index=True)

            self.impute_values[col] = col_impute_values

        # Fallback values
        for col, method in self.strategy.items():
            city_month_values = df.groupby(['city', 'month'])[col].agg(method).reset_index()
            self.impute_values[col]['city_month'] = city_month_values
            self.global_values[col] = df[col].median() if method == 'median' else df[col].mean()

        self.fitted = True
        return self
    
    def transform(self, df):
        if not self.fitted:
            raise ValueError("Imputer belum di-fit. Panggil fit() terlebih dahulu.")
        
        df = df.copy()
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year

        for col in self.strategy.keys():
            print(f"Applying KNN historical imputation for: {col}")
            filled_col = df[col].copy()

            for y in df['year'].unique():
                for w in range(1, 6):
                    hist_year = y - w
                    key = f"{hist_year}-{y}"

                    if key not in self.impute_values[col]:
                        continue

                    hist_df = self.impute_values[col][key]
                    if hist_df.empty:
                        continue

                    mask_target = (df['year'] == y) & (df[col].isnull())
                    target_df = df.loc[mask_target, ['city', 'month']]

                    if target_df.empty:
                        continue

                    combined = pd.concat([
                        hist_df,
                        pd.DataFrame({'city': target_df['city'], 'month': target_df['month'], col: np.nan})
                    ], ignore_index=True)

                    combined_encoded = pd.get_dummies(combined[['city', 'month']])
                    combined_encoded[col] = combined[col].values

                    imputer = KNNImputer(n_neighbors=self.n_neighbors)
                    imputed_array = imputer.fit_transform(combined_encoded)

                    imputed_values = imputed_array[-len(target_df):, -1] 
                    filled_col.loc[mask_target] = imputed_values

            df[col] = filled_col

        for col in self.strategy.keys():
            print(f"Applying fallback imputation for: {col}")
            city_month_values = self.impute_values[col]['city_month']
            df = df.merge(city_month_values, on=['city', 'month'], how='left', suffixes=('', '_impute'))
            df[col] = df[col].fillna(df[f'{col}_impute'])
            df.drop(columns=[f'{col}_impute'], inplace=True)

            df[col] = df[col].fillna(self.global_values[col])

        print("Missing values after imputation:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

        return df

    def save_imputer(self, filepath):
        if not self.fitted:
            raise ValueError("Imputer belum di-fit. Panggil fit() terlebih dahulu.")
        
        data_to_save = {
            'strategy': self.strategy,
            'n_neighbors': self.n_neighbors,
            'global_values': self.global_values,
            'impute_values': {
                col: {
                    key: df.to_dict('records') if isinstance(df, pd.DataFrame) else df
                    for key, df in hist.items()
                }
                for col, hist in self.impute_values.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data_to_save, f)

    @classmethod
    def load_imputer(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        imputer = cls(strategy=data['strategy'], n_neighbors=data['n_neighbors'])
        imputer.global_values = data['global_values']
        imputer.impute_values = {
            col: {
                key: pd.DataFrame(records) if isinstance(records, list) else records
                for key, records in hist.items()
            }
            for col, hist in data['impute_values'].items()
        }
        imputer.fitted = True
        return imputer
    
# ============
# TRAIN DATA
# ============
def process_image(img_path, city_name, year):
    img = cv2.imread(img_path)
    if img is None: return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    mask = cv2.inRange(img, np.array([165,105,20]), np.array([195, 135,40]))
    points = cv2.findNonZero(mask)
    
    plot_area_mask = cv2.inRange(img, np.array([200, 200, 200]), np.array([255, 255, 255]))
    
    rain_data = np.full(365, np.nan)
    
    X_LEFT, X_RIGHT = 100, width - 100
    Y_TOP, Y_BOTTOM = 50, height - 80
    x_scale = (X_RIGHT - X_LEFT)/365
    y_max_rain = 150
    
    baseline = Y_BOTTOM
    
    if points is not None:
        points = points[:,0,:]
        for x, y in points:
            if X_LEFT <= x <= X_RIGHT and Y_TOP <= y <= Y_BOTTOM:
                day = int((x - X_LEFT) / x_scale)
                if 0 <= day < 365:
                    rainfall = ((baseline - y) / (baseline - Y_TOP)) * y_max_rain
                    rainfall = max(0, rainfall) 
                    if np.isnan(rain_data[day]) or rainfall > rain_data[day]:
                        rain_data[day] = rainfall

    for day in range(365):
        if np.isnan(rain_data[day]):
            x = int(X_LEFT + day * x_scale)
            if X_LEFT <= x <= X_RIGHT:
                vertical_line = plot_area_mask[Y_TOP:Y_BOTTOM, x]
                if np.any(vertical_line > 0): 
                    rain_data[day] = 0  
    
    start_date = datetime(year, 1, 1)
    return [
        [f"{city_name}_{(start_date + timedelta(days=i)).strftime('%Y_%m_%d')}",
         (start_date + timedelta(days=i)).strftime("%Y-%m-%d"),
         round(rain_data[i], 2) if not np.isnan(rain_data[i]) else np.nan,
         city_name]
        for i in range(365)
    ]

combined_data = []

for city_folder in os.listdir(src):
    city_path = os.path.join(src, city_folder)
    if not os.path.isdir(city_path): continue

    city_name = city_folder.lower()
    for img_file in sorted(os.listdir(city_path)):
        if img_file.startswith("Plot_Daily_Rainfall_") and img_file.endswith(".png"):
            try:
                year = int(img_file.split("_")[-1].split(".")[0])
                if data := process_image(os.path.join(city_path, img_file), city_name, year):
                    combined_data.extend(data)
            except ValueError:
                continue

if combined_data:
    train_df = pd.DataFrame(combined_data, columns=['ID', 'date', 'prediksi', 'city'])
    train_df.to_csv('train_data.csv', index=False)

train_df = pd.read_csv('train_data.csv')
train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df[train_df['date'].dt.year >= 1982]
train_df['month'] = train_df['date'].dt.month
train_df.to_csv('train_data.csv', index=False)

train_df=pd.read_csv('train_data.csv')
train_df['date'] = pd.to_datetime(train_df['date'])
weather_data_list = []

for city_folder in os.listdir(src):
    city_path=os.path.join(src, city_folder)
    if not os.path.isdir(city_path):continue
    city_name = city_folder.lower()

    for filename in os.listdir(city_path):
        if filename.startswith('Data_Gabungan_Lainnya_') and filename.endswith('.csv'):
            try:
                year = int(filename.split('_')[-1].split('.')[0])
                file_path = os.path.join(city_path, filename)
                weather_df = pd.read_csv(file_path)
                
                weather_df.columns = [col.replace('min', 'Min') for col in weather_df.columns]
                weather_df.columns = [col.strip() for col in weather_df.columns]
                
                # Convert date and add merge keys
                weather_df['Date'] = pd.to_datetime(weather_df['Date'])
                weather_df['city'] = city_name
                weather_df['year'] = weather_df['Date'].dt.year
                weather_df['month'] = weather_df['Date'].dt.month
                weather_df['day'] = weather_df['Date'].dt.day
                
                weather_data_list.append(weather_df)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

if weather_data_list:
    all_weather_data = pd.concat(weather_data_list, ignore_index=True)
    duplicate_cols = all_weather_data.columns[all_weather_data.columns.duplicated()]
    if len(duplicate_cols)>0:
        all_weather_data = all_weather_data.loc[:,~all_weather_data.columns.duplicated()]

    train_df['year'] = train_df['date'].dt.year
    train_df['month'] = train_df['date'].dt.month
    train_df['day'] = train_df['date'].dt.day        

    merge_df = pd.merge(
        train_df,
        all_weather_data,
        how='left',
        on=['city','year','month','day']
    )

    merge_df = merge_df.drop(columns='Date')
    dup_cols = set([col for col in merge_df.columns if merge_df.columns.tolist().count(col) > 1])
    if dup_cols:
        print(f"Duplicate columns in final data: {dup_cols}")
        merge_df = merge_df.loc[:,~merge_df.columns.duplicated()]
    
    merge_df.to_csv('train_data.csv', index=False)

train_df = pd.read_csv('train_data.csv')
train_df['date']=pd.to_datetime(train_df['date'])
train_df['year_month']=train_df['date'].dt.to_period('M')

dmi=pd.read_csv('Data Eksternal/Dipole Mode Index (DMI).csv')
dmi['Date'] = pd.to_datetime(dmi['Date'])
dmi['year_month'] = dmi['Date'].dt.to_period('M')
dmi = dmi.rename(columns={' DMI HadISST1.1  missing value -9999 https://psl.noaa.gov/data/timeseries/month/':'DMI'})[['year_month','DMI']]

oni=pd.read_csv('Data Eksternal/OceanicNinoIndex (ONI).csv')
oni['Date'] = pd.to_datetime(oni['Date'], format='%d/%m/%Y')
oni['year_month'] = oni['Date'].dt.to_period('M')
oni = oni.rename(columns={'  ONI':'ONI'})[['year_month','ONI']]

humi=pd.read_csv('Data Eksternal/RelativeHumidityMonthlyMean.csv')
humi['year_month']=pd.to_datetime(humi['month']).dt.to_period('M')
humi = humi[['year_month','mean_rh']]

climate_data = dmi.merge(oni, on='year_month', how='outer')
climate_data = climate_data.merge(humi, on='year_month', how='outer')
train_df = train_df.merge(climate_data, on='year_month', how='left')

train_df = train_df.drop(columns='year_month')
train_df.to_csv('train_data.csv', index=False)

train_df.info()

train_df = pd.read_csv('train_data.csv')

columns_with_nulls = [
    'Highest 30 Min Rainfall (mm)',
    'Highest 60 Min Rainfall (mm)',
    'Highest 120 Min Rainfall (mm)',
    'Mean Temperature (°C)',
    'Maximum Temperature (°C)',
    'Minimum Temperature (°C)',
    'Mean Wind Speed (km/h)',
    'Max Wind Speed (km/h)'
]

for col in columns_with_nulls:
    train_df[col] = train_df[col].replace(['', ' ', 'NA', 'N/A', '-', 'NaN', 'null'], np.nan)
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

col_to_convert = [
    'prediksi',
    'Highest 30 Min Rainfall (mm)',
    'Highest 60 Min Rainfall (mm)',
    'Highest 120 Min Rainfall (mm)',
    'Mean Temperature (°C)',
    'Maximum Temperature (°C)',
    'Minimum Temperature (°C)',
    'Mean Wind Speed (km/h)',
    'Max Wind Speed (km/h)'
]
def convert_to_float(x):
    return float(x)

for col in col_to_convert:
    train_df[col] = train_df[col].apply(convert_to_float)

null_sum = train_df.isnull().sum()
null_percent = (train_df.isnull().mean()*100).round(2)
null_report = pd.DataFrame({
    'Null Count' : null_sum,
    'Null Percentage' : null_percent
}).sort_values('Null Percentage', ascending=False)
print(null_report[null_report['Null Count']>0])

rainfall_cols = [
    'Highest 30 Min Rainfall (mm)',
    'Highest 60 Min Rainfall (mm)',
    'Highest 120 Min Rainfall (mm)'
]

condition_to_drop = (
    train_df[rainfall_cols].isnull().all(axis=1) & 
    (train_df['prediksi'] == 0.0)
)
condition_to_keep = (
    train_df[rainfall_cols].notnull().any(axis=1) & 
    (train_df['prediksi'] == 0.0)
)


print(f"Jumlah data yang akan dihapus: {condition_to_drop.sum()}")
print(f"Jumlah data dengan prediksi 0.0 yang dipertahankan: {condition_to_keep.sum()}")

indices_to_drop = train_df[condition_to_drop].index
train_df = train_df.drop(indices_to_drop)

print("\nSetelah penghapusan:")
print(f"Total data dengan prediksi 0.0: {(train_df['prediksi'] == 0.0).sum()}")
print(f"Data dengan prediksi 0.0 dan semua kolom rainfall null: {(train_df[rainfall_cols].isnull().all(axis=1) & (train_df['prediksi'] == 0.0)).sum()}")

train_df.to_csv('train_data.csv', index=False)

def extract_city(id_string):
    parts=id_string.split('_')
    year_idx = next((i for i, part in enumerate(parts) if part.isdigit() and len(part) == 4), None)

    if year_idx is not None:
        city = '_'.join(parts[:year_idx])
    else:
        city = '_'.join(parts[:-3])
    return city

train_df['city'] = train_df['ID'].apply(extract_city)
train_df['month'] = pd.to_datetime(train_df['date']).dt.month
le = LabelEncoder()
train_df['city_encoded'] = le.fit_transform(train_df['city'])

train_df = train_df.rename(columns={'date':'Date'})
train_df = train_df.rename(columns={'Highest 30 Min Rainfall (mm)':'Highest_30min_Rainfall_mm'})
train_df = train_df.rename(columns={'Highest 60 Min Rainfall (mm)':'Highest_60min_Rainfall_mm'})
train_df = train_df.rename(columns={'Highest 120 Min Rainfall (mm)':'Highest_120min_Rainfall_mm'})
train_df = train_df.rename(columns={'Mean Temperature (°C)':'Mean_Temperature_C'})
train_df = train_df.rename(columns={'Maximum Temperature (°C)':'Max_Temperature_C'})
train_df = train_df.rename(columns={'Minimum Temperature (°C)':'Min_Temperature_C'})
train_df = train_df.rename(columns={'Mean Wind Speed (km/h)':'Mean_Wind_Speed_kmh'})
train_df = train_df.rename(columns={'Max Wind Speed (km/h)':'Max_Wind_Speed_kmh'})

desired_order = [
    'ID', 'Date', 'month', 'city', 'city_encoded',
    'Highest_30min_Rainfall_mm', 'Highest_60min_Rainfall_mm', 'Highest_120min_Rainfall_mm',
    'Mean_Temperature_C', 'Max_Temperature_C', 'Min_Temperature_C',
    'Mean_Wind_Speed_kmh', 'Max_Wind_Speed_kmh',
    'DMI', 'ONI', 'mean_rh',
    'prediksi'
]

train_df = train_df[desired_order]
train_df.to_csv('train_data.csv', index=False)

def detect_outliers(series, method='iqr', threshold = 1.5):
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold*IQR
        upper_bound = Q3 - threshold*IQR
        return (series<lower_bound) | (series>upper_bound)
    elif method == 'zscore':
        z_score = np.abs(stats.zscore(series))
        return z_score>threshold
    
def handle_outliers(train_df, col, method='cap', **kwargs):
    if method == 'cap':
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - kwargs.get('threshold', 1.5)*IQR
        upper_bound = Q3 + kwargs.get('threshold', 1.5)*IQR
        train_df[col] = train_df[col].clip(lower_bound, upper_bound)
    elif method=='remove':
        outliers=detect_outliers(train_df[col], **kwargs)
        train_df = train_df[~outliers]
    elif method=='transform':
        train_df[col]=np.log1p(train_df[col])
    return train_df

outlier_strategy = {
    'Mean_Temperature_C': {'method': 'cap', 'threshold': 2.5, 'detect_method': 'zscore'},
    'Max_Temperature_C': {'method': 'cap', 'threshold': 2.5, 'detect_method': 'zscore'},
    'Min_Temperature_C': {'method': 'cap', 'threshold': 2.5, 'detect_method': 'zscore'},
    'Max_Wind_Speed_kmh': {'method': 'cap', 'threshold': 3, 'detect_method': 'iqr'}
}

for col, params in outlier_strategy.items():
    is_outlier = detect_outliers(
        train_df[col],
        method=params['detect_method'],
        threshold=params['threshold']
    )
    print(f"Jumlah outlier ditemukan: {is_outlier.sum()}")

    plt.figure(figsize=(10, 4))
    plt.boxplot(train_df[col].dropna())
    plt.title(f'Before Outlier Handling - {col}')
    plt.show()

    train_df = handle_outliers(train_df, col, **params)
    plt.figure(figsize=(10, 4))
    plt.boxplot(train_df[col].dropna())
    plt.title(f'After Outlier Handling - {col}')
    plt.show()

print(train_df[list(outlier_strategy.keys())].describe())

train_df.to_csv("train_data.csv", index=False)

# =============
# TESTING DATA
# =============
def preprocess_dmi_oni(input_file, output_file, value_col, date_col='Date', missing_val=-9999):
    df = pd.read_csv(input_file)

    if df[date_col].str.contains('/').any():  
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
    else:
        df[date_col] = pd.to_datetime(df[date_col])
    
    df[value_col] = df[value_col].replace(missing_val, np.nan)
    
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    imputer = KNNImputer(n_neighbors=5)
    features = ['year', 'month_sin', 'month_cos', value_col]
    
    temp_df = df[features].copy()
    imputed_values = imputer.fit_transform(temp_df)
    
    df[value_col] = imputed_values[:, -1] 
    
    df.to_csv(output_file, index=False, columns=[date_col, value_col])
    print(f"Data berhasil diproses dan disimpan di {output_file}")

preprocess_dmi_oni('Data Eksternal/Dipole Mode Index (DMI).csv', 'DMI_imputed.csv', ' DMI HadISST1.1  missing value -9999 https://psl.noaa.gov/data/timeseries/month/')
preprocess_dmi_oni('Data Eksternal/OceanicNinoIndex (ONI).csv', 'ONI_imputed.csv', '  ONI', date_col='Date')

all_test_data = []

for city_folder in os.listdir(src_test):
    city_path = os.path.join(src_test, city_folder)
    if not os.path.isdir(city_path): continue

    for filename in os.listdir(city_path):
        if filename.startswith('Data_Gabungan_Lainnya_') and filename.endswith('.csv'):
            file_path = os.path.join(city_path, filename)
            year=filename.split('_')[-1].split('.')[0]

            try:
                test_df=pd.read_csv(file_path)
                test_df.columns = [col.replace('min','Min').strip() for col in test_df.columns]

                test_df['city'] = city_folder.lower()
                test_df['year'] = year

                if 'Date' in test_df.columns:
                    test_df['Date'] = pd.to_datetime(test_df['Date'])
                
                all_test_data.append(test_df)

            except Exception as e:
                print(f"{filename}:{str(e)}")
                continue

if all_test_data:
    test_df=pd.concat(all_test_data, ignore_index=True)
    test_df.to_csv('test_data.csv', index=False)
    
test_df.info()

test_df = pd.read_csv('test_data.csv')
test_df['year_month'] = pd.to_datetime(test_df['Date']).dt.to_period('M')

dmi = pd.read_csv('DMI_imputed.csv')
dmi['Date'] = pd.to_datetime(dmi['Date'])
dmi['year_month'] = dmi['Date'].dt.to_period('M')
dmi = dmi.rename(columns={' DMI HadISST1.1  missing value -9999 https://psl.noaa.gov/data/timeseries/month/':'DMI'})[['year_month','DMI']]

oni = pd.read_csv('ONI_imputed.csv')
oni['Date'] = pd.to_datetime(oni['Date'])
oni['year_month'] = oni['Date'].dt.to_period('M')
oni = oni.rename(columns={'  ONI':'ONI'})[['year_month','ONI']]

humi = pd.read_csv('Data Eksternal/RelativeHumidityMonthlyMean.csv')
humi['year_month'] = pd.to_datetime(humi['month']).dt.to_period('M')
humi = humi[['year_month', 'mean_rh']]

climate_data = dmi.merge(oni, on='year_month', how='outer')
climate_data = climate_data.merge(humi, on='year_month', how='outer')
test_df = test_df.merge(climate_data, on='year_month', how='left')

test_df.to_csv('test_data.csv', index=False)
test_df.info()

test_df = pd.read_csv('test_data.csv')
test_df.info()

columns_with_nulls = [
    'Highest 30 Min Rainfall (mm)',
    'Highest 60 Min Rainfall (mm)',
    'Highest 120 Min Rainfall (mm)',
    'Mean Temperature (°C)',
    'Maximum Temperature (°C)',
    'Minimum Temperature (°C)',
    'Mean Wind Speed (km/h)',
    'Max Wind Speed (km/h)',
    'DMI', 'ONI', 'mean_rh'
]

for col in columns_with_nulls:
    test_df[col] = test_df[col].replace(['', ' ', 'NA', 'N/A', '-', 'NaN', 'null'], np.nan)
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

col_to_convert = [
    'Highest 30 Min Rainfall (mm)',
    'Highest 60 Min Rainfall (mm)',
    'Highest 120 Min Rainfall (mm)',
    'Mean Temperature (°C)',
    'Maximum Temperature (°C)',
    'Minimum Temperature (°C)',
    'Mean Wind Speed (km/h)',
    'Max Wind Speed (km/h)'
]
def convert_to_float(x):
    return float(x)

for col in col_to_convert:
    test_df[col] = test_df[col].apply(convert_to_float)

null_sum = test_df.isnull().sum()
null_percent = (test_df.isnull().mean()*100).round(2)
null_report = pd.DataFrame({
    'Null Count' : null_sum,
    'Null Percentage' : null_percent
}).sort_values('Null Percentage', ascending=False)
print(null_report[null_report['Null Count']>0])

def extract_city(id_string):
    parts=id_string.split('_')
    year_idx = next((i for i, part in enumerate(parts) if part.isdigit() and len(part) == 4), None)

    if year_idx is not None:
        city = '_'.join(parts[:year_idx])
    else:
        city = '_'.join(parts[:-3])
    return city

test_df['month'] = pd.to_datetime(test_df['Date']).dt.month
le = LabelEncoder()
test_df['city_encoded'] = le.fit_transform(test_df['city'])

test_df = test_df.rename(columns={'Date':'Date'})
test_df = test_df.rename(columns={'Highest 30 Min Rainfall (mm)':'Highest_30min_Rainfall_mm'})
test_df = test_df.rename(columns={'Highest 60 Min Rainfall (mm)':'Highest_60min_Rainfall_mm'})
test_df = test_df.rename(columns={'Highest 120 Min Rainfall (mm)':'Highest_120min_Rainfall_mm'})
test_df = test_df.rename(columns={'Mean Temperature (°C)':'Mean_Temperature_C'})
test_df = test_df.rename(columns={'Maximum Temperature (°C)':'Max_Temperature_C'})
test_df = test_df.rename(columns={'Minimum Temperature (°C)':'Min_Temperature_C'})
test_df = test_df.rename(columns={'Mean Wind Speed (km/h)':'Mean_Wind_Speed_kmh'})
test_df = test_df.rename(columns={'Max Wind Speed (km/h)':'Max_Wind_Speed_kmh'})

desired_order = [
    'Date', 'month', 'city', 'city_encoded',
    'Highest_30min_Rainfall_mm', 'Highest_60min_Rainfall_mm', 'Highest_120min_Rainfall_mm',
    'Mean_Temperature_C', 'Max_Temperature_C', 'Min_Temperature_C',
    'Mean_Wind_Speed_kmh', 'Max_Wind_Speed_kmh',
    'DMI', 'ONI', 'mean_rh'
]

test_df = test_df[desired_order]
test_df.to_csv('test_data.csv', index=False)

# ===============
# BUILD MODEL
# ===============
def load_data():
    df = pd.read_csv('train_data.csv', parse_dates=['Date'])
    df['target'] = df['prediksi'] 
    df = df.dropna(subset=['target'])
    return df

def create_features(df, is_training=True):
    df['month'] = df['Date'].dt.month
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['is_monsoon'] = df['month'].isin([6,7,8,9]).astype(int)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    df['temp_range'] = df['Max_Temperature_C'] - df['Min_Temperature_C']
    df['temp_humidity'] = df['Mean_Temperature_C'] * df['mean_rh']
    df['rain_wind'] = df['Highest_30min_Rainfall_mm'] * df['Max_Wind_Speed_kmh']
    
    df['ENSO_phase'] = np.where(df['ONI'] > 0.5, 1, np.where(df['ONI'] < -0.5, -1, 0))
    
    for col in ['Highest_30min_Rainfall_mm', 'Mean_Temperature_C', 'Max_Wind_Speed_kmh']:
        for window in [3, 7, 14]:
            df[f'{col}_rolling_mean_{window}d'] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_rolling_max_{window}d'] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window, min_periods=1).max())
    
    if is_training:
        for lag in [1, 2, 3, 7, 14]:
            df[f'target_lag_{lag}'] = df.groupby('city')['target'].shift(lag)
    else:
        for lag in [1, 2, 3, 7, 14]:
            df[f'target_lag_{lag}'] = 0
    
    return df

def select_features(df):
    num_features = [
        'Highest_30min_Rainfall_mm', 'Mean_Temperature_C', 'Max_Wind_Speed_kmh',
        'DMI', 'ONI', 'mean_rh', 'temp_range', 'temp_humidity', 'rain_wind',
        'Highest_30min_Rainfall_mm_rolling_mean_7d',
        'Mean_Temperature_C_rolling_mean_7d',
        'Max_Wind_Speed_kmh_rolling_max_7d',
        'target_lag_1', 'target_lag_7',
        'month_sin', 'month_cos'
    ]
    
    cat_features = ['city_encoded', 'month', 'is_monsoon', 'ENSO_phase']
    
    return num_features, cat_features

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'early_stopping_rounds': 100, 
            'eval_metric': 'rmse', 
        }
        
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, val_pred))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=3600)
    return study.best_params

def main():
    df = load_data()
    df = create_features(df)
    
    num_features, cat_features = select_features(df)
    
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    
    df = df.sort_values(['city', 'Date'])
    split_date = pd.to_datetime('2022-01-01')
    train = df[df['Date'] < split_date].dropna(subset=num_features + ['target'])
    val = df[df['Date'] >= split_date].dropna(subset=num_features + ['target'])
    
    X_train, y_train = train[num_features + cat_features], train['target']
    X_val, y_val = val[num_features + cat_features], val['target']
    
    print("Tuning hyperparameters...")
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)
    print("Best parameters:", best_params)
    
    print("Training final model...")

    model = XGBRegressor(
        **best_params,
        early_stopping_rounds=100,
        eval_metric='rmse',
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"\nValidation MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=20, importance_type='gain')
    plt.title("Feature Importance (by Gain)")
    plt.tight_layout()
    plt.show()
    
    return model, scaler, num_features, cat_features

def predict_test_data(model, scaler, num_features, cat_features):
    test_df = pd.read_csv('test_data.csv', parse_dates=['Date'])
    test_df = create_features(test_df, is_training=False)
    test_df[num_features] = scaler.transform(test_df[num_features])
    test_df = test_df.fillna(0)
    
    X_test = test_df[num_features + cat_features]
    test_df['prediksi'] = model.predict(X_test)
    
    test_df['ID (kota)'] = (
        test_df['city'] + '_' + 
        test_df['Date'].dt.year.astype(str) + '_' + 
        test_df['Date'].dt.month.astype(str).str.zfill(2) + '_' + 
        test_df['Date'].dt.day.astype(str).str.zfill(2)
    )
    
    final_output = test_df[['ID (kota)', 'Date']].copy()
    final_output['tahun'] = test_df['Date'].dt.year
    final_output['bulan'] = test_df['Date'].dt.month
    final_output['hari'] = test_df['Date'].dt.day
    final_output['prediksi'] = test_df['prediksi']
    
    final_output = final_output[['ID (kota)', 'tahun', 'bulan', 'hari', 'prediksi']]
    final_output = final_output.drop_duplicates(subset=['ID (kota)'])
    final_output.to_csv('prediksi_curah_hujan_method3.csv', index=False, float_format='%.2f')
    
    return final_output

if __name__ == "__main__":
    model, scaler, num_features, cat_features = main()
    predictions = predict_test_data(model, scaler, num_features, cat_features)
    print(predictions.head())
