import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv(
    "data/train.csv",
    sep=",",
)
data.columns = ['id', 'выручка от продаж(млн)', 'объём продаж',
                'всего детей', 'детей дома', 'машин дома', 'вес брутто',
                'перерабатываемая упаковка(да/нет)', 'низкое содержание жира(да/нет)',
                'единиц в упаковке', 'площать магазина', 'кафе(да/нет)',
                'магазин видеотехники(да/нет)', 'салат бар(да/нет)',
                'готовая еда(да/нет)', 'цветочный магазин(да/нет)', 'стоимость']

print("Количество уникальных значений: \n")
print(data.nunique())

# так как в данных столбцах мало уникальных значений,
# их тип -- категориальные
data['объём продаж'] = data['объём продаж'].astype('category')
data['всего детей'] = data['всего детей'].astype('category')
data['детей дома'] = data['детей дома'].astype('category')
data['машин дома'] = data['машин дома'].astype('category')
data['перерабатываемая упаковка(да/нет)'] = data['перерабатываемая упаковка(да/нет)'].astype('category')
data['низкое содержание жира(да/нет)'] = data['низкое содержание жира(да/нет)'].astype('category')
data['единиц в упаковке'] = data['единиц в упаковке'].astype('category')
data['площать магазина'] = data['площать магазина'].astype('category')
data['кафе(да/нет)'] = data['кафе(да/нет)'].astype('category')
data['магазин видеотехники(да/нет)'] = data['магазин видеотехники(да/нет)'].astype('category')
data['салат бар(да/нет)'] = data['салат бар(да/нет)'].astype('category')
data['готовая еда(да/нет)'] = data['готовая еда(да/нет)'].astype('category')
data['цветочный магазин(да/нет)'] = data['цветочный магазин(да/нет)'].astype('category')

# Подсчет общего количества пропусков во всем датасете
total_missing_values = data.isna().sum().sum()
print("\nОбщее количество пропусков в датасете:", total_missing_values)

# Закодируем категории
categorical_features = data.select_dtypes(include=['category']).columns
for col in categorical_features:
    data[col] = LabelEncoder().fit_transform(data[col])

# Нормализуем данные
scaler = StandardScaler()
numerical_features = data.select_dtypes(include=[np.number]).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

with open("train_prepared.csv", "w") as file:
    file.write(data.to_csv())
