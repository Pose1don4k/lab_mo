import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from urllib.request import urlretrieve


pd.set_option("display.precision", 2)


if not os.path.exists('titanic_train.csv'):
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    urlretrieve(url, 'titanic_train.csv')


data = pd.read_csv('titanic_train.csv', index_col='PassengerId')

# 1. Сколько мужчин / женщин было на борту?
gender_count = data['Sex'].value_counts()

# 2. Определите распределение функции Pclass. Для мужчин и женщин отдельно. 
# Сколько людей из второго класса было на борту?
pclass_gender = pd.crosstab(data['Pclass'], data['Sex'])
second_class_male = pclass_gender.loc[2, 'male']

# 3. Каковы медиана и стандартное отклонение Fare?
fare_median = data['Fare'].median()
fare_std = data['Fare'].std()

# 4. Правда ли, что средний возраст выживших людей выше, чем у пассажиров, которые в конечном итоге умерли?
age_by_survival = data.groupby('Survived')['Age'].mean()
survived_older = age_by_survival[1] > age_by_survival[0]

# 5. Правда ли, что пассажиры моложе 30 лет выжили чаще, чем те, кому больше 60 лет?
young_passengers = data[data['Age'] < 30]
old_passengers = data[data['Age'] > 60]
young_survival_rate = young_passengers['Survived'].mean() * 100
old_survival_rate = old_passengers['Survived'].mean() * 100

# 6. Правда ли, что женщины выживали чаще мужчин?
survival_by_gender = data.groupby('Sex')['Survived'].mean() * 100

# 7. Какое имя наиболее популярно среди пассажиров мужского пола?
def extract_first_name(full_name):
    if pd.isna(full_name):
        return "Unknown"
    if 'Mr.' in full_name:
        name_part = full_name.split('Mr. ')[1]
    elif 'Master.' in full_name:
        name_part = full_name.split('Master. ')[1]
    elif 'Dr.' in full_name:
        name_part = full_name.split('Dr. ')[1]
    elif 'Rev.' in full_name:
        name_part = full_name.split('Rev. ')[1]
    else:
        if ', ' in full_name:
            name_part = full_name.split(', ')[1]
        else:
            name_part = full_name
    first_name = name_part.split(' ')[0].strip('()"\'')
    return first_name

male_data = data[data['Sex'] == 'male'].copy()
male_data['FirstName'] = male_data['Name'].apply(extract_first_name)
name_popularity = male_data['FirstName'].value_counts()
most_popular_name = name_popularity.index[0]

# 8. Как средний возраст мужчин / женщин зависит от Pclass?
age_by_class_gender = data.groupby(['Pclass', 'Sex'])['Age'].mean()
statement_a = age_by_class_gender.loc[(1, 'male')] > 40
statement_b = age_by_class_gender.loc[(1, 'female')] > 40

men_older_in_all_classes = True
for pclass in [1, 2, 3]:
    men_age = age_by_class_gender.loc[(pclass, 'male')]
    women_age = age_by_class_gender.loc[(pclass, 'female')]
    if men_age <= women_age:
        men_older_in_all_classes = False
        break
statement_c = men_older_in_all_classes

age_decreases_by_class = True
for gender in ['male', 'female']:
    class1_age = age_by_class_gender.loc[(1, gender)]
    class2_age = age_by_class_gender.loc[(2, gender)]
    class3_age = age_by_class_gender.loc[(3, gender)]
    if not (class1_age > class2_age > class3_age):
        age_decreases_by_class = False
        break
statement_d = age_decreases_by_class


print("ОТВЕТЫ:")
print(f"1. {gender_count['male']} мужчин и {gender_count['female']} женщин")
print(f"2. {second_class_male}")
print(f"3. медиана {fare_median:.2f}, стандартное отклонение {fare_std:.2f}")
print(f"4. {'Да' if survived_older else 'Нет'}")
print(f"5. {young_survival_rate:.1f}% среди молодежи и {old_survival_rate:.1f}% среди пожилых")
print(f"6. {survival_by_gender['male']:.1f}% среди мужчин и {survival_by_gender['female']:.1f}% среди женщин")
print(f"7. {most_popular_name}")
print(f"8. Правильные утверждения: а) {statement_a}, б) {statement_b}, в) {statement_c}, г) {statement_d}")
