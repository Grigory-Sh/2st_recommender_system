import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List
import os
from typing import List
from fastapi import Depends, FastAPI
from schema import PostGet
from datetime import datetime

app = FastAPI()

def get_model_path(path: str) -> str:
    # проверяем где выполняется код в лмс, или локально. Немного магии
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    # LOAD MODEL HERE PLS :)
    filename = './cat_model.pkl'
    model_path = get_model_path(filename)
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model


# cкачиваем все данные из user_data
user_data = pd.read_sql(
    "SELECT * FROM public.user_data",
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@""postgres.lab.karpov.courses:6432/startml"
)

# скачиваем среднее по target по каждому user_id из feed_data
user_stat = pd.read_sql(
    "SELECT user_id, AVG(target) AS user_stat FROM public.feed_data GROUP BY user_id",
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@""postgres.lab.karpov.courses:6432/startml"
)

# cкачиваем все данные из post_text
post_text = pd.read_sql(
    "SELECT * FROM public.post_text_df",
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@""postgres.lab.karpov.courses:6432/startml"
)

# скачиваем среднее по target по каждому post_id из feed_data
post_stat = pd.read_sql(
    "SELECT post_id, AVG(target) AS post_stat FROM public.feed_data GROUP BY post_id",
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@""postgres.lab.karpov.courses:6432/startml"
)

# формируем DataFrame для предсказания
note = post_text.copy()
# преобразуем колонку text при помощи TfidfVectorizer в числа
vectorizer = TfidfVectorizer()
F = vectorizer.fit_transform(note['text'])
note['text_mean'] = F.toarray().mean(axis=1)
note['text_max'] = F.toarray().max(axis=1)
note['text_sum'] = F.toarray().sum(axis=1)
note = note.drop('text', axis=1)
# добавляем статистику по постам
note = pd.merge(note,
                post_stat,
                on='post_id',
                how='left')
# добавляем к таблицу user_data статистику по юзерам
users = pd.merge(user_data,
                 user_stat,
                 on='user_id',
                 how='left')

# функция добавления к Dataframe данных пользователя и всремени
def add_user_time(note, pers, time):
    # добавляем колонки с данными пользователя
    note['gender'] = pers[0]
    note['age'] = pers[1]
    note['country'] = pers[2]
    note['city'] = pers[3]
    note['exp_group'] = pers[4]
    note['os'] = pers[5]
    note['source'] = pers[6]
    note['user_stat'] = pers[7]
    # добавляем колонки с данными времени
    note['year'] = time.year
    note['month'] = time.month
    note['day'] = time.day
    note['hour'] = time.hour
    note['minute'] = time.minute
    note['second'] = time.second

    return note

# загружаем модель
model = load_models()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int,
                      time: datetime,
                      limit: int = 10) -> List[PostGet]:

    # получаем данные пользователя по id из users
    pers = users[users['user_id']==id].drop('user_id', axis=1).iloc[0]
    # создаём DataFrame для предсказания
    pred = add_user_time(note, pers, time).drop('post_id', axis=1)
    # создаём DataFrame с предсказаниями
    predict = pd.DataFrame(post_text['post_id'])
    # делаем предсказания на основе pred
    predict['predict'] = model.predict_proba(pred)[:,1]
    # сортируем посты по предсказанной вероятности от большей к меньшей
    predict.sort_values('predict', ascending=False, inplace=True, ignore_index=True)
    # отбираем limit постов
    posts = predict.head(limit)['post_id'].values
    # формируем DataFrame с предсказанными постами
    posts = post_text[post_text['post_id'].isin(posts)]

    return [PostGet(**{"id": i[0], "text": i[1], "topic": i[2]})
            for i in posts.itertuples(index=False)]
