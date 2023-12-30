from fastapi import FastAPI
import pandas as pd

app = FastAPI(debug=True)

df = pd.read_csv("Csv/Df_Merged.csv.gz", compression='gzip', encoding='utf-8')

def get_genre_data(genre: str):
    genre = genre.capitalize()
    if genre not in df.columns:
        return None
    return df[df[genre] == 1]

@app.get("/")
def Index():
    return {"Poner Indice"}

@app.get("/PlayTimeGenre")
def PlayTimeGenre(genre: str) -> dict:
    genre_df = get_genre_data(genre)
    if genre_df is None:
        return {"Error": f"Género {genre} no encontrado en el dataset."}
    year_playtime_df = genre_df.groupby('posted_year')['playtime_forever'].sum().reset_index()
    max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'posted_year']
    return {"Género": genre, f"Año de lanzamiento con más horas jugadas para Género {genre} :": int(max_playtime_year)}

@app.get("/UserForGenre")
def UserForGenre(genre: str) -> dict:
    genre_df = get_genre_data(genre)
    if genre_df is None:
        return {"Error": f"Género {genre} no encontrado en el dataset."}
    user_playtime_df = genre_df.groupby('user_id')['playtime_forever'].sum().reset_index()
    max_playtime_user = user_playtime_df.loc[user_playtime_df['playtime_forever'].idxmax(), 'user_id']
    return {"Género": genre, f"Usuario con más horas jugadas para Género {genre} :": max_playtime_user}

def get_year_data(year: int):
    if year < 2010 or year > 2015:
        return None
    return df[df['posted_year'] == year]

@app.get("/UsersRecommend")
def UsersRecommend(year: int):
    year_df = get_year_data(year)
    if year_df is None or year_df.empty:
        return {"Año": year, "Juegos más recomendados para el año dado": "No hay datos para el año dado"}
    year_df = year_df.groupby('item_name')['sentiment_score'].sum().reset_index()
    year_df = year_df.sort_values(by='sentiment_score', ascending=False)
    year_df = year_df.head(3)
    return {"Año": year, "Juegos más recomendados para el año dado": year_df['item_name'].tolist()}

@app.get("/UsersWorstDeveloper")
def UsersWorstDeveloper(year: int):
    year_df = get_year_data(year)
    if year_df is None or year_df.empty:
        return {"Año": year, "Juegos menos recomendados para el año dado": "No hay datos para el año dado"}
    year_df = year_df.groupby('item_name')['sentiment_score'].sum().reset_index()
    year_df = year_df.sort_values(by='sentiment_score', ascending=True)
    year_df = year_df.head(3)
    return {"Año": year, "Juegos menos recomendados para el año dado": year_df['item_name'].tolist()}

@app.get("/sentiment_analysis")
def sentiment_analysis(developer: str):
    df['developer'] = df['developer'].str.capitalize()
    developer = developer.capitalize()
    if developer not in df['developer'].values:
        return {developer: ["No está registrado"]}
    developer_df = df[df['developer'] == developer]
    negative = len(developer_df[developer_df['sentiment_score'] == 0])
    neutral = len(developer_df[developer_df['sentiment_score'] == 1])
    positive = len(developer_df[developer_df['sentiment_score'] == 2])
    return {developer: ["Negative:", negative, "Neutral:", neutral, "Positive:", positive]}
