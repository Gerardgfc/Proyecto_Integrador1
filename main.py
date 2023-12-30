from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(debug=True)

df_games_item = pd.read_csv('Csv/steam_games_items.csv.gz', compression='gzip', encoding='utf-8')
df_games_review = pd.read_csv('Csv/steam_games_reviews.csv.gz', compression='gzip', encoding='utf-8')


@app.get("/")
def Index():
    return {"Poner Indice"}

@app.get("/PlayTimeGenre")
def PlayTimeGenre(genre: str) -> dict:
    """ 
    Obtiene el año con más horas jugadas para un género específico. 

    Genre: Genero a obtener el número de años de lanzamiento con más horas

    Returns:
        Año con más horas jugadas para un género específico teniendo en cuenta los generos disponibles.
    """
    genre = genre.capitalize()
    if genre not in df_games_item.columns:
        return {"Error": f"Género {genre} no encontrado en el dataset."}
    else:
        genre_df = df_games_item[df_games_item[genre] == 1]
        year_playtime_df = genre_df.groupby('year')['playtime_forever'].sum().reset_index()
        max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'year']
        return {"Género": genre, f"Año de lanzamiento con más horas jugadas para Género {genre} :": int (max_playtime_year)} 

@app.get("/UserForGenre")
def UserForGenre(genre: str) -> dict:
    """Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
    
    Genre: Genero a obtener el número de usuarios que han jugado

    Returns:
        Usuario que ha jugado más horas para un género específico teniendo en cuenta los generos disponibles.
    """
    genre = genre.capitalize()
    if genre not in df_games_item.columns:
        return {"Error": f"Género {genre} no encontrado en el dataset."}
    else:
        genre_df = df_games_item[df_games_item[genre] == 1]
        user_playtime_df = genre_df.groupby('user_id')['playtime_forever'].sum().reset_index()
        max_playtime_user = user_playtime_df.loc[user_playtime_df['playtime_forever'].idxmax(), 'user_id']
        return {"Género": genre, f"Usuario con más horas jugadas para Género {genre} :": max_playtime_user}


@app.get("/UsersRecommend")
def UsersRecommend( year : int ):
    """Devuelve el top 3 de juegos más recomendados por usuarios para el año dado.

    Year: Año a obtener los juegos más recomendados

    Returns:
        Lista con los juegos más recomendados para el año dadoy teniendo en cuenta los años que disponemos.
    """
    if year < 2010 or year > 2015:
        return {"Año": year, "Juegos más recomendados para el año dado": "No hay datos para el año dado"}
    else:
        year_df = df_games_review[df_games_review['posted_year'] == year]
        year_df = year_df.groupby('title')['sentiment_score'].sum().reset_index()
        year_df = year_df.sort_values(by='sentiment_score', ascending=False)
        year_df = year_df.head(3)
        return {"Año": year, "Juegos más recomendados para el año dado": year_df['title'].tolist()}
    
@app.get("/UsersWorstDeveloper")
def UsersWorstDeveloper( year : int ):
    """Devuelve el top 3 de juegos menos recomendados por usuarios para el año dado.

    Year: Año a obtener los juegos menos recomendados

    Returns:
        Lista con los juegos menos recomendados para el año dado y teniendo en cuenta los años que disponemos.
    """
    if year < 2010 or year > 2015:
        return {"Año": year, "Juegos menos recomendados para el año dado": "No hay datos para el año dado"}
    else:
        year_df = df_games_review[df_games_review['posted_year'] == year]
        year_df = year_df.groupby('title')['sentiment_score'].sum().reset_index()
        year_df = year_df.sort_values(by='sentiment_score', ascending=True)
        year_df = year_df.head(3)
        return {"Año": year, "Juegos menos recomendados para el año dado": year_df['title'].tolist()}

@app.get("/sentiment_analysis")
def sentiment_analysis( developer: str ):
    """Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor.
    
    Developer: Empresa desarrolladora
    
    Returns:
        Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}, en caso de que no se encuentre ningúna desarrolladora dara como rsultado error.
    """
    df_games_review['developer'] = df_games_review['developer'].str.capitalize()
    developer = developer.capitalize()
    if developer not in df_games_review['developer'].values:
        return {developer: ["No está registrado"]}
    else:
        developer_df = df_games_review[df_games_review['developer'] == developer]
    
        negative = len(developer_df[developer_df['sentiment_score'] == 0])
        neutral = len(developer_df[developer_df['sentiment_score'] == 1])
        positive = len(developer_df[developer_df['sentiment_score'] == 2])

        return {developer: ["Negative:",negative, "Neutral:", neutral, "Positive:", positive]}
    