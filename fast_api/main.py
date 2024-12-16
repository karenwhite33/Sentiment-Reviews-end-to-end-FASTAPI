# main.py fastapi


from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import collections

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}


# Cargar el dataframe de reseñas de productos
df = pd.read_pickle("./to_deeplearning.pkl")


# Vectorizar las reseñas de productos con TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df['fullProcessedReview'])

# Función para obtener productos similares basados en la reseña
def get_similar_products(query: str, top_n: int = 5):
    query_vector = vectorizer.transform([query])  # Vectorizar la consulta del usuario
    similarities = cosine_similarity(query_vector, X)  # Calcular similitud de coseno
    
    # Obtener los índices de las reseñas más similares
    similar_indices = similarities.argsort()[0][-top_n:][::-1]
    similar_reviews = df.iloc[similar_indices]  # Obtener las reseñas recomendadas
    
    return similar_reviews[['fullProcessedReview', 'sentiment_label']]

# Definir modelo para la consulta del usuario
class ProductQuery(BaseModel):
    review: str
    min_rating: Optional[float] = 4.0  # Puntuación mínima deseada para el producto

# 1. Módulo de Recomendación de Productos basado en Reseñas (Sin HuggingFace)
@app.post('/recommend_products')
def recommend_products(query: ProductQuery):
    similar_reviews = get_similar_products(query.review)
    return {'Recommended Reviews': similar_reviews.to_dict(orient='records')}

# 2. Módulo de Análisis de Sentimientos usando HuggingFace
@app.get('/sentiment')
def sentiment_classifier(query: str): 
    sentiment_pipeline = pipeline('sentiment-analysis')
    result = sentiment_pipeline(query)
    return {'Sentiment': result[0]['label'], 'Confidence': result[0]['score']}

# 3. Módulo de Resumen de Reseña usando HuggingFace
@app.get('/summarize')
def summarize_text(query: str): 
    summarization_pipeline = pipeline('summarization')
    summary = summarization_pipeline(query)
    return {'Summary': summary[0]['summary_text']}

# 4. Módulo de Estadísticas Básicas del Dataset (Sin HuggingFace)
@app.get('/basic_stats')
def basic_stats(): 
    sentiment_counts = df['sentiment_label'].value_counts()
    return {'Sentiment Distribution': sentiment_counts.to_dict()}

# 5. Módulo de Contar Palabras más Comunes en Reseñas (Sin HuggingFace)
@app.get('/count_words')
def count_words(): 
    text = ' '.join(df['fullProcessedReview'])
    words = text.split()
    word_counts = collections.Counter(words)
    common_words = word_counts.most_common(10)
    return {'Most Common Words': common_words}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)

