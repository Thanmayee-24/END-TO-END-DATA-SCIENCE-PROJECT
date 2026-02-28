import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI
import uvicorn
import json

def load_data():
    """Load Titanic dataset from URL."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

def preprocess_and_train(df):
    """Preprocess data and train a model."""
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Pclass', 'Sex']
    
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model trained and saved!")

def create_api():
    """Deploy model using FastAPI."""
    app = FastAPI()
    
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    @app.post("/predict")
    def predict(data: dict):
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    
    return app

def main():
    df = load_data()
    preprocess_and_train(df)
    app = create_api()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
if __name__ == "__main__":
    main()
