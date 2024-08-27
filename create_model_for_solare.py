import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def load_data(csv_string: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(StringIO(csv_string))
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def filter_data(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    try:
        filtered_df = df.query(condition)
        return filtered_df
    except Exception as e:
        print(f"Error filtering data: {e}")
        return None

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        scaler = MinMaxScaler()
        normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return normalized_df
    except Exception as e:
        print(f"Error normalizing data: {e}")
        return None

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Extract additional features from the data
        df['mag_strength_avg'] = df['mag_strength'].rolling(window=3).mean()
        df['temp_avg'] = df['temp'].rolling(window=3).mean()
        return df
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model.

    Args:
        df (pd.DataFrame): The data to train the model on.

    Returns:
        RandomForestClassifier: The trained model.
    """
    try:
        X = df.drop(['flare'], axis=1)
        y = df['flare']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def evaluate_model(model: RandomForestClassifier, df: pd.DataFrame) -> dict:
    """
    Evaluate the performance of a RandomForestClassifier model.

    Args:
        model (RandomForestClassifier): The model to evaluate.
        df (pd.DataFrame): The data to evaluate the model on.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    try:
        X = df.drop(['flare'], axis=1)
        y = df['flare']
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)
        return {'accuracy': accuracy, 'report': report, 'matrix': matrix}
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def tune_hyperparameters(model: RandomForestClassifier, df: pd.DataFrame) -> dict:
    """
    Tune the hyperparameters of a RandomForestClassifier model.

    Args:
        model (RandomForestClassifier): The model to tune.
        df (pd.DataFrame): The data to tune the model on.

    Returns:
        dict: A dictionary containing the best hyperparameters.
    """
    try:
        param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 5, 10, 15]}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(df.drop(['flare'], axis=1), df['flare'])
        return grid_search.best_params_
    except Exception as e:
        print(f"Error tuning hyperparameters: {e}")
        return None

def visualize_results(evaluation_results: dict) -> None:
    """
    Visualize the evaluation results.

    Args:
        evaluation_results (dict): A dictionary containing the evaluation metrics.
    """
    try:
        print("Model Evaluation Results:")
        print(f"Accuracy: {evaluation_results['accuracy']}")
        print(f"Classification Report:\n{evaluation_results['report']}")
        print(f"Confusion Matrix:\n{evaluation_results['matrix']}")
        
        # Visualize the confusion matrix
        plt.imshow(evaluation_results['matrix'], interpolation='nearest')
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(f"Error visualizing results: {e}")

def main():
    csv_string = '''time,mag_strength,temp,flare
2022-01-01,100.0,5000.0,0
2022-01-02,120.0,5500.0,0
2022-01-03,150.0,6000.0,1
2022-01-04,180.0,6500.0,1
2022-01-05,200.0,7000.0,0'''

    df = load_data(csv_string)
    if df is None:
        return

    filtered_df = filter_data(df, 'mag_strength > 150')
    if filtered_df is None:
        return

    normalized_df = normalize_data(filtered_df)
    if normalized_df is None:
        return

    feature_df = extract_features(normalized_df)
    if feature_df is None:
        return

    model = train_model(feature_df)
    if model is None:
        return

    evaluation_results = evaluate_model(model, feature_df)
    if evaluation_results is None:
        return

    visualize_results(evaluation_results)

if __name__ == "__main__":
    main()