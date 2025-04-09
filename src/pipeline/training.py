import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import json

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Klasse til at træne og evaluere XGBoost model."""
    
    def __init__(self):
        """Initialiserer ModelTrainer med standard parametre."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.features_dir = self.project_root / "data" / "features"
        self.models_dir = self.project_root / "models"
        
        # Opret models mappe hvis den ikke findes
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model parametre
        self.model_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'scale_pos_weight': 1,
            'random_state': 42
        }
        
        # Cross-validation parametre
        self.n_splits = 5
    
    def load_data(self) -> pd.DataFrame:
        """
        Indlæser feature data til træning.
        
        Returns:
            DataFrame med feature data
        """
        try:
            input_file = self.features_dir / "bitcoin_usd_365d_features.csv"
            df = pd.read_csv(input_file)
            logger.info(f"Indlæste feature data fra {input_file}")
            return df
        except Exception as e:
            logger.error(f"Fejl ved indlæsning af feature data: {str(e)}")
            raise
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Forbereder data til træning.
        
        Args:
            df: DataFrame med feature data
            
        Returns:
            Tuple med (X_scaled, y, feature_columns, scaler)
        """
        # Separer features og target
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'target']]
        X = df[feature_columns]
        y = df['target'].values
        
        # Skaler features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Opret ny DataFrame med skalede værdier og originale kolonnenavne
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Tilføj feature navne til scaler
        scaler.feature_names_in_ = X.columns
        
        return X_scaled_df.values, y, feature_columns, scaler
    
    def train_model(self, X: np.ndarray, y: np.ndarray, feature_columns: list) -> tuple:
        """
        Træner XGBoost model med time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_columns: Liste med feature navne
            
        Returns:
            Tuple med (model, cv_scores, feature_importance)
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        model = xgb.XGBClassifier(**self.model_params)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        feature_importance = np.zeros(len(feature_columns))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_scores['precision'].append(precision_score(y_val, y_pred))
            cv_scores['recall'].append(recall_score(y_val, y_pred))
            cv_scores['f1'].append(f1_score(y_val, y_pred))
            cv_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            
            # Akkumuler feature importance
            feature_importance += model.feature_importances_
            
            logger.info(f"Fold {fold}/{self.n_splits} - Accuracy: {cv_scores['accuracy'][-1]:.4f}")
        
        # Gennemsnitlig feature importance
        feature_importance /= self.n_splits
        
        # Opret feature importance dictionary
        feature_importance_dict = dict(zip(feature_columns, feature_importance))
        
        # Sorter features efter importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Log top 10 vigtigste features
        logger.info("Top 10 vigtigste features:")
        for feature, importance in sorted_features[:10]:
            logger.info(f"{feature}: {importance:.4f}")
        
        # Træn endelig model på alle data
        model.fit(X, y)
        
        return model, cv_scores, feature_importance_dict
    
    def save_model(self, model: xgb.XGBClassifier, scaler: StandardScaler, 
                  cv_scores: dict, feature_importance: dict) -> bool:
        """
        Gemmer model, scaler og metrics.
        
        Args:
            model: Trænet XGBoost model
            scaler: Fittet StandardScaler
            cv_scores: Cross-validation scores
            feature_importance: Feature importance dictionary
            
        Returns:
            True hvis gemning lykkedes, False ellers
        """
        try:
            # Gem model
            model_path = self.models_dir / "xgboost_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Model gemt til {model_path}")
            
            # Gem scaler
            scaler_path = self.models_dir / "scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler gemt til {scaler_path}")
            
            # Gem feature navne
            feature_names_path = self.models_dir / "feature_names.joblib"
            joblib.dump(list(feature_importance.keys()), feature_names_path)
            logger.info(f"Feature navne gemt til {feature_names_path}")
            
            # Beregn og gem gennemsnitlige metrics
            metrics = {
                'accuracy': np.mean(cv_scores['accuracy']),
                'precision': np.mean(cv_scores['precision']),
                'recall': np.mean(cv_scores['recall']),
                'f1_score': np.mean(cv_scores['f1']),
                'roc_auc': np.mean(cv_scores['roc_auc']),
                'feature_importance': feature_importance
            }
            
            metrics_path = self.models_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics gemt til {metrics_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Fejl ved gemning af model artifacts: {str(e)}")
            return False
    
    def train(self) -> bool:
        """
        Kører hele træningsprocessen.
        
        Returns:
            True hvis træning lykkedes, False ellers
        """
        try:
            logger.info("Starter model træning...")
            
            # Indlæs data
            df = self.load_data()
            
            # Forbered data
            X, y, feature_columns, scaler = self.prepare_data(df)
            
            # Træn model
            model, cv_scores, feature_importance = self.train_model(X, y, feature_columns)
            
            # Gem model og metrics
            if self.save_model(model, scaler, cv_scores, feature_importance):
                logger.info("Model træning gennemført succesfuldt")
                return True
            else:
                logger.error("Model træning fejlede (fejl ved gemning)")
                return False
                
        except Exception as e:
            logger.error(f"Fejl under model træning: {str(e)}")
            raise

def main():
    """Hovedfunktion til at køre model træning."""
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
