import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate

class ModelTrainer():
    def train(self, model, X_train, Y_train):
        model.fit(X_train, Y_train)
        train_score = model.score(X_train, Y_train)
        return model, train_score
    
    def train_with_cv(self, model, X_train, Y_train, cv_folds=5, random_state=42):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        cv_results = cross_validate(model,
                                    X_train,
                                    Y_train,
                                    cv=cv,
                                    scoring=['accuracy', 'precision', 'recall', 'f1'],
                                    return_train_score=True)
        
        model.fit(X_train, Y_train)

        return model, cv_results
    
    def hyperparameter_search(self, model, X_train, Y_train, param_distributions,
                              n_iter=100, cv_folds=5, random_state=42):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        randomized_search = RandomizedSearchCV(estimator=model,
                                               param_distributions=param_distributions,
                                               n_iter=n_iter,
                                               cv=cv,
                                               scoring='f1',
                                               verbose=1,
                                               return_train_score=True,
                                               random_state=random_state)
        
        randomized_search.fit(X_train, Y_train)

        return randomized_search.best_estimator_, randomized_search.best_params_, randomized_search.best_score_
    
    def save_model(self, model, filepath):
        joblib.dump(model, filepath)

    def load_model(self, filepath):
        return joblib.load(filepath)
    