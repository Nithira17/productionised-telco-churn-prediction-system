import joblib

class ModelTrainer():
    def train(self, model, X_train, Y_train):
        model.fit(X_train, Y_train)
        train_score = model.score(X_train, Y_train)
        return model, train_score
    
    def save_model(self, model, filepath):
        joblib.dump(model, filepath)

    def load_model(self, filepath):
        return joblib.load(filepath)
    