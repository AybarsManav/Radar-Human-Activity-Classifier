import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

class SVM_Pipeline:
    
    def __init__(self):
        self.svm = SVC()
        self.scaler = StandardScaler()
        self.leave_one_group_out = LeaveOneGroupOut()
        self.param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf'],
            'svm__gamma': ['scale', 'auto']
        }


    def init_and_split_data(self, dataframe, split_ratio=0.2, random_state=42):
        """
        Initialize and split the data into training and testing sets.
        """
        X = dataframe.drop(columns=['subject_id', 'activity'])
        y = dataframe['activity']
        groups = dataframe['subject_id']
    
        # Split the data
        self.X_train, self.X_test, \
        self.y_train, self.y_test, \
        self.groups_train, self.groups_test = train_test_split(
            X, y, groups, test_size=split_ratio,
            random_state=random_state, stratify=y
        )
        
        return self.X_train, self.X_test, \
        self.y_train, self.y_test, \
        self.groups_train, self.groups_test
    
    def set_param_grid(self, param_grid):
        """
        Set the parameter grid for GridSearchCV.
        """
        self.param_grid = param_grid

    def set_scaler(self, scaler):
        """
        Set the scaler for the pipeline.
        """
        self.standard_scaler = scaler

    def run_grid_search(self, X_train, y_train, groups, scoring ='accuracy'):
        """
        Run GridSearchCV to find the best hyperparameters.
        """
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('svm', self.svm)
        ])
        
        self.grid_search = GridSearchCV(
            pipeline, self.param_grid, cv=self.leave_one_group_out.split(X_train, y_train, groups=groups),
            scoring=scoring, n_jobs=-1
        )
        
        self.grid_search.fit(X_train, y_train, groups=groups)
        
        return self.grid_search
    
    def show_grid_search_results(self):
        """
        Show the results of the grid search.
        """
        if hasattr(self, 'grid_search'):
            results = pd.DataFrame(self.grid_search.cv_results_)
            # print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
            return results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
        else:
            raise ValueError("Grid search has not been run yet.")
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        """
        if hasattr(self, 'grid_search'):
            y_pred = self.grid_search.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            print("Best params:", self.grid_search.best_params_)
            # print("Classification Report:")
            # print(report)
            return report
        else:
            raise ValueError("Grid search has not been run yet.")
        

    def plot_learning_curve(self, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42):
        """
        Plot the learning curve (training and validation accuracy) for the best model from grid search.
        """
        if not hasattr(self, 'grid_search'):
            raise ValueError("Grid search has not been run yet.")

        X, y = X_train, y_train

        best_pipeline = self.grid_search.best_estimator_

        train_sizes_arr, train_scores, val_scores = learning_curve(
            best_pipeline, X, y, cv=5, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1, random_state=random_state
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        plt.figure()
        plt.title("Learning Curve (Best Grid Search Model)")
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.grid()

        plt.fill_between(train_sizes_arr, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes_arr, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes_arr, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes_arr, val_scores_mean, 'o-', color="g", label="Validation score")

        plt.legend(loc="best")
        plt.show()
        
    
    