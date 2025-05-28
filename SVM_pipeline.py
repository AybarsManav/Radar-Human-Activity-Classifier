import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

class SVM_Pipeline:
    
    def __init__(self):
        self.svm = SVC()
        self.leave_one_group_out = LeaveOneGroupOut()


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

    def build_pipeline(self, scaler=None):
        steps = []
        if scaler is not None:
            steps.append(('scaler', scaler))
        else:
            steps.append(('scaler', 'passthrough'))
        steps.append(('sfs', SFS(
            SVC(),
            k_features=1,
            forward=True,
            floating=False,
            scoring='accuracy',
            cv=0,  # CV handled outside in grid search function
            n_jobs=-1,
            verbose=0
        )))
        steps.append(('svm', SVC()))
        return Pipeline(steps)

    def run_grid_search(self, X_train, y_train, groups, scoring ='accuracy'):
        """
        Run GridSearchCV to find the best hyperparameters.
        """
        pipeline = self.build_pipeline()
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
            print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
            return results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
        else:
            raise ValueError("Grid search has not been run yet.")
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test set and show the confusion matrix.
        """
        if hasattr(self, 'grid_search'):
            y_pred = self.grid_search.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            print("Best params:", self.grid_search.best_params_)
            # print("Classification Report:")
            # print(report)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
            plt.show()
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
        

if __name__ == "__main__": 
    random_state = 42

    # Generate random data
    np.random.seed(random_state)
    n_samples = 120
    n_features = 4

    feature_columns = [f'feature_{i}' for i in range(n_features)]
    # Generate six clusters centered at different locations
    centers = [np.full(n_features, fill_value=v) for v in [-4, -2, 0, 2, 4, 2]]
    samples_per_class = n_samples // 6

    X_list = []
    activities = []
    for i, center in enumerate(centers):
        X_class = np.random.randn(samples_per_class, n_features) + np.array(center)
        X_list.append(X_class)
        activities.extend([i] * samples_per_class)

    X_random = np.vstack(X_list)
    activities = np.array(activities)

    # Assign subject_ids randomly
    subject_ids = np.random.randint(0, 6, size=n_samples)
    df = pd.DataFrame(X_random, columns=feature_columns)
    df['subject_id'] = subject_ids
    df['activity'] = activities

    # 1. Load data
    # df = pd.read_csv('your_data.csv')
    X = df[feature_columns]
    y = df['activity']
    groups = df['subject_id']

    # Set up SVM pipeline by choosing hyperparameter grid and scaler.
    # The hyperparameter selection uses leave-one-group-out strategy to decide on hyperparameters.
    svm_pipeline = SVM_Pipeline()
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'scaler' : [StandardScaler(), MinMaxScaler(), None],
        'sfs__k_features': [n_features],  # Select all features but we will use this for report
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']
    }
    svm_pipeline.set_param_grid(param_grid)

    # Initialize and split the dataset.
    X_train, X_test, y_train, y_test, groups_train, groups_test \
    = svm_pipeline.init_and_split_data(df, split_ratio=0.2, random_state=random_state)

    # Conduct the gridsearch to find the best model.
    svm_pipeline.run_grid_search(X_train, y_train, groups_train)
    svm_pipeline.show_grid_search_results()

    # Evaluate the best model on the test set
    svm_pipeline.evaluate_model(X_test, y_test)

    # Plot learning curve to check overfitting
    svm_pipeline.plot_learning_curve(X_train, y_train)


    """ Predict on the mesh grid to visualize the decision boundary.
        For now only works for two features """
    
    """ 
    # Get best estimator from grid search
    best_svm = svm_pipeline.grid_search.best_estimator_

    # Create a mesh to plot decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X_train['feature_0'].min() - 1, X_train['feature_0'].max() + 1
    y_min, y_max = X_train['feature_1'].min() - 1, X_train['feature_1'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)

    # Plot the training points
    scatter = plt.scatter(X_test['feature_0'], X_test['feature_1'], c=y_test, cmap=plt.cm.Set1, edgecolor='k', s=40)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.title('SVM Decision Boundary and Data Points')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()
    """