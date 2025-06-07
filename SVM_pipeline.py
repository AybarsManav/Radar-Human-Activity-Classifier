import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA
import joblib

class SVM_Pipeline:
    
    def __init__(self, n_splits=5, random_state=42):
        self.svm = SVC(
            cache_size=1000, # Suggested in https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
            )
        self.cv = KFold(n_splits= n_splits, shuffle=True, random_state=random_state)


    def init_and_split_data(self, dataframe, split_ratio=0.2, random_state=42):
        """
        Initialize and split the data into training and testing sets.
        """
        X = dataframe.drop(columns=['label'])
        y = dataframe['label']
    
        # Split the data
        self.X_train, self.X_test, \
        self.y_train, self.y_test, \
        = train_test_split(
            X, y, test_size=split_ratio,
            random_state=random_state, stratify=y)
        
        return self.X_train, self.X_test, \
        self.y_train, self.y_test
    
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
        self.best_model = None

    def save_best_model(self, filepath):
        """
        Save the best model from grid search to a file.
        """
        if hasattr(self, 'grid_search'):
            joblib.dump(self.grid_search, filepath)
        else:
            raise ValueError("Grid search has not been run yet.")
    
    def load_best_model(self, filepath):
        """
        Load the best model from a file.
        """
        self.grid_search = joblib.load(filepath)


    def build_pipeline(self, scaler=None, scoring = 'accuracy'):
        steps = []
        if scaler is not None:
            steps.append(('scaler', scaler))
        else:
            steps.append(('scaler', 'passthrough'))
        steps.append(('svm', SVC()))
        return Pipeline(steps)

    def run_grid_search(self, X_train, y_train, scoring ='accuracy'):
        """
        Run GridSearchCV to find the best hyperparameters.
        """
        pipeline = self.build_pipeline()
        self.grid_search = GridSearchCV(
            pipeline, self.param_grid, cv=self.cv,
            scoring=scoring, n_jobs=-1
        )
        
        self.grid_search.fit(X_train, y_train) #,groups=groups) needed if we were to use leave-one-group-out strategy
        
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
        Evaluate the model on the test set and show the confusion matrix.
        """
        if hasattr(self, 'grid_search'):
            y_pred = self.grid_search.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            print("Best params:", self.grid_search.best_params_)
            # Show confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
            plt.show()
            # Visualize classification report
            report_df = pd.DataFrame(report).transpose()
            metrics = ['precision', 'recall', 'f1-score']
            # Plot per-class metrics
            ax = report_df.iloc[:-3, :-1][metrics].plot(kind='bar', figsize=(10, 5))

            # Add macro and weighted averages as horizontal lines
            for metric in metrics:
                macro_avg = report_df.loc['macro avg', metric]
                weighted_avg = report_df.loc['weighted avg', metric]
                ax.axhline(macro_avg, linestyle='--', color='blue', alpha=0.7, label=f'Macro avg {metric}')
                ax.axhline(weighted_avg, linestyle=':', color='red', alpha=0.7, label=f'Weighted avg {metric}')

            # Annotate each bar with support
            # supports = report_df.iloc[:-3]['support']
            # for idx, support in enumerate(supports):
            #     for bar in ax.containers:
            #         height = bar[idx].get_height()
            #         ax.annotate(f'n={int(support)}', 
            #                     (bar[idx].get_x() + bar[idx].get_width() / 2, height),
            #                     ha='center', va='bottom', fontsize=8, rotation=90)

            plt.title("Precision, Recall, F1-score per Class (with Averages)", fontsize=22)
            plt.ylabel("Score", fontsize=20)
            plt.xlabel("Classes", fontsize=20)
            plt.ylim(0, 1)
            plt.xticks(rotation=90, fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=12)
            plt.tight_layout()
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
        
    def run_sfs_with_best_estimator(self, X, y, scoring='f1_macro', cv=5):
        """
        Run Sequential Forward Selection (SFS) using the best estimator (scaler and SVM) found by grid search.
        Returns the fitted SFS object.
        """
        if not hasattr(self, 'grid_search'):
            raise ValueError("Grid search must be run before SFS feature selection.")
        
        # Get the best pipeline (scaler + SVM)
        best_pipeline = self.grid_search.best_estimator_
        scaler = best_pipeline.named_steps.get('scaler', None)
        svm = best_pipeline.named_steps['svm']

        # Scale data if needed (this is required because we cannot pass a pipeline directly to SFS)
        if scaler is not None and scaler != 'passthrough':
            scaler.fit(X)
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X

        # Run SFS (choose all of the features, we will use their ordering in report)
        sfs = SFS(svm, k_features=svm.n_features_in_, forward=True, floating=False, scoring=scoring, cv=cv, n_jobs=-1)
        sfs = sfs.fit(X_scaled, y)
        return sfs
    
    def plot_sfs_feature_importance(self, sfs, feature_names, metric='avg_score'):
        """
        Plot the effect of each feature added by SFS on the chosen metric (e.g., f1-macro).
        Annotates each step with the feature name added.
        """
        sfs_metrics = sfs.get_metric_dict()
        sfs_metrics_df = pd.DataFrame.from_dict(sfs_metrics, orient='index')
        n_features = sfs_metrics_df['feature_idx'].apply(lambda x: len(x))
        scores = sfs_metrics_df[metric]
        # Find which feature was added at each step
        added_features = []
        prev = set()
        for idx in sfs_metrics_df['feature_idx']:
            current = set(idx)
            new_feat = list(current - prev)
            if new_feat:
                added_features.append(feature_names[new_feat[0]])
            else:
                added_features.append("")
            prev = current

        plt.figure(figsize=(10, 6))
        plt.plot(n_features, scores, marker='o')
        for x, y, feat in zip(n_features, scores, added_features):
            plt.annotate(feat, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, rotation=45)
        plt.xlabel('Number of Features Selected')
        plt.ylabel(metric)
        plt.title(f'SFS: {metric} vs. Number of Features')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_decision_boundaries_using_PCA(self, X_train, X_test, y_train, y_test):
        if not hasattr(self, 'grid_search'):
            raise ValueError("Grid search must be run before SFS feature selection.")
        
        scaler = self.grid_search.best_estimator_.named_steps['scaler']
        svm = self.grid_search.best_estimator_.named_steps['svm']

        # Scale the data if needed
        if scaler is not None and scaler != 'passthrough':
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Fit PCA on scaled training data
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Create a mesh grid in PCA space
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        h = .02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Inverse transform mesh grid points back to original feature space
        grid_points_original = pca.inverse_transform(grid_points)

        # Predict using the trained SVM
        Z = svm.predict(grid_points_original)
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
        scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.Set1, edgecolor='k', s=40)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('SVM Decision Boundary in PCA Space')
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.show()
        