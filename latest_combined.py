import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import os
from scipy.spatial.distance import cdist
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import time
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Function to save plots
def save_plot(plot_name):
    save_dir='./plots_graphs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,plot_name))
    plt.close()


# K-mean Cluster
def cluster_kmean(data):
    # Filter the data to include only employees who left
    left_employees = data[data['left'] == 1]
    
    # Selecting relevant features for clustering
    X = left_employees[['satisfaction_level', 'last_evaluation']]
    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    k_values = range(1, 11)  # Testing from 1 to 10 clusters
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # WCSS is the sum of squared distances to closest centroid
    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters k=3')
    plt.ylabel('WCSS')
    plt.show()
    # Initialize the K-Means algorithm with the optimal number of clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    # Fit the model and predict the cluster for each data point
    left_employees['KMeans_Cluster'] = kmeans.fit_predict(X)
    # Ensure cluster labels are integers
    left_employees['KMeans_Cluster'] = left_employees['KMeans_Cluster'].astype(int)
    # Calculate the distance of each point to its cluster centroid
    distances = np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)
    # Define outlier threshold (95th percentile)
    threshold = np.percentile(distances, 95)
    outliers = distances > threshold
    # Add outliers column to left_employees and ensure it is an integer (0 or 1)
    left_employees['outliers'] = outliers.astype(int)
    # Exclude outliers from the data used for plotting
    X_no_outliers = X[~outliers]
    clusters_no_outliers = left_employees['KMeans_Cluster'][~outliers]
    # Assign descriptive labels based on centroid analysis
    cluster_labels = {
        0: 'Low Satisfaction & Low Evaluation',
        1: 'High Satisfaction & High Evaluation',
        2: 'Medium Satisfaction & Medium Evaluation'
    }
    # Map the labels to the clusters
    left_employees['Cluster_Label'] = left_employees['KMeans_Cluster'].map(cluster_labels)
    # Create a new DataFrame including all columns from `data` and the new columns
    new_df = data.copy()  # Start with the original `data`
    # Add clustering and outlier information
    new_df['KMeans_Cluster'] = np.nan  # Initialize with NaN values for employees who left
    new_df['Cluster_Label'] = np.nan
    new_df['outliers'] = np.nan
    new_df.loc[new_df['left'] == 1, 'KMeans_Cluster'] = left_employees['KMeans_Cluster']
    new_df.loc[new_df['left'] == 1, 'Cluster_Label'] = left_employees['Cluster_Label']
    new_df.loc[new_df['left'] == 1, 'outliers'] = left_employees['outliers']
    # Convert columns to integer type
    new_df['KMeans_Cluster'] = new_df['KMeans_Cluster'].astype('Int64')
    new_df['outliers'] = new_df['outliers'].astype('Int64')
    # Visualize the clusters with outliers
    plt.figure(figsize=(12, 8))
    scatter_with_outliers = plt.scatter(
        X['satisfaction_level'],
        X['last_evaluation'],
        c=left_employees['KMeans_Cluster'],
        cmap='viridis',
        label='Clusters'
    )
    # Plot centroids
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c='red',
        marker='o',
        label='Centroids'
    )
    # Highlight outliers
    plt.scatter(
        X.loc[outliers, 'satisfaction_level'],
        X.loc[outliers, 'last_evaluation'],
        edgecolor='black',
        facecolor='none',
        s=100,
        label='Outliers'
    )
    # Create a legend with the descriptive cluster labels, placing it outside the graph
    handles_with_outliers, _ = scatter_with_outliers.legend_elements()
    # Manually reorder the clusters for the legend
    ordered_cluster_indices = [1, 0, 2]  # Corresponding to the desired order
    legend_labels_with_outliers = [
        'High Satisfaction & High Evaluation',
        'Medium Satisfaction & Medium Evaluation',
        'Low Satisfaction & Medium Evaluation'
    ]
    # Reorder the handles according to the desired cluster order
    handles_with_outliers = [handles_with_outliers[i] for i in ordered_cluster_indices]
    # Create handles for Centroids and Outliers
    centroid_handle = plt.Line2D([0], [0], marker='o', color='w', label='Centroids',
                                markerfacecolor='red', markersize=10)
    outlier_handle = plt.Line2D([0], [0], marker='o', color='w', label='Outliers',
                                markerfacecolor='none', markeredgecolor='black', markersize=10)
    # Insert Centroids at the beginning
    handles_with_outliers.insert(0, centroid_handle)
    legend_labels_with_outliers.insert(0, 'Centroids')
    # Append Outliers at the end
    handles_with_outliers.append(outlier_handle)
    legend_labels_with_outliers.append('Outliers')
    plt.legend(handles_with_outliers, legend_labels_with_outliers, title="Clusters", loc='center left', bbox_to_anchor=(1, 0.5))
    # Format x,y-axis labels as percentages
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title('K-Means Clustering of Employees Who Left (Including Outliers)')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Last Evaluation (Percentage)')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 8))
    scatter_no_outliers = plt.scatter(
        X_no_outliers['satisfaction_level'],
        X_no_outliers['last_evaluation'],
        c=clusters_no_outliers,
        cmap='viridis',
        label='Clusters'
    )
    # Plot centroids
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c='red',
        marker='o',
        label='Centroids'
    )
    # Create a legend with the descriptive cluster labels, placing it outside the graph
    handles_no_outliers, _ = scatter_no_outliers.legend_elements()
    # Manually reorder the clusters for the legend
    ordered_cluster_indices = [1, 0, 2]  # Corresponding to the desired order
    legend_labels_no_outliers = [
        'High Satisfaction & High Evaluation',
        'Medium Satisfaction & Medium Evaluation',
        'Low Satisfaction & Medium Evaluation'
    ]
    # Reorder the handles according to the desired cluster order
    handles_no_outliers = [handles_no_outliers[i] for i in ordered_cluster_indices]
    # Create handles for Centroids
    centroid_handle = plt.Line2D([0], [0], marker='o', color='w', label='Centroids',
                                markerfacecolor='red', markersize=10)
    # Insert Centroids at the beginning
    handles_no_outliers.insert(0, centroid_handle)
    legend_labels_no_outliers.insert(0, 'Centroids')
    plt.legend(handles_no_outliers, legend_labels_no_outliers, title="Clusters", loc='center left', bbox_to_anchor=(1, 0.5))
    # Format x,y-axis labels as percentages
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title('K-Means Clustering of Employees Who Left (Excluding Outliers)')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Last Evaluation (Percentage)')
    plt.tight_layout()
    plt.show()
    # Return the DataFrame with clustering results and original columns
    return new_df


# Preprocess
def data_preprocess(path):
    df = pd.read_csv(path)
    # sample of the dataset
    # print(df.head())
    # checking the null or nan values
    # df.isna().sum()
    for col_name in df.columns:
        count_and_fill_null(df, col_name)
    # print(f"finding null values: \n{df.isna().sum()}")
    # df.fillna(0)
    df.rename(columns={'average_montly_hours':'hours', 'time_spend_company':'years', 'promotion_last_5years':'promoted', 'sales':'dept'}, inplace=True)
    # print(df.columns)
    
    """
    defining a column with name as satisfaction_category based on the satisfaction_level
    0 to 0.3 = low
    >0.3 to 0.7 = medium
    >0.7 to 1 = high
    """
    
    df['satisfaction_category'] = pd.cut(df['satisfaction_level'], bins=[0, 0.3, 0.7, 1], labels=['low', 'medium', 'high'])
    
    
    # workload is the thing to get the fraction value of hours with number_project
    
    df['workload'] = (df['hours'] / df['number_project']).round(4)
    # print(df[['workload']])
    
    
    # encoding on categorical columns making their values as numerical
    new_data = df.copy()
    
    new_data['salary'] = new_data['salary'].map({'low': 1, 'medium': 2, 'high': 3})
    new_data['satisfaction_category'] = new_data['satisfaction_category'].map({'low': 1, 'medium': 2, 'high': 3})
    
    # checcking any outliers in the entire dataset
    dt = new_data[['satisfaction_level', 'last_evaluation', 'number_project', 'hours', 'years', 'workload']]
    
    # choosing the IQR method as the normalization graph is mostly skewed

    Q1 = dt.quantile(0.25)
    Q3 = dt.quantile(0.75)
    IQR = Q3 - Q1
    
    IQR_Outliers= ((dt < (Q1 - 1.5 * IQR)) | (dt > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Count of outliers before capping
    # print(f"Number of outliers before capping: {len(dt[IQR_Outliers])}")
    
    # 1563 rows are outliers, to handle those capping technique is the best  
    low_limit = Q1 - 1.5 * IQR
    up_limit = Q3 + 1.5 * IQR
    new_data[['satisfaction_level', 'last_evaluation', 'number_project', 'hours', 'years', 'workload']] = dt.clip(lower=low_limit, upper=up_limit, axis=1)
    
    
    
    dt_capped = new_data[['satisfaction_level', 'last_evaluation', 'number_project', 'hours', 'years', 'workload']]
    IQR_Outliers_after = ((dt_capped < (Q1 - 1.5 * IQR)) & (dt_capped > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # print(f"Number of outliers after capping: {len(new_data[IQR_Outliers_after])}")
    
    """
    # chekcing any outliers on each individual column by plotting individually
    
    plt.figure()
    sns.boxplot(x = df['satisfaction_level'])    
    plt.figure()
    sns.boxplot(x = df['last_evaluation'])    
    plt.figure()
    sns.boxplot(x = df['number_project'])    
    plt.figure()
    sns.boxplot(x = df['hours'])    
    plt.figure()
    sns.boxplot(x = df['years'])    
    plt.figure()
    sns.boxplot(x = df['workload'])
    
    """
    
    # By plotting each individual columns we find  workload has the outliers in it
    # q1 = df['workload'].quantile(0.25)
    # q3 = df['workload'].quantile(0.75)
    # iqr = q3-q1
    
    
    
    # wrk_up_limit = q3 + (1.5*iqr)
    # wrk_low_limit =q1 - (1.5*iqr)
    
    upper_limit, lower_limit,outliers = find_outliers(df, 'workload')
    # print(f"Outliers before Capping: {len(df.loc[(df['workload']>wrk_up_limit) | (df['workload']<wrk_low_limit)])}")
    
    # there are 316 outliers by IQR to handle those using capping
    new_data.loc[new_data['workload'] > upper_limit, 'workload'] = upper_limit
    new_data.loc[new_data['workload'] < lower_limit, 'workload'] = lower_limit

    # print(f"Outliers after Capping:  {len(df.loc[(new_data['workload']>wrk_up_limit) | (new_data['workload']<wrk_low_limit)])}")
    """    
    plt.figure()
    sns.boxplot(x=df['workload'])
        
    plt.figure()
    sns.boxplot(x=new_data['workload'])

    """
    # qq1 = df['years'].quantile(0.25)
    # qq3 = df['years'].quantile(0.75)
    # iQr = qq3-qq1

    # yr_up_limit = qq3 + (1.5*iQr)
    # yr_low_limit =qq1 - (1.5*iQr)
    upper_limit, lower_limit, outliers = find_outliers(df, 'years')
    #capping
    new_data.loc[new_data['years'] > upper_limit, 'years'] = upper_limit
    new_data.loc[new_data['years'] < lower_limit, 'years'] = lower_limit

    # returning the new_data after cleaning the NAN values and converting to numerical values and finding and handling outliers
    
    return new_data

def count_and_fill_null(df, col_name):
    if isinstance(col_name, pd.Series):
        col_name = col_name.name
    elif not isinstance(col_name, str):
        raise TypeError(f"The '{col_name}' should be either a str or a series")
    
    if col_name not in df.columns:
        print(f"column: '{col_name}' is not in DataFrame")
        return
    
    null_count = df[col_name].isna().sum()
    if null_count>0:
        if df[col_name].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col_name]):
            fill_val = 'unknown'
        else:
            fill_val = 0

        df[col_name].fillna(fill_val, inplace=True)
        print(f"Column : '{col_name}' had '{null_count}' null values. Filled with '{fill_val}'.")

    else:
        print(f"Column: '{col_name}' has no null values")

def find_outliers(df, col_name):
    if col_name not in df.columns:
        print(f"{col_name} is not in dataframe")
        return None, None, None

    if not pd.api.types.is_numeric_dtype(df[col_name]):
        print(f"Column '{col_name} is not numeric")
        return None, None, None
    
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + (1.5*IQR)
    lower_limit = Q1 - (1.5*IQR)
    
    outliers = ((df[col_name] < lower_limit) | (df[col_name] > upper_limit)).sum()
    # print(f"Outliers in '{col_name}' is/are '{outliers}'.")
    return upper_limit, lower_limit, outliers

# EDA
def EDA_main(path, ndata):
    # Call the EDA function with the updated path to your dataset
    df_with_outliers, df_without_outliers = EDA(path, ndata)

    # Get the outlier counts
    outlier_counts = count_outliers(df_with_outliers, df_without_outliers)

    # Outliers count for each feature before and after
    outliers_before_after(outlier_counts)

    # Plot boxplots for the original data with outliers and without outliers
    plot_boxplots(df_with_outliers, df_without_outliers)
    print("\nDataFrame without outliers:")
    print(df_without_outliers)
    print(f"Length of DataFrame without outliers: {len(df_without_outliers)}")


# EDA function
def EDA(dataset_path, ndata):
    # Load the dataset into a DataFrame
    df = pd.read_csv(dataset_path)
    df.rename(columns={'average_montly_hours':'hours', 'time_spend_company':'years', 'promotion_last_5years':'promoted', 'sales':'dept'}, inplace=True)
    
    df['workload'] = (df['hours']/df['number_project']).round(4)
    
    # Select specific columns for correlation analysis (only numeric columns)
    df_corr = ndata[['satisfaction_level', 'last_evaluation', 'number_project', 'hours', 'years', 'Work_accident', 'left', 'promoted']]
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    # Save the plot
    save_plot('Correlation Heatmap.png')
    plt.show()
    # Plot distribution of 'satisfaction_level'
    plt.figure(figsize=(10, 6))
    sns.histplot(df['satisfaction_level'], kde=True, bins=30)
    plt.title('Distribution of Employee Satisfaction')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Frequency')
    save_plot('satisfaction_distribution.png')
    plt.show()
    # Plot distribution of 'last_evaluation'
    plt.figure(figsize=(10, 6))
    sns.histplot(df['last_evaluation'], kde=True, bins=30)
    plt.title('Distribution of Employee Evaluation')
    plt.xlabel('Last Evaluation')
    plt.ylabel('Frequency')
    save_plot('evaluation_distribution.png')
    plt.show()
    # Plot distribution of 'hours'
    plt.figure(figsize=(10, 6))
    sns.histplot(df['hours'], kde=True, bins=30)
    plt.title('Distribution of Employee Average Monthly Hours')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    save_plot('hours_distribution.png')
    plt.show()
    # Plot count of 'number_project' grouped by 'left'
    plt.figure(figsize=(12, 6))
    sns.countplot(x='number_project', hue='left', data=df, palette='Set2')
    plt.title('Employee Project Count by Turnover')
    plt.xlabel('Number of Projects')
    plt.ylabel('Number of Employees')
    plt.legend(title='Turnover Status', loc='upper right', labels=['Stayed', 'Left'])
    save_plot('project_count_by_turnover.png')
    plt.show()
    return df, ndata


# Function to calculate the number of outliers in each column
def count_outliers(df, ndata):
    outlier_counts = {}
    numeric_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'hours', 'years', 'workload']
    for column in numeric_columns:
        # Outliers in the original data
        upper_limit, lower_limit, outliers = find_outliers(df, column)
        # Q1 = df[column].quantile(0.25)
        # Q3 = df[column].quantile(0.75)
        # IQR = Q3 - Q1
        # outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
        original_outlier_count = outliers
        # Outliers in the cleaned data
        # cleaned_outliers = ((ndata[column] < (Q1 - 1.5 * IQR)) | (ndata[column] > (Q3 + 1.5 * IQR)))
        upper_limit, lower_limit, outliers = find_outliers(ndata, column)
        cleaned_outlier_count = outliers
        # Store the counts
        outlier_counts[column] = {
            'Original Outliers': original_outlier_count,
            'Cleaned Outliers': cleaned_outlier_count
        }
    return outlier_counts

# Print the outlier counts
def outliers_before_after(outlier_counts):
    print("Outlier Counts (Original vs Cleaned):")
    for column, counts in outlier_counts.items():
        print(f"{column}:")
        print(f"\tOriginal Outliers: {counts['Original Outliers']}")
        print(f"\tCleaned Outliers: {counts['Cleaned Outliers']}")
        print()

# Plot boxplots to show the difference with and without outliers in the same row
def plot_boxplots(df_with_outliers, df_without_outliers):
    columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'hours', 'years', 'workload']

    for column in columns:
        plt.figure(figsize=(14, 6))
        # Boxplot with outliers
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df_with_outliers[column])
        plt.title(f'{column} with Outliers')
        # Boxplot without outliers
        plt.subplot(1, 2, 2)

        sns.boxplot(x=df_without_outliers[column])
        plt.title(f'{column} without Outliers')
        save_plot(f'{column}_boxplot.png')
        plt.show()

# without SMOTE
def without_smote(df):
    df_cp = df.copy()
    
    # Prepare the data
    X = df_cp.drop(columns=['left'])
    y = df_cp['left']
    
    # Handle missing values
    count_and_fill_null(df_cp, df_cp['KMeans_Cluster'])
    count_and_fill_null(df_cp, df_cp['Cluster_Label'])
    count_and_fill_null(df_cp, df_cp['outliers'])
    
    df_cp['KMeans_Cluster'] = df_cp['KMeans_Cluster'].astype(int)
    df_cp['outliers'] = df_cp['outliers'].astype(int)
    
    # Encode categorical columns consistently
    categorical_columns = ['dept', 'salary', 'satisfaction_category', 'Cluster_Label']
    df_cp = pd.get_dummies(df_cp, columns=categorical_columns, drop_first=False)
    
    # Prepare the data
    X = df_cp.drop(columns=['left'])
    y = df_cp['left']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    
    # Print class distribution before SMOTE
    print("Class distribution before SMOTE:")
    print(y_train.value_counts())
    
    return X_train, y_train, X_test, y_test


# SMOTE
def smote_data(df):
    df_cp = df.copy()
    
    # Handle missing values
    count_and_fill_null(df_cp, df_cp['KMeans_Cluster'])
    count_and_fill_null(df_cp, df_cp['Cluster_Label'])
    count_and_fill_null(df_cp, df_cp['outliers'])

    # Convert columns to appropriate types
    df_cp['KMeans_Cluster'] = df_cp['KMeans_Cluster'].astype(int)
    df_cp['outliers'] = df_cp['outliers'].astype(int)
    
    # Encode categorical variables
    categorical_columns = ['dept', 'salary', 'satisfaction_category', 'Cluster_Label']
    df_cp = pd.get_dummies(df_cp, columns=categorical_columns, drop_first=False)
    
    # Prepare the data
    X = df_cp.drop(columns=['left'])
    y = df_cp['left']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Applying SMOTE to balance the training data

    smote = SMOTE(random_state=123)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    #The formula for generating a synthetic sample is:
    #Synthetic Sample=Xminority+λ×(Xneighbor−Xminority)
    #Xminority is the original minority class sample.
    #Xneighbor is the randomly selected neighbor.
    #λlambda is a random number between 0 and 1.
    
    # Print class distribution after SMOTE
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Print the size of the 'left' column after SMOTE
    print(f"Size of 'left' column after SMOTE: {len(y_train_resampled)}")

    # Plot: Class distribution before SMOTE
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    y_train.value_counts().plot(kind='bar', title='Class Distribution Before SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Frequency')

    # Plot: Class distribution after SMOTE
    plt.subplot(1, 2, 2)
    pd.Series(y_train_resampled).value_counts().plot(kind='bar', title='Class Distribution After SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    return X_train_resampled, y_train_resampled, X_test, y_test


# 5-Fold Cross validation
def evaluate_model(model):
    n_samples = 10000
    n_features = 20
    n_classes = 2
    n_splits = 5
    random_state = 123
    # Generate synthetic data
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)
    # Stratified K-Fold Cross Validator
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    # Initialize list to store mean values for each fold
    fold_means = []
    # Cross-validation predictions with mean calculation for each fold
    for fold_index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        model.fit(X[train_index], y[train_index])
        y_pred_fold = model.predict(X[test_index])
        fold_mean = np.mean(y_pred_fold)
        fold_means.append(fold_mean)
        print(f"Mean of Predictions for Fold {fold_index}: {fold_mean:.4f}")
    # Calculate and print the overall mean across all folds
    overall_mean = np.mean(fold_means)
    print(f"\nOverall Mean of Predictions for {model.__class__.__name__}: {overall_mean:.4f}\n")
    # Cross-validation predictions
    y_pred = cross_val_predict(model, X, y, cv=skf)
    # Confusion matrix and classification report
    conf_matrix = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for {model.__class__.__name__}:\n{conf_matrix}\n")
    
    class_report = classification_report(y, y_pred, output_dict=True)
    print(f"{model.__class__.__name__} Classification Report:\n")
    print(classification_report(y, y_pred))
    # Heatmap for classification report
    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title(f'{model.__class__.__name__} Classification Report')
    plt.show()
    
    return model,X,y,skf


def plot_roc_curve(models):
    plt.figure(figsize=(8, 8))
    for value in models:
        model,X,y,skf=value
        # Cross-validation probabilities
        y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {roc_auc:.4f})')
    # Plot diagonal line for random guessing
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Different Models')
    plt.legend(loc='lower right')
    plt.show()


def five_fold():
# Instantiate models
    log_reg = LogisticRegression(max_iter=10000)
    rf = RandomForestClassifier(max_depth=5, random_state=123)
    gb = GradientBoostingClassifier(random_state=123)

    # List to store trained models
    models = []

    # Evaluate each model
    for model in [log_reg, rf, gb]:
        trained_model = evaluate_model(model)
        models.append(trained_model)

    # Return the list of trained models
    return models




# Model
def model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123)
    rf.fit(X_train,y_train)

    y_prediction_probability = rf.predict_proba(X_test)[:, 1]
    y_prediction = rf.predict(X_test)

    def risk_categorize(val):
        if val<0.20:
            return 'Safe Zone (Green)'
        elif 0.20<=val<0.60:
            return 'Low Risk Zone (Yellow)'
        elif 0.60<=val<0.90:
            return 'Medium Risk Zone (Orange)'
        else:
            return 'High Risk Zone'

    risk_zones = np.array([risk_categorize(val) for val in y_prediction_probability])

    result = pd.DataFrame({
        'probability' : y_prediction_probability,
        'Risk_Zone' : risk_zones
    })
    acc_score = accuracy_score(y_test, y_prediction)
    print(f"Accuracy score : \n{acc_score}")

    cr = classification_report(y_test, y_prediction)
    print(f"CLassification Report: \n{cr}")

    roc_auc = roc_auc_score(y_test, y_prediction_probability)
    print(f"ROC AUC : \n{roc_auc}")

    return result


def plot_zones(result, val):
    plt.figure(figsize=(4, 4))
    sns.countplot(x='Risk_Zone', data=result, palette={
        "Safe Zone (Green)": "green",
        "Low Risk Zone":"yellow",
        "Medium Risk Zone":"orange",
        "High Risk Zone":"red"
    })

    plt.title(f"plot {val}: Risk zones")
    plt.xlabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plots(result1, result2, val1, val2):
    plt.figure(figsize=(10, 6))
    # count-plot without smote
    plt.subplot(1, 2, 1)
    sns.countplot(x='Risk_Zone', data=result1, palette={
        "Safe Zone (Green)": "green",
        "Low Risk Zone":"yellow",
        "Medium Risk Zone":"orange",
        "High Risk Zone":"red"
    })
    plt.title(f"plot {val1}: Risk zones")
    plt.xlabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # count-plot with smote
    plt.subplot(1, 2, 2)
    sns.countplot(x='Risk_Zone', data=result2, palette={
        "Safe Zone (Green)": "green",
        "Low Risk Zone":"yellow",
        "Medium Risk Zone":"orange",
        "High Risk Zone":"red"
    })
    plt.title(f"plot {val2}: Risk zones")
    plt.xlabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# replace the path with file directory
# path = 'C:\\pushpa\\python material\\projects_assignments_8-6-24\\project-2\\HR_comma_sep.csv'
path ='./HR_comma_sep.csv'

data = data_preprocess(path)
EDA_main(path,data)
new_df = cluster_kmean(data)
# print(new_df)
# print(len(new_df))

X_train, y_train, X_test, y_test = without_smote(new_df)
X_train_resampled, y_train_resampled, X_test, y_test = smote_data(new_df)

trained_models = five_fold()
plot_roc_curve(trained_models)

# plot_zones(result1, "with_smote")
result1 = model(X_train, y_train, X_test, y_test)

# plot_zones(result1, "without_smote")
result2 = model(X_train_resampled, y_train_resampled, X_test, y_test)

# plot_zones(result2, "with smote")
plots(result1, result2, "without smote", "with smote")

print(f"\nThe data set without smote : \n{result1}")
print(f"\nThe data set with smote : \n{result2}")