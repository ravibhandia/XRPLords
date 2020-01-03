from sspipe import p, px

## Load all python dependencies 

def load_dependencies():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()
    from sklearn import preprocessing
    #from scipy.stats import shapiro
    from scipy.cluster.hierarchy import dendrogram, linkage 
    from scipy.cluster.hierarchy import fcluster
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn import metrics
    from scipy.spatial.distance import cdist
    return ["Dependencies have been loaded"]

def ts_to_strings(ts):
    """
    Returns a list of strings from XRP timestamps.

    Parameters:
    -----------
    ts: pandas timestamp col
    """
    return [str(x)[:19] for x in ts];

def strings_to_datetime(col):
    """
    Returns a Pandas column of datetime objects from list of strings.

    Parameters:
    -----------
    col: list
      List of strings representing datetimes
    """
    return [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in col];

###################################################################

## Data manipulating + scaling fns 

def get_nine_pollut(raw_data):
    return raw_data.iloc[:, 0:9]

def log_transform_df(dat):
    return dat.apply(np.log)

def mean_normalize_df(dat):
    return (dat - dat.mean()) / dat.std()

## Scale variables (pollutants) by: (x_i - min(x)) / (max(x) - min(x))

def min_max_scaler(dat):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(dat)

## Robust scaling handles outliers by doing min-max with interquartile range

def robust_scaler(dat):
    scaler = preprocessing.RobustScaler()
    return scaler.fit_transform(dat)

## Normalizer scaling divides each value by its magnitude in n-dimensional space 
##   where n is the number of variables (i.e. pollutants)

def normalizer_scaler(dat):
    scaler = preprocessing.Normalizer()
    return scaler.fit_transform(dat)

##################################################################

## Data viz fns 

def show_beautiful_bivariate_groupings(dat, color_by = False):
    g = sns.PairGrid(dat, hue = color_by)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.add_legend();
    
def show_correlation_heatmap(corr_matrix):
    fig, ax = plt.subplots(figsize = (12, 12))  
    ax = sns.heatmap(corr_matrix, annot = True, linewidths = 1, ax = ax, cmap = "Reds")
    fig = ax.get_figure()

## Make boxplot of single column 

def show_col_boxplot(df, col):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.gca()
    df.boxplot(column = col, ax = ax)
    ax.set_title("Filtered column: " + col)
    #ax.set_xlabel(col)
    ax.set_ylabel("Values")
    return "show_col_boxplot() for " + col

def get_each_col_boxplot(df, col_names):
    ## List comprehension (call fn on each item in list)
    # [fn_to_call() for item in the_list]
    return [show_col_boxplot(df, col_name) for col_name in col_names]

## NOTE: need "import seaborn as sns"

def get_cluster_boxplot(clusters,feature):
    return sns.boxplot(x=clusters, y=feature, data=data)

def get_boxplot(feature):
    return sns.boxplot(y=data[feature])

def get_hist(feature):
    return sns.distplot(data[feature])


###################################################################

## K-means fns 


def get_kmeans_model(df, k):
    """
        k = number of clusters
    """
    return KMeans(n_clusters = k).fit(df)

def get_kmeans_labels(df, kmeans_models):
    return kmeans_models.predict(df)

def get_cdist(df, clust_centers):
    return (cdist(df, clust_centers, 'euclidean'))  # axis = 1

def find_np_min(df, clust_centers):
    return (np.min(get_cdist(df, clust_centers)), 1)

def sum_np_min_cdist(df, clust_centers):
    return [sum(find_np_min(df, clust_centers)) / df.shape[0]]

def get_kmeans_distortions(df, kmeans_model):
    #return (sum(np.min(cdist(df, kmeans_models.cluster_centers_, 'euclidean'), axis = 1)) / df.shape[0])
    return sum_np_min_cdist(df, kmeans_model.cluster_centers_)

def return_cluster_indices(kmeans_model, k):
    """
        k = cluster number to check 
        kmeans_model = (after KMeans(...).fit(df)) predicted labels on data (i.e. df)
        
        This function can be used to return the indices of our original data 
        that kmeans_model predicted to be of cluster-id = k 
        
        E.g. > return_cluster_indices(3, kmeans_model) 
        
            Will return the indices of the original data that were clustered 
            to cluster-id = 3 (i.e. the 3rd cluster of K clusters total)
    """
    return (np.where(k == kmeans_model.labels_)[0])

def return_n_kmeans(df, n = 5):
    """
        n = number of k-means models to generate
            with number-of-clusters from 1:n 
            
        This function returns a 2-d array: 
            index-0 | k-means model 
            index-1 | respective labels of model on data (predict)
            index-2 | respective cluster centroid distances (distortions)
    """
    k_models = []
    model_predicts = []
    distortions = []
    n_iters = range(1, n)
    for k in n_iters:
        print(k)
        
        ## Generate kmeans for number-of-clusters = k
        k_models.append(get_kmeans_model(df, k))
        
        ## Predict kth kmeans model on data 
        model_predicts.append(get_kmeans_labels(df, k_models[(k - 1)]))
        
        ## Calculate cluster centroid distances (distortions)
        distortions.append(get_kmeans_distortions(df, k_models[(k - 1)]))
        
    return [k_models, model_predicts, distortions]
        
######################################################################

 ## find the nearest neighbors
def knearst_plot(data):
    neigh = NearestNeighbors(n_neighbors=2) 
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances_sort = np.sort(distances, axis=0)
    distances_nearest = distances_sort[:,1]
    plt.plot(distances_nearest)
    
# Compute DBSCAN

def call_dbscan (eps, min_samples, data):  
    return DBSCAN(eps=eps, min_samples=min_samples).fit(data)

def get_dbscan_results(db_model):
    labels = db_model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1) 
    outliers_index=[i for i,x in enumerate(list(labels)) if x == -1]
    return [labels, n_clusters, n_noise, outliers_index]

def get_point_boxplot(pollutant, data):
    ax=sns.boxplot(x=data[pollutant])
    ax=sns.swarmplot(x=data[pollutant],color=".25")
    return ax

