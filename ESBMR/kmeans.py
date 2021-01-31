from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

def KMeans_partitions(posterior_similarity_matrix, method = "silhouette"):

    k_kwargs = {
        "init": "random",
        "n_init": 15,
        "max_iter": 500,
        "random_state": 20136,
    }
   
    ssd = [] # list containing the sum of squared distances fr each k
    silhouette_coefficients = {}
    kmeans_labels = {}
    for k in range(1,10):
        clustering = KMeans(n_clusters = k, **k_kwargs)
        clustering.fit(posterior_similarity_matrix)
        ssd.append(clustering.inertia_)
        if np.unique(clustering.labels_).shape[0] != 1:
            score = silhouette_score(posterior_similarity_matrix, clustering.labels_)
            silhouette_coefficients[k] = score
            kmeans_labels[k] = clustering.labels_
            
    if method == "silhouette":
        n_clusters = max(silhouette_coefficients, key=silhouette_coefficients.get)
        model.zu_est_kmeans = kmeans_labels[n_clusters]
    
    if method == "elbow":
        kl = KneeLocator(range(1,len(ssd)+1), ssd, curve="convex", direction="decreasing")
        n_custers = kl.elbow
        model.zu_est_kmeans = kmeans_labels[n_clusters]
    
    else:
        raise ValueError("Partition estimates can be obtained by selecting 'silhouette' or 'elbow' method for KMeans.")
    
    return model.zu_est_lmeans