import numpy as np
import numpy.linalg as la


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # initialize to k random points
        self.k_means = x[np.random.randint(x.shape[0], size=self.n_cluster), :]
        # for each cluster, recalc cluster mean
        # distortion
        j = 0
        indicators = None
        update_count = self.max_iter

        for iter in range(self.max_iter):
            indicators = self.indicator(x)
            j_new = 0
            for cluster_num in range(len(self.k_means)):
                # calculate nearest points to each mean and calc new mean
                points_in_cluster = []
                for xn, indn in zip(x,indicators):
                    # get indicator index use argmax because storing euclidean distance to mean instead of 0,1
                    if indn[0] == cluster_num:
                        points_in_cluster.append(xn)
                        euclidean_distance = indn[1]
                        # calculating distortion measure
                        j_new += euclidean_distance / len(x)

                self.k_means[cluster_num] = np.mean(points_in_cluster, axis=0)
                
            if(np.abs(j_new - j) <= self.e):
                update_count = iter
                break
            j = j_new
            
        membership = np.array([ i[0] for i in indicators ])
        return self.k_means, membership, update_count+1
            
        # DONOT CHANGE CODE BELOW THIS LINE
    def indicator(self, x):
        indicator_mat = []
        for i,xn in enumerate(x, start=0):
            euclidean_distances = [ la.norm(xn - u) for u in self.k_means ]          
            centroid = np.argmin(euclidean_distances)
            distance = min(euclidean_distances)
            indicator_mat.append((centroid,distance))
        return indicator_mat

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape

        # - assign means to centroids
        # - assign labels to centroid_labels

        k_means = KMeans(self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, memberships, _ = k_means.fit(x)
        
        centroid_labels_buckets = [[0] for l in range(self.n_cluster)]

        for i, yi in enumerate(y):
            centroid_labels_buckets[memberships[i]].append(yi)

        centroid_labels = []
        for u in centroid_labels_buckets:
            centroid_labels.append(np.bincount(u).argmax()) 

        self.centroid_labels = np.array(centroid_labels) 
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        predictions = []
        for xn in x:
            min_distance_ind = np.argmin([ np.power(la.norm(xn - u), .5) for u in self.centroids ])
            predictions.append(self.centroid_labels[min_distance_ind])

        return np.array(predictions)

        # return np.arrays([ self.centroid_labels[self.calc_nearest_centroid(xn)] for xn in x ])

    # def calc_nearest_centroid(self, xn):
    #     euclidean_distances = [ la.norm(self.centroids - u) for u in self.centroids ]
    #     centroid = np.argmin(euclidean_distances)
    #     return centroid