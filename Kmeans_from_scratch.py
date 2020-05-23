import numpy as np

class Kmeans():
    '''
    Implements the Kmeans algorithm for clustering image data.
    
    Parameters
    ----------
    n_clusters:
        The number of clusters the algorithm is going to fit.
        
    max_iteration:
        The number of iterations the EM algorithm is going to execute.
        
    tolerance:
        The percentage of change between the centroids of conconsecutive iterations
        tolerated. If the change is smaller the algorithm will be considered as
        converged.

    '''
    
    def __init__(self, n_clusters=3, max_iterations=100, tolerance=0.03):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def Initialize_centroids(self, data_points):
        '''
        Randomly initializes n_clusters centroids.
        Assigns the randomly initialized centroids to the class' centroids.

        Parameters
        ----------
        data_points :
            A numpy array of shape (number_of_datapoints, datapoint size).

        Returns
        -------
        None.

        '''
        sample = np.random.randint(0, data_points.shape[0], self.n_clusters)
        centroids = []
        for i in sample: centroids.append(data_points[i])
        self.centroids = np.array(centroids)
    
    def Initialize_labels(self):
        '''
        Initialize a dictionary of n_clusters empty lists. 
        The lists will hold the datapoints that belong to each cluster.
        Assigns the initialized labels to the class' labels.

        Returns
        -------
        None.

        '''
        labels = {}
        for k in range(self.n_clusters):
            labels[k] = []
        self.labels = labels
    
    def copy_centroids(self):
        '''
        Creates a copy of the class' centroids for comparison with the chnaged 
        centroids in the next iteration.

        Returns
        -------
        old_centroids
            A copy of the class' centroids.

        '''
        old_centroids = np.zeros(self.centroids.shape)
        for c in range(self.n_clusters):
            old_centroids[c] = self.centroids[c]
        return old_centroids.astype(int)
    
    def Convergence(self, data_points, old_centroids):
        '''
        Measures the change between the old and current centroids after each iteration.
        If the change is smaller than the tolerance then the algorithm has converged.

        Parameters
        ----------
        data_points :
            A numpy array of shape (number_of_datapoints, datapoint size).
        old_centroids :
            A numpy array of the previous iteration's centroids.

        Returns
        -------
        convergence
            True or False value for the convergene of the algorithm.

        '''
        convergence = False
        change = np.sum(np.abs(self.centroids - old_centroids) / old_centroids * 100.0)
        if change < self.tolerance:
            convergence = True
        return convergence
    
    def E_Step(self, data_points):
        '''
        Estimation step of the EM algorithm. 
        In this step all the datapoints are assigned to the cluster that they are 
        closest to.\n
        Euclidean distance is used to measure the distance between 
        a datapoint and each centroid of n_clusters.

        Parameters
        ----------
        data_points :
            A numpy array of shape (number_of_datapoints, datapoint size).

        Returns
        -------
        None.

        '''
        for point in data_points:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            cluster = distances.index(min(distances))
            self.labels[cluster].append(point)
    
    def M_Step(self):
        '''
        Maximization step of the EM algorithm.
        In this step the new centroids get calculated as the mean of all the 
        datapoints they have been assigned.

        Returns
        -------
        None.

        '''
        for k in self.labels:
            self.centroids[k] = np.average(self.labels[k], axis=0).astype(int)
    
    def fit(self, data):
        '''
        The n_cluster centroids are fitted with the input image data via the EM
        algorithm.\n
        The centroids and labels are randomly initialized.\n
        The EM algorithm iterates a total of max_iterations, unless the convergence
        criterium is met.\n
        In each iteration the Estimation step and the Maximization step are 
        executed.\n 
        The centroids of each iteration are compared with the centroids
        of the previous iteration, to check for convergence.

        Parameters
        ----------
        data : 
            Input image data.

        Returns
        -------
        centroids
            The centroids that the EM algorithm has converged to.
        labels
            The final labels that hold the datapoints that belong to each cluster.

        '''
        data_points = data.reshape(-1, data.shape[-1])
        self.Initialize_centroids(data_points)
        
        for iteration in range(self.max_iterations):
            self.Initialize_labels()
            self.E_Step(data_points)
            old_centroids = self.copy_centroids()
            self.M_Step()
            if self.Convergence(data_points, old_centroids): break
        return self.centroids, self.labels

    def predict(self, data):
        '''
        Calculates the distances of each piexl in the input image data with the
        fitted centroids.

        Parameters
        ----------
        data : 
            Input image data.

        Returns
        -------
        clustered_data
            Each pixel of the input image data gets replaced by the centroid that
            is closest to.

        '''
        data_points = data.reshape(-1, data.shape[-1])
        clustered_data = np.zeros(data_points.shape)
        
        for i in range(data_points.shape[0]):
            distances = [np.linalg.norm(data_points[i] - centroid) for centroid in self.centroids]
            cluster = distances.index(min(distances))
            clustered_data[i] = self.centroids[cluster]
        return clustered_data.reshape(data.shape).astype(int)