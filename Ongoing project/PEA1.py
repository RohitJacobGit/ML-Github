from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn.cluster
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
#from kmodes.kmodes import KModes
import math

class PerformanceEnrichmentAnalysisClassifier():

    """
    PerformanceEnrichmentAnalysisClassifier is a classifier that makes
    predictions using enrichment analysis.
    Parameters
    ----------
    random_state : int, RandomState instance or None, optional, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    classes_ : array or list of array of shape = [n_classes]
        Class labels for each output.
    n_classes_ : array or list of array of shape = [n_classes]
        Number of label for each output.
    """

    def __init__(self,
                 cluster_method='KMeans',
                 number_of_clusters=10,
                 permutations=100,
                 random_state=None):
        self.cluster_method = cluster_method
        self.number_of_clusters = number_of_clusters
        self.permutations = permutations
        self.random_state = random_state

    def _cluster(self, df_train):
        """Cluster data using a given clustering algorithm.
        Args:
            df_normalized: data frame with normalized data
            cluster_method: string, by default 'kmeans', more in future
            number_of_clusters: number of clusters, int

        Returns:
            df_result: data frame with results of clustering in a compact form
        """

        # by default use KMeans
        if self.cluster_method == "KMeans":
            #print('here')
            cluster_method_instance = sklearn.cluster.KMeans(n_clusters=self.number_of_clusters)
        if self.cluster_method == "MeanShift":
            cluster_method_instance = sklearn.cluster.MeanShift()
        #if self.cluster_method == "KModes":
        #    cluster_method_instance = KModes(n_clusters=self.number_of_clusters)

        #print(cluster_method_instance)
        c = cluster_method_instance.fit(df_train)

        df_result = pd.DataFrame()
        if(self.cluster_method == "KModes"):
            df_result["centers_scaled"] = c.cluster_centroids_ .tolist()
        else:
            df_result["centers_scaled"] = c.cluster_centers_.tolist()
            d = pd.DataFrame(c.labels_, columns=['cluster_id']) \
              .groupby('cluster_id') \
              .indices \
              .items()
        df_result['rows'] = [rows for cluster_id, rows in d]

        return df_result

    def fit(self, X, y, **kwargs):
        """Fit the random classifier.
        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.
        Returns
        -------
        self : object
        """
        if not isinstance(X, pd.DataFrame):
            column_size = X.shape[1]
            columns = [x + str(i) for x, i in zip(["x"]*column_size, range(column_size))]
            X = pd.DataFrame(X, columns=columns)

        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)
        else:
            y = pd.Series(y, name="y")

        #print(X)
        df_result = self._cluster(X)
        df_for_pea = self._prepare_for_pea(df_result, y)
        self.z_score = self._get_pea_z_scores(df_for_pea)
        df_result['z_score'] = self.z_score
        df_result['class_label'] = self.z_score.z_score.apply(lambda x: max(x, key=x.get))
        self.df_result = df_result
        self.y_categories = np.unique(y)
        return self

    def _prepare_for_pea(self, df_result, target_feature_values_encoded):
        """Extracts data from res and prepares it in a form suitable for the
        performance enrichment analysis.

        Args:
            df_result: data frame with the cluster-relevant information (scaled cluster
                 centers, rows of the initial df in each cluster)
            target_feature_values_encoded: codes used in encoding

        Returns:
            df_for_pea: data frame with cluster_id and the encoded observable for
                        each row of the initial df
        """

        r_series = df_result['rows']
        rows_in_cluster = [[row, c] for c in range(len(r_series))
                           for row in r_series[c]]
        rows_in_cluster = pd.DataFrame(rows_in_cluster,
                                       columns=['index', 'cluster_id']) \
            .set_index('index')
        # by default join by index!
        df_for_pea = rows_in_cluster.join(target_feature_values_encoded)
        # print(df_for_pea)        index(index) and cluster_id, lateness
        # print(target_feature_values_encoded)
        return(df_for_pea)

    def _get_pea_z_scores(self, df_for_pea):
        """Performance enrichment analysis

        Args:
            df_for_pea: data frame with cluster_id and the encoded observable for
                        each row of the initial df
            permutations: number of permutations for pea algorithm

        Returns:
            z_score: data frame with a dict of z-scores for each cluster
        """

        # rename 2nd column to zero-th permutation
        # Add permutations columns to df_for_pea
        #print(list(df_for_pea))
        #print({list(df_for_pea)[1]: 0})
        df_for_pea.rename(columns={list(df_for_pea)[1]: 0},
                          inplace=True)
        # print(df_for_pea.columns)
        # add and randomize values
        #print(df_for_pea[0].sample(frac=1))
        for i in range(1, self.permutations + 1):
            # create 100 columns with different permutations
            # frac=1 for selecting sample with same number of rows but with random selection
            # shuffling on column = [0]
            df_for_pea[i] = df_for_pea[0].sample(frac=1).reset_index(drop=True)

        #print(df_for_pea)

        df_for_pea = df_for_pea.fillna(0)
        #print(pd.melt(df_for_pea, id_vars=['cluster_id']))

        p_count = pd.melt(df_for_pea,
                          id_vars=['cluster_id'],
                          var_name='permutation',
                          value_name='value').astype(int) \
            .groupby(['cluster_id',
                      'permutation',
                      'value'])['cluster_id'] \
            .count() \
            .reset_index(name='count')
        #print(p_count)
        # real measured count of different values of observable_encoded
        #print(p_count.loc[p_count["permutation"] == 0, :])
        p_0 = p_count.loc[p_count["permutation"] == 0, :] \
                     .drop("permutation", axis=1)
        #print(p_0)  # cluster_id  value  count
        #print(p_count) # cluster_id  permutation  value  count
        p_count = p_count.groupby(["cluster_id", 'value'])["count"] \
                         .agg(["mean", "std"]) \
                         .reset_index() \
                         .merge(p_0, on=["cluster_id", 'value'], how="left")
        #print(p_count) # cluster_id  value        mean        std  count
        p_count["z_score"] = (p_count["count"] - p_count["mean"]) / p_count["std"]
        p_count["z_score"] = p_count["z_score"].fillna(0)
        z_score = p_count.drop(["mean", "std", "count"], axis=1) \
                         .pivot(index="cluster_id",
                                columns='value',
                                values="z_score") \
                         .reset_index() \
                         .drop(columns='cluster_id') \
                         .apply(lambda x: x.to_dict(), axis=1) \
                         .rename('z_score') \
                         .to_frame()
        #print(z_score)
        return(z_score)

    def _pea_predict(self, testing_data):
        """Predics (assigns) in which cluster a new order will fall.

        Args:
            testing_data: data frame with new orders

        Returns:
            df_predicted: data frame with cluster and z-score for each order

        """
        df_testing_data = pd.DataFrame(testing_data)
        c = self.df_result["centers_scaled"].values
        #print(c) # cluster centroids
        c = np.array([np.array(ci) for ci in c])
        predicted_cluster_id = cdist(testing_data, c).argmin(axis=1)
        df_predicted = self.df_result.loc[predicted_cluster_id, "class_label"].reset_index()

        return(df_predicted.class_label)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df_predicted = self._pea_predict(X)
        return df_predicted.values

    def get_sum_of_z_scores(self):
        d = {}
        for category in self.y_categories:
            d[category] = 0

        for k, val in self.df_result.z_score.iteritems():
            for category in self.y_categories:
                if (not math.isnan(val[category])):
                    d[category] = d[category] + abs(val[category])

        return d
    # Todo store these two values in fit method.

    def get_sum_of_z_scores_normalized(self):
        d = self.get_sum_of_z_scores()
        for k in d:
            d[k] = d[k] / self.number_of_clusters
        return d

    def get_prediction_quality(self):
        prediction_quality = 0
        d = self.get_sum_of_z_scores()
        for k in d:
            prediction_quality = prediction_quality + d[k] / math.sqrt(self.number_of_clusters)
        return prediction_quality