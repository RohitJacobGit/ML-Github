from PEA import PerformanceEnrichmentAnalysisClassifier
from sklearn.neighbors import KNeighborsClassifier
import helper_data_generator
import numpy as np

classifier_list = [
     PerformanceEnrichmentAnalysisClassifier(
        number_of_clusters=20, permutations=100),
      KNeighborsClassifier(3),
     # SVC(kernel="linear", C=0.025)
     # SVC(gamma=2, C=1),
     # DecisionTreeClassifier(max_depth=5),
     # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
     # MLPClassifier(alpha=0.01),
     # AdaBoostClassifier(),
     # GaussianNB(),
     # QuadraticDiscriminantAnalysis()
]

# Balanced Labels, no noise, medium complexity
num_of_iterations = 5
number_of_samples = 1000
number_of_features = 100
number_informative_features = 90
number_redundant_features = 10
number_repeated_features = 0
number_classes = 3
number_clusters_per_class = 1
class_separator = 0.3
flip_y = 0
weights = [0.5, 0.5, 0.5]


# pea_list_accuracies, ml_list_accuracies = helper_data_generator.get_list_accuracies_classification_data(
#                                                       num_of_iterations,
#                                                       classifier_list,
#                                                       number_of_samples,
#                                                       number_of_features,
#                                                       number_informative_features,
#                                                       number_redundant_features,
#                                                       number_repeated_features,
#                                                       number_classes,
#                                                       number_clusters_per_class,
#                                                       class_separator,
#                                                       flip_y,
#                                                       weights)
#
# print(np.mean(pea_list_accuracies))
# print(np.mean(ml_list_accuracies))

list_noise_vals = list(np.round(np.arange(0,3,0.1),2))
list_1 = [helper_data_generator.get_accuracies(classifier_list, noise_level) for noise_level in list_noise_vals]
x, y = np.array(list_1).T
print(x.shape)
