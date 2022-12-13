import numpy as np
from sklearn.tree import DecisionTreeClassifier
from time import time

###########################################
#                 WARNING                 #
#        DO NOT CALL THIS FUNCTION        #
# WITHOUT NARROWING SOME PARAMETERS FIRST #
###########################################

def get_best_tree(data,
                  labels,
                  criterions = ['gini', 'entropy', 'log_loss'],
                  splitters = ['best', 'random'],
                  depths = np.append(np.arange(1, 1001), None),
                  min_samples_split = np.arange(2, 2001),
                  min_samples_leaf = np.arange(1, 2001),
                  max_features = [None, "sqrt", "log2"],
                  max_leaf_nodes = np.append(np.arange(1, 1501), None),
                  ccp_alpha = np.arange(0.0, 1.01, step = 0.01, dtype = float),
                  with_best = False,
                  with_time = False,
                  iterations = 100,
                  sample_size = 20):
    # get the time at the start of the call
    start = time()
    # set the best Mean Average Error as an absurdly large number
    best = 1e12
    # initiate the best tree as a new DecisionTreeClassifier
    best_tree = DecisionTreeClassifier()

    # for every possible comination of the specified parameters
    for crit in criterions:
        for split in splitters:
            for depth in depths:
                for sample_split in min_samples_split:
                    for sample_leaf in min_samples_leaf:
                        for max_feature in max_features:
                            for leaf_nodes in max_leaf_nodes:
                                for alpha in ccp_alpha:
                                    # create a DecisionTreeClassifier for that specific combination
                                    print("Generating new tree with new combination of parameters.\n" +
                                          "*******************************************************\n")
                                    clf = DecisionTreeClassifier(
                                        criterion=crit,
                                        splitter=split,
                                        max_depth=depth,
                                        min_samples_split=sample_split,
                                        min_samples_leaf=sample_leaf,
                                        max_features=max_feature,
                                        max_leaf_nodes=leaf_nodes,
                                        ccp_alpha=alpha
                                    )
                                    totaltestmae = 0
                                    totaltrainmae = 0
                                    print("Training and testing new tree...\n")
                                    for _ in range(iterations):
                                        xtest, ytest, xtrain, ytrain = randomize_test(data, labels, sample_size)
                                        # fit it to our data and labels
                                        clf = clf.fit(xtrain, ytrain)
                                        # train the DecisionTreeClassifier
                                        train = clf.predict(xtrain)
                                        # compute the absolute error
                                        trainerror = np.abs(ytrain - train)
                                        # test the DecisionTreeClassifier
                                        test = clf.predict(xtest)
                                        # compute the test error
                                        testerror = np.abs(ytest - test)
                                        # compute the Mean Absolute Error for the training samples
                                        trainmae = sum(trainerror) / len(trainerror)
                                        totaltrainmae += trainmae
                                        # compute the Mean Absolute Error for the testing samples
                                        testmae = sum(testerror) / len(testerror)
                                        totaltestmae += testmae
                                        print(f'Training MAE: {trainmae}. Testing MAE: {testmae}.')
                                    # if it is better than our best, then assign it to be our new best
                                    if((totaltestmae / iterations) < best):
                                        best = totaltestmae / iterations
                                        print(f'New Best: {best}')
                                        best_tree = clf
    # get the time at the end of the call
    end = time()
    # compute the time differnce
    diff = end - start
    # return based on the specified flags
    if with_best and with_time:
        return best_tree, best, diff
    elif with_best:
        return best_tree, best
    elif with_time:
        return best_tree, diff
    return best_tree

def randomize_test(data, labels, num_samples = 20):
    # create an array that lists all of the possible indices
    possible_indices = np.arange(0, len(data))
    # get a list of unique indices from the possible indices of size num_samples
    indices = np.random.choice(possible_indices, size = num_samples, replace = False)
    # get the test and train data and labels
    test_data = np.take(data, indices, 0)
    test_labels = labels[indices]
    train_data = np.delete(data, indices, 0)
    train_labels = np.delete(labels, indices)
    return test_data, test_labels, train_data, train_labels