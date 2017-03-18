from sklearn.svm import LinearSVC

def get_linear_svc(features_train, label_train):
    clf = LinearSVC()
    clf.fit(features_train, label_train)
    return clf
