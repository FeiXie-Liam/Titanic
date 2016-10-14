import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import preprocessing


class data_io(object):

    def __init__(self):
        pass

    def read_train_data(self, path):
        with open(path, 'rb') as contents:
            df = pd.read_csv(contents, header=0)
            # print(df.shape)
            df['Sex'] = df['Sex'].map(
                {'female': 0, 'male': 1}).astype(int)
            # print (df.head(10))
            df['Agefill'] = df['Age']
            # print("origin:",df['Agefill'].head(10))
            median = np.zeros((2, 3))
            fare_median= np.zeros((2, 3))
            for i in range(0, 2):
                for j in range(0, 3):
                    median[i][j] = df[(df['Sex'] == i) &
                                      (df['Pclass'] == j + 1)]['Age'].dropna().median()
                    fare_median[i][j] = df[(df.Sex == i) & (
                        df.Pclass == j + 1)]['Fare'].dropna().median()
                    df.loc[(df.Age.isnull()) & (df.Sex == i) &
                           (df.Pclass == j + 1), 'Agefill'] = median[i][j]
                    df.loc[(df.Fare.isnull()) & (df.Sex == i) &
                           (df.Pclass == j + 1), 'Fare'] = median[i][j]
            # print('filled:',df['Agefill'].head(10))
            # print(type(df[(df.Age.isnull())]))
            df = df.drop(['PassengerId', 'Name', 'Ticket',
                          'Cabin', 'Embarked', 'Age'], axis=1)
            df['Family'] = df.SibSp + df.Parch
            df = df.drop(['Agefill', 'SibSp', 'Parch'], axis=1)
            # df['newFeature'] = df['Agefill'] * df['Pclass']
            print(df.head(10))
            train_x = np.array(df)[:, 1:]
            train_y = np.array(df)[:, 0]
            # print('train_x shape:', train_x.shape)
            return train_x, train_y

    def read_test_data(self, path):
        with open(path, 'rb') as contents:
            df = pd.read_csv(contents, header=0)
            ids = df.PassengerId.values
            df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
            median = np.zeros((2, 3))
            fare_median= np.zeros((2, 3))
            df['Agefill'] = df['Age']
            for i in range(0, 2):
                for j in range(0, 3):
                    median[i][j] = df[(df['Sex'] == i) &
                                      (df['Pclass'] == j + 1)]['Age'].dropna().median()
                    fare_median[i][j] = df[(df.Sex == i) &
                                           (df.Pclass == j + 1)]['Fare'].dropna().median()
                    df.loc[(df.Age.isnull()) & (df.Sex == i) &
                           (df.Pclass == j + 1), 'Agefill'] = median[i][j]
                    df.loc[(df.Fare.isnull()) & (df.Sex == i) &
                           (df.Pclass == j + 1), 'Fare'] = median[i][j]
            # df['newFeature'] = df['Agefill'] * df['Pclass']
            df = df.drop(['PassengerId', 'Name', 'Ticket',
                          'Cabin', 'Embarked', 'Age'], axis=1)
            df['Family'] = df.SibSp + df.Parch
            df = df.drop(['Agefill', 'SibSp', 'Parch'], axis=1)
            test_x = np.array(df)
            return test_x, ids


if __name__ == '__main__':
    io = data_io()
    train_x, train_y = io.read_train_data('./train.csv')
    test_x, ids = io.read_test_data('./test.csv')
    split_size = len(train_x) * 3 / 4
    # Data preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x_minmax = min_max_scaler.fit_transform(train_x)
    test_x_minmax = min_max_scaler.fit_transform(test_x)

    cross_validation_x = train_x_minmax[split_size:]
    cross_validation_y = train_y[split_size:]
    # # neural_network method
    # clf = MLPClassifier(solver='lbgfs', alpha=1e-5,
    #                     hidden_layer_sizes=(5, 5), random_state=1)
    # clf.fit(train_x_minmax, train_y)
    #
    # # # svm method
    # # clf = svm.SVC()
    # # clf.fit(train_x, train_y)
    # # print(test_x)
    # result = clf.predict(test_x_minmax)
    # my_submission = pd.DataFrame()
    # my_submission['PassengerId'] = ids
    # my_submission['Survived'] = result.astype(int)
    # print(my_submission)
    # my_submission.to_csv('./my_submission.csv', index=False)

    for i in range(0, 10):
        part_train_x = train_x_minmax[:split_size * (i + 1) / 10]
        part_train_y = train_y[:split_size * (i + 1) / 10]
        # neural_network method
        clf = MLPClassifier(solver='lbgfs', alpha=1e-5,
                            hidden_layer_sizes=(5, 5), random_state=1)
        clf.fit(part_train_x, part_train_y)

        # # svm method
        # clf = svm.SVC()
        # clf.fit(part_train_x, part_train_y)

        train_result = clf.predict(part_train_x)
        tp = np.array([(train_result[i] == part_train_y[i])
                       for i in range(0, train_result.shape[0])]).astype(int)
        train_accracy = tp.mean()

        cv_result = clf.predict(cross_validation_x)
        tp = np.array([(cv_result[i] == cross_validation_y[i])
                       for i in range(0, cv_result.shape[0])]).astype(int)
        cv_accuracy = tp.mean()
        print('train data count:', len(part_train_y), 'train accuracy:',
              train_accracy, 'cv_accuracy', cv_accuracy)
