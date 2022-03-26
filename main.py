# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:21:02 2020

@author: Jiang Jilian
"""

from all_models import *
from globalVar import *
from utils.normlization import *
import warnings

warnings.filterwarnings("ignore")


def prepare_data():
    """
    Returns:

    """
    zircons_data = pd.read_excel(dataPath + fileName + ".xlsx")
    zircons_data.loc[zircons_data["Zircon"] == "S-type zircon", "label"] = 1
    zircons_data.loc[zircons_data["Zircon"] == "I-type zircon", "label"] = 0
    cols = [x for x in zircons_data.index for i in elements if zircons_data.loc[x, i] == 0]
    zircons_data.drop(cols, inplace=True)
    zircons_data.reset_index(inplace=True, drop=True)
    zircons_data.dropna(subset=elements, inplace=True)
    zircons_data.reset_index(inplace=True, drop=True)
    raw_prediction_set = zircons_data[zircons_data["Set"] == "Prediction set"]
    raw_prediction_set.reset_index(inplace=True, drop=True)

    # Normalize all zircons
    x_train = zircons_data.loc[zircons_data["Set"] == "Training set", elements]
    x_data = preprocess_data(x_train, zircons_data[elements])
    x_data_df = pd.DataFrame(x_data, columns=elements)

    zircons_data["P_copy"] = zircons_data["P"].copy()

    data = pd.concat(
        [x_data_df, zircons_data[info_list + ["P_copy"]]],
        axis=1)

    data.to_csv(dataPath + fileName + "_processed.csv")
    train_set = data[(data["Set"] == "Training set")]
    train_set.reset_index(inplace=True, drop=True)
    test_set = data[(data["Set"] == "Testing set")]
    test_set.reset_index(inplace=True, drop=True)
    predict_set = data[(data["Set"] == "Prediction set")]
    predict_set.reset_index(inplace=True, drop=True)

    print("--------------------------------")

    X = np.array(data[elements])
    y = np.array(data["label"])
    X_train = train_set[elements]
    Y_train = train_set["label"]
    X_test = test_set[elements]
    Y_test = test_set["label"]
    X_predict = predict_set[elements]

    return X_train, Y_train, X_test, Y_test, X_predict


def model_train(method):
    X_train, Y_train, X_test, Y_test, X_predict = prepare_data()


    print("-------------------------" + method + "-------------------------------")
    if method == 'decisiontree':
        pred_test, pred, classify = decisiontree(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'my_GaussianNB':
        pred_test, pred, classify = my_GaussianNB(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'mlp':
        pred_test, pred, classify = mlp(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'Bernoulli':
        pred_test, pred, classify = Bernoulli(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'randomtree':
        pred_test, pred, classify = randomtree(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'logistic':
        pred_test, pred, classify = logistic(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'SVM':
        pred_test, pred, classify = SVM(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'KNN':
        pred_test, pred, classify = KNN(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'voting':
        pred_test, pred, classify = voting(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'bagging':
        pred_test, pred, classify = bagging(X_train, Y_train, X_test, Y_test, X_predict)
    if method == 'adaboost':
        pred_test, pred, classify = adaboost(X_train, Y_train, X_test, Y_test, X_predict)

    return pred_test, pred, classify


def model_predict(method):
    final_pred_test, pred, classify = model_train(method)
    final_pred_test = final_pred_test + 1
    pred = pred + 1
    final_pred_test = pd.DataFrame(final_pred_test, columns=['litho'])
    pred = pd.DataFrame(pred, columns=['litho'])
    if not os.path.exists(".//result//" + method + "//"):
        os.makedirs(".//result//" + method + "//")
    final_pred_test.to_excel(".//result//" + method + "//" + method + "_pred_test.xlsx")
    pred.to_excel(".//result//" + method + "//" + method + "_pred.xlsx")

    with open(".//result//" + method + "//" + method + "_classify.txt", 'w') as f:
        f.write(classify)
        f.write("\n")


if __name__ == '__main__':
    cv = "LeaveOneOut"
    make_print_to_file(path=outputPath, cv=cv)

    model_predict('decisiontree')
    model_predict('my_GaussianNB')

    model_predict('Bernoulli')
    model_predict('randomtree')
    model_predict('logistic')
    model_predict('SVM')
    model_predict('KNN')
    model_predict('mlp')
    model_predict('voting')

    model_predict('bagging')
    model_predict('adaboost')


