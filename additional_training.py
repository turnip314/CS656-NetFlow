from classifier import load_data, classify, test
TEST_RATIOS = [0.2, 0.3, 0.33, 0.4, 0.5]

MODELS = ["svm", "sgd", "gnb", "dt", "rf", "ab",]# "knn"]
SIZES = [700000]

METRICS = ["acc", "f2", "auc"]

def run_all_tests(size=700000, model="rf", encoding="le"):
    ratio = 0.4
    description = f"{size} rows with a {round(100*(1-ratio))}/{round(100*ratio)} train/test split and {encoding} encoding."
    X_train, X_test, y_train, y_test = load_data(frac=ratio, size=size, encoding=encoding)
    print(f"Loaded dataset: {description}")

    cf, ttime = classify(model, X_train, y_train)
    for metric in METRICS:
        result = test(cf, X_test, y_test, metric=metric)
    print("All done!")

if __name__ == "__main__":
    run_all_tests(encoding="oh")
