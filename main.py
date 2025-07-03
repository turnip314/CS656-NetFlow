from classifier import load_data, classify, test
TEST_RATIOS = [0.2, 0.3, 0.33, 0.4, 0.5]

MODELS = ["svm", "sgd", "gnb", "dt", "rf", "ab", "knn"] #, 
SIZES = [700000]

METRICS = ["acc", "f2", "auc"]

def run_all_tests(size=700000, encoding="le"):
    f = open(f"output_{encoding}.txt", "w+")
    all_datasets = []
    for ratio in TEST_RATIOS:
        description = f"{size} rows with a {round(100*(1-ratio))}/{round(100*ratio)} train/test split and {encoding} encoding."
        all_datasets.append((description, load_data(frac=ratio, size=size, encoding=encoding)))
        print(f"Loaded dataset: {description}")

    print("All datasets loaded.")
    print()
    for model in MODELS:
        print(f"Running model {model}.")
        f.write("="*80+"\n")
        f.write(f"Model {model}.\n")
        for description, (X_train, X_test, y_train, y_test) in all_datasets:
            cf, ttime = classify(model, X_train, y_train)
            f.write(f"Dataset: {description}\n")
            f.write(f"Training time: {round(ttime, 2)}\n")
            for metric in METRICS:
                result = test(cf, X_test, y_test, metric=metric)
                f.write(f"{metric}: {result}\n")
            f.write("\n")
            print(f"Completed tests for dataset {description}")
        f.write("\n")
        print(f"Completed tests for model {model}.")
        print()
    print("All done!")

    f.close()


if __name__ == "__main__":
    run_all_tests()
