from src.data_loader import load_heart_disease


def main():
    X, y = load_heart_disease()
    print(X.head())
    print(y.head())


if __name__ == "__main__":
    main()