from ucimlrepo import fetch_ucirepo


def load_heart_disease():
    heart = fetch_ucirepo(id=45)

    X = heart.data.features
    y = heart.data.targets

    return X, y