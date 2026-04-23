from collections import Counter
import random


def oversample(X, y):
    counter = Counter(y)

    max_count = max(counter.values())

    X_new, y_new = [], []

    data = list(zip(X, y))

    for label in counter:
        samples = [d for d in data if d[1] == label]

        if len(samples) < max_count:
            samples = samples + random.choices(samples, k=max_count-len(samples))

        X_new.extend([s[0] for s in samples])
        y_new.extend([s[1] for s in samples])

    return X_new, y_new


def undersample(X, y):
    from collections import Counter
    import random

    counter = Counter(y)
    min_count = min(counter.values())

    data = list(zip(X, y))
    X_new, y_new = [], []

    for label in counter:
        samples = [d for d in data if d[1] == label]

        if len(samples) > min_count:
            k = int((len(samples)+min_count)/2)
            samples = random.sample(samples, k=k)

        X_new.extend([s[0] for s in samples])
        y_new.extend([s[1] for s in samples])

    return X_new, y_new
