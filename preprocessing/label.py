def get_label(score: int) -> str:
    if score < 3:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"