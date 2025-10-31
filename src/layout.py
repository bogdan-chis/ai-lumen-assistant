def reading_order(items):
    # simple top-left ordering
    return sorted(items, key=lambda t: (t["box"][1], t["box"][0]))

def summarize(items, max_len=200):
    s = " ".join(t["text"] for t in reading_order(items))
    return s[:max_len]