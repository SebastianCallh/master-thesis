import time as t

def time(f, msg):
    t0 = t.time()
    res = f()
    tn = t.time()
    print(msg, '- Time elapsed:', tn - t0, msg, )
    return res