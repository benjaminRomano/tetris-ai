import itertools

with open("results_bak.txt", 'r') as f:
    lines = [line for line in f]
    groups = [lines[i:i+4] for i in xrange(0, len(lines), 4)]
    for group in groups:
        print ",".join(map(lambda x: x.split("=")[1].rstrip(), group))
