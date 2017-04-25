import itertools

with open("parameter_learner_results.txt", 'r') as f:
    lines = [line for line in f]
    groups = [lines[i:i+6] for i in xrange(0, len(lines), 6)]
    for group in groups:
        print ",".join(map(lambda x: x.split("=")[1].rstrip(), group))
