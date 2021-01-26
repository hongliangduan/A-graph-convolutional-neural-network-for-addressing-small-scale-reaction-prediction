#python scripts/coverage.py
candidate_bonds_file = open('../core_wln_global/model-300-3-direct/test.cbond')
gold = open('../data/test.txt.proc')


k_values = [6, 8, 10, 12, 14, 16, 18, 20]
topks = [0 for k in k_values]
total = 0
for line in candidate_bonds_file:
    total += 1
    candidate_bonds = []
    for v in line.split():
        x,y,t = v.split('-')
        candidate_bonds.append((int(x), int(y), float(t)))
    
    line = gold.readline()
    tmp = line.split()[1]
    gold_bonds = []
    for v in tmp.split(';'):
        x,y,t = v.split('-')
        x,y = int(x),int(y)
        x,y = min(x,y), max(x,y)
        gold_bonds.append((x, y ,float(t)))

    for i in range(len(k_values)):
        if set(gold_bonds) <= set(candidate_bonds[:k_values[i]]):
            topks[i] += 1.0

print(k_values)
print [topk / total for topk in topks]
