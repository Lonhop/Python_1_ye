lst = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

d = {}
for i in lst:
    d[i] = d.get(i, 0) + 1

c = dict(sorted(d.items(), key=lambda x: (-x[1], x[0])))
print(max(c.values()))

# your code here
# expected: 5