strs = ["a", "Geeks", "for", "Geeks"]

if not strs:
    min = ""

min = ""
for i in strs:
    if len(i) <= len(min):
        min = i

strs.sort(key = len)
print(min)

