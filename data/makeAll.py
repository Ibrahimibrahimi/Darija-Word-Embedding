

# to combine all data to one file "all.txt"

import os


# all datasets ".txt" except "all.txt"
data = [i for i in os.listdir("data") if i.endswith(".txt") and i != "all.txt"]


DATA = []
for file in data:
    with open(f"data/{file}", "r", encoding="utf-8") as f:
        for line in f.readlines():
            DATA.append(line)
print(len(DATA))


# re create
with open("data/all.txt","w",encoding="utf-8") as f :
    f.writelines(DATA)