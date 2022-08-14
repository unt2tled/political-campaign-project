import csv
import re
import matplotlib.pyplot as plt

def count_avg_questions(path):
    x = []
    y = ([], [], [])
    with open(path, "r") as tags_file:
        csv_reader = csv.reader(tags_file)
        next(csv_reader)
        counter = [0, 0, 0]
        counter_total = [0, 0, 0]
        for i, row in enumerate(csv_reader):
            x.append(i)
            y[0].append(0)
            y[1].append(0)
            y[2].append(0)
            text = row[1]
            counter_total[int(row[2])] += 1
            counter[int(row[2])] += len(re.findall("\?", text))
            y[int(row[2])][-1] = len(re.findall("\?", text))
    plt.plot(x, y[0])
    #plt.plot(x, y[1])
    plt.plot(x, y[2])
    print(y[2])
    plt.show()
    return [(counter[i]/counter_total[i]) for i in range(len(counter))]

def count_pronouns(path):
    with open(path, "r") as tags_file:
        csv_reader = csv.reader(tags_file)
        next(csv_reader)
        counter = [0, 0, 0]
        counter_total = [0, 0, 0]
        for row in csv_reader:
            text = row[1]
            counter_total[int(row[2])] += 1
            #pattern = "(he)|(she)|(her)|(his)|(them)|(they)|(their)"
            pattern = "(Obama)"
            counter[int(row[2])] += len(re.findall(pattern, text, re.IGNORECASE))
    return [(counter[i]/counter_total[i]) for i in range(len(counter))]

print(count_avg_questions("tags.csv"))
print(count_pronouns("tags.csv"))
