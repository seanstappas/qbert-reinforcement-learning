import csv


def save_to_csv(lst, filename):
    with open('report/csv/{}.csv'.format(filename), 'wb') as f:
        wr = csv.writer(f)
        wr.writerow(lst)


def read_from_csv(filename):
    with open('report/csv/{}.csv'.format(filename), 'rb') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        lst = list(reader)
    return lst[0]
