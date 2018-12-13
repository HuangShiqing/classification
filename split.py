import os
import random


def split_data(data_dir, train_percent=0.8):
    train_list = []
    valid_list = []

    dirs = os.listdir(data_dir)
    os.chdir(data_dir)
    class_txt = open('class.txt', 'w')
    for i, dir in enumerate(dirs):
        if dir in ['train.txt', 'valid.txt', 'class.txt']:
            continue
        temp_list = []
        files = os.listdir(dir)
        for file in files:
            if file[0] == '.':
                continue
            temp_list.append(dir + '/' + file + '!' + str(i))
        random.shuffle(temp_list)
        threshold = int(len(temp_list) * train_percent)
        train_list += temp_list[0:threshold]
        valid_list += temp_list[threshold::]

        class_txt.write(dir + ' ' + str(i) + '\n')
    random.shuffle(train_list)
    random.shuffle(valid_list)
    class_txt.close()

    with open('./train.txt', 'w') as f:
        for path in train_list:
            f.write(path.strip().split('!')[0] + ' ' + path.strip().split('!')[1] + '\n')
    with open('./valid.txt', 'w') as f:
        for path in valid_list:
            f.write(path.strip().split('!')[0] + ' ' + path.strip().split('!')[1] + '\n')

    print('Finished writing ')


def read_data(dir):
    x_train, y_train, x_valid, y_valid = [], [], [], []
    with open(dir + 'train.txt', 'r') as f:
        for line in f.readlines():
            x_train.append(line.strip().split(' ')[0])
            y_train.append(line.strip().split(' ')[1])
    with open(dir + 'valid.txt', 'r') as f:
        for line in f.readlines():
            x_valid.append(line.strip().split(' ')[0])
            y_valid.append(line.strip().split(' ')[1])
    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    split_data('D:/DeepLearning/data2/birds/images/')
    x_train, y_train, x_valid, y_valid = read_data('D:/DeepLearning/data2/birds/images/')
    exit()
