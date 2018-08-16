from numpy import *
import matplotlib.pyplot as plot

love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}


def file_to_matrix(filename):
    file = open(filename)
    line_count = len(file.readlines())
    ret = zeros((line_count, 3))

    class_label_vector = []
    label_list = []
    index = 0
    file = open(filename)
    for line in file.readlines():
        line = line.strip()
        list_of_line = line.split('\t')
        ret[index, :] = list_of_line[0:3]
        class_label_vector.append(list_of_line[-1])
        label_list.append(love_dictionary.get(list_of_line[-1]))
        index += 1
    return ret, class_label_vector, label_list


def auto_normal(dataset):
    min_value = dataset.min(0)
    max_value = dataset.max(0)
    ranges = max_value - min_value
    normal_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normal_dataset = dataset - tile(min_value, (m, 1))
    normal_dataset = normal_dataset / tile(ranges, (m, 1))
    return normal_dataset, ranges, min_value


data, vector, labels = file_to_matrix(
    '/private/var/root/Workspace/PTE/study/machine_learning_in_action/source_code/Ch02/datingTestSet.txt')
print(data)
print(labels)

# fig = plot.figure()
# ax = fig.add_subplot(111)
# ax.scatter(data[:, 0], data[:, 1], 15.0 * array(labels), 15.0 * array(labels))
# plot.show()
normMat, ranges, minVals = auto_normal(data)
