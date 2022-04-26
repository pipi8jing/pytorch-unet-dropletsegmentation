import cv2
import numpy as np
import os


def dice_coeff(pred, target):
    smooth = 1.
    predict = np.array(pred)
    label = np.array(target)
    m1 = predict[1].flatten()  # Flatten
    m2 = label[1].flatten() # Flatten
    intersection = (m1 * m2).sum()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    precision = intersection / (m1.sum() + smooth)
    recall = intersection / (m2.sum() + smooth)
    f1 = 2 * precision * recall / (precision + recall)
    return dice


def main():
    path1 = ''
    path2 = ''
    path_list1 = os.listdir(path1)
    path_list2 = os.listdir(path2)
    path_list1.sort(key=lambda x: int(x.split('.')[0]))
    path_list2.sort(key=lambda x: int(x.split('.')[0]))
    lent = len(path_list1)
    total = 0
    for i in range(lent):
        predict = cv2.imread(os.path.join(path1, path_list1[i]), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(os.path.join(path2, path_list2[i]), cv2.IMREAD_GRAYSCALE)
        prediction = cv2.threshold(predict, 0, 1, cv2.THRESH_BINARY)
        label = cv2.threshold(lab, 0, 1, cv2.THRESH_BINARY)
        dice = dice_coeff(prediction, label)
        print(dice)
        total += dice
    print(total/lent)


if __name__ == '__main__':
    main()