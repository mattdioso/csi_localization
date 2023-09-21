#!/usr/bin/env python3
import math
import numpy as np

def calculate_coordinates(cell):
    row = math.floor(cell/3)
    if cell % 3 == 0:
        col = 3
    else:
        col = cell - (row*3)

    if cell % 3 != 0:
        row += 1
    return row, col

def distance_function(x1, y1, x2, y2):
    d = math.sqrt((x2-x1)**2 + (y2 - y1)**2)
    return d

def calculate_mean_error(predictions, y_test):
    total_distance = 0
    num_wrong = 0
    for i in range(0, len(predictions)):
        if (np.argmax(predictions[i]) != y_test[i]):
            print(str(np.argmax(predictions[i])) + "\t" + str(y_test[i]))
            predictions_x, predictions_y = calculate_coordinates(np.argmax(predictions[i]))
            print("predictions x and y: (%d, %d)"%(predictions_x, predictions_y))
            answer_x, answer_y = calculate_coordinates(y_test[i])
            print("answer x and y: (%d, %d)"%(answer_x, answer_y))
            total_distance += abs(distance_function(predictions_x, predictions_y, answer_x, answer_y))
            num_wrong += 1
    print("num_wrong: " + str(num_wrong))
    return total_distance / num_wrong

def get_coor(cell):
    diff = abs(10 - cell)
    row = math.ceil(diff/3)

    if (diff) % 3 == 0:
        y = 0
    elif (diff) % 3 == 1:
        y = -1
    else:
        y = 1

    if (diff) % 3 == 1:
        row -= 1
    if cell > 11:
        x = -row
        y = -y

    elif cell in [9, 10 , 11]:
        x=0
        if cell == 11:
            y = -y
    else:
        x = row

    return x, y

def calculate_distance_error(prediction, Y):
#    pred_x, pred_y = calculate_coordinates(prediction)
#    ans_x, ans_y = calculate_coordinates(Y)
    pred_x, pred_y = get_coor(prediction)
    ans_x, ans_y = get_coor(Y)
    return distance_function(pred_x, pred_y, ans_x, ans_y)
    
