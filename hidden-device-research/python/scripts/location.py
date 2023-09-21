#!/usr/bin/env python3
import math
from calculate_distance_error import distance_function

def pythagorean(a, b):
    return math.sqrt(a**2 + b**2)

def get_location(L, W, pred):
    cell_width = W/3
    cell_height = L/7
    if pred in [0, 3]:
        card_dir = "NE"
    elif pred in [1, 4, 7]:
        card_dir = "E"
    elif pred in [2, 5]:
        card_dir = "NW"
    elif pred in [6, 9, 12]:
        card_dir = "N"
    elif pred in [8, 11, 14]:
        card_dir = "S"
    elif pred in [15, 18]:
        card_dir = "SE"
    elif pred in [13, 16, 19]:
        card_dir = "W"
    else:
        card_dir = "SW"
    #print(card_dir)
    #print(cell_width)
    #print(cell_height)

    diff = abs(10 - pred)
    row = math.ceil(diff/3)
    #print(diff)
    #print(row)
    if (diff) %3 == 0:
        y = 0
    elif (diff)%3 == 1:
        y = -1
    else:
        y = 1

    if diff %3 ==1:
        row -= 1

    if pred > 11:
        x = -row
        y= -y
    elif pred in [9, 10, 11]:
        x = 0
        if pred == 11:
            y = -y
    else:
        x = row

    print("%d\t(%d, %d) %s"%(pred, x, y, card_dir))
    c = pythagorean(cell_width, cell_height)
    d = distance_function(0, 0, x, y)
    print("%f to the %s"%((c*d*3)/4, card_dir))


if __name__ == '__main__':
    for i in range(21):
        get_location(6.7, 3.6, i)
