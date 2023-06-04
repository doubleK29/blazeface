import cv2
import matplotlib.pyplot as plt
import numpy as np
test_path = "data/combine/test/test.txt"
check_path = "data/combine/test/infer.txt"

f = open(test_path).readlines()[:1000]
with open(check_path, "w") as file:
    for i in f:
        file.write(i)