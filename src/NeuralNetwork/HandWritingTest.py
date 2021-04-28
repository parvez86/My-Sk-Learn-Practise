import joblib
import numpy as np
from PIL import Image
model = joblib.load('HandDigitRecognizer.joblib')

image_list = ['one.png', 'two.png', 'three.png', 'four.png', 'five.png', 'nine.png', 'three_fourty_six.png', 'nine_seventy.png']

# image_data = list()
# for i in range(len(image_list)):
#     img = Image.open(image_list[i])
#     data = list(img.getdata())
#     data = np.reshape(data, (-1, 28 * 28))
#     # print(data)
#     for j in range(len(data)):
#         data[j] = 255 - data[j]
#     data = np.array(list(data)) / 256
#     image_data.append(np.array(data))

img = Image.open('seven.png')
data = list(img.getdata())
print(data)
data = np.reshape(data, (-1, 28*28))
# data = []
for j in range(len(data)):
    data[j] = 255 - data[j]

print(data)
image_data = np.array(list(data))/256
print(image_data)
print(model.predict(image_data))
# # print(model.predict(image_data[1]))