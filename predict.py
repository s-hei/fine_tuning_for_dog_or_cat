# coding: utf-8
import sys
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import math

# if len(sys.argv) != 2:
#     print("usage: python predict.py [filename]")
#     sys.exit(1)

# filename = sys.argv[1]
# print('input:', filename)

filenames = list(map(lambda i: '%s.jpg' % i, range(1, 10)))

img_height, img_width = 224,224

model = load_model('finetuning_model.h5')

# 画像を読み込んで4次元テンソルへ変換
img = image.load_img(filenames, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
# これを忘れると結果がおかしくなるので注意
x = x / 255.0

# print(x)
# print(x.shape)

# クラスを予測
# 入力は1枚の画像なので[0]のみ
pred = model.predict(x)
print(['cat', 'dog'])
print(list(map(lambda x: "{:.0%}".format(x), list(pred))))
