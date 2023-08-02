import matplotlib.pyplot as plt
from PIL import Image

# 設定Matplotlib的後端為Agg（非互動式）
plt.switch_backend('Agg')

# 創建一個包含多張圖片的列表
images = []

# 使用Matplotlib繪製圖片並將每張圖片加入到images列表中
for i in range(10):
    plt.plot([x for x in range(i+1)], [x**2 for x in range(i+1)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Plot {i+1}')
    plt.grid(True)
    plt.savefig(f'plot_{i+1}.png')
    plt.clf()
    images.append(Image.open(f'plot_{i+1}.png'))

# 將圖片列表images保存為GIF動畫
images[0].save('animated_plot.gif', save_all=True, append_images=images[1:], duration=200, loop=0)

# 清除暫存的圖片文件
import os
for i in range(10):
    os.remove(f'plot_{i+1}.png')

print("GIF動畫已成功創建！")
