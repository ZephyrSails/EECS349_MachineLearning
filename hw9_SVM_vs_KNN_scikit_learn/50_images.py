import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist

i = 100
images, labels = load_mnist(digits=range(0, 10), path='.')
print '# of image : ', len(images)
print len(images[1])
# plt.imshow(images[i], cmap = 'gray')
# plt.title('Handwritten image of the digit ' + str(labels[i]))
# plt.show()

idx = np.random.randint(1, len(images) - 1, size=5)
for count, i in enumerate(idx): ## will open an empty extra figure :(
    print i
    plt.imsave('image' + str(count) + 'png', images[i], cmap = 'gray')

for i in range(1, 10):
	images, labels = load_mnist(digits=[i], path='.')
	print i, ' : ', len(images)