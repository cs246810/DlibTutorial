## Part II: Use dlib to detect especially picture that you want

![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.56390.cn%2F16874%2F2021%2F0424%2F20210424klo1619238495.jpg&refer=http%3A%2F%2Fimg.56390.cn&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1629484941&t=cde67c4a84165b3810344d0825be8c66)

上一篇说到，我们要检测飞机，那该怎么做？

<details>
<summary><mark>train_object_detector.py</mark></summary>

```python
#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
'''
这个示例程序展示了如何使用dlib来制作一个基于HOG的对象

的检测器，如人脸、行人和任何其他半刚性的

物体。 特别是，我们将通过以下步骤来训练那种滑动

窗口物体检测器，该检测器由Dalal和Triggs在2005年首次发表于

论文《用于人类检测的定向梯度直方图》。
'''
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import os
import sys
import glob

import dlib

'''
在这个例子中，我们将基于examples/faces目录下的小型人脸数据集来训练一个人脸检测器。

面孔数据集进行训练。 这意味着你需要提供

这个面孔文件夹的路径作为一个命令行参数，这样我们就可以知道

它在哪里。
'''
if len(sys.argv) != 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ../examples/faces")
    exit()
faces_folder = sys.argv[1]


'''
现在我们来进行训练

train_simple_object_detector()函数有很多选项。

有许多选项，所有这些选项都有合理的默认值。 接下来的

几行将介绍这些选项中的一些。
'''
options = dlib.simple_object_detector_training_options()

'''
由于人脸是左右对称的，我们可以告诉训练器训练一个

对称的检测器。 这有助于它从训练数据中获得最大的价值。

数据的最大价值。
'''
options.add_left_right_image_flips = True

'''
训练器是一种支持向量机，因此具有通常的

SVM C参数。 一般来说，较大的C值可以鼓励它更好地适应训练数据。

数据，但可能会导致过度拟合。 你必须找到最佳的C值

你必须根据经验找到最佳的C值，方法是检查训练好的检测器在你没有训练过的图像的测试集上的工作情况。

你没有训练过的图像的测试集上的效果。 不要只是把值设置为5。

尝试几个不同的C值，看看什么对你的数据最有效。
'''
options.C = 5

'''
告诉代码你的计算机有多少个CPU核心，以获得最快的训练。
'''
options.num_threads = 4
options.be_verbose = True


training_xml_path = os.path.join(faces_folder, "training.xml")
testing_xml_path = os.path.join(faces_folder, "testing.xml")

'''
这个函数做实际的训练。 它将把最终的检测器保存到

detector.svm。 它的输入是一个XML文件，其中列出了训练数据集中的图像

数据集中的图像，还包括人脸框的位置。 要创建你自己的

自己的XML文件，你可以使用imglab工具，该工具可以在

tools/imglab文件夹中。 它是一个简单的图形化工具，用于为图像中的物体贴上方框标签。

它是一个简单的图形工具，用于在图像中用方框标记对象。 要了解如何使用它，请阅读tools/imglab/README.txt

文件。 但在这个例子中，我们只是使用了随附的training.xml文件。
'''
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)



'''
现在我们有了一个人脸检测器，我们可以测试它。 第一个语句是在训练数据上测试

在训练数据上测试。 它将打印（精度、召回率，然后是

平均精度。
'''
print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))

'''
然而，为了了解它是否真的在没有过度拟合的情况下工作，我们需要

在它没有被训练过的图像上运行它。 下一行就是这样做的。 令人高兴的是，我们

看到物体检测器在测试图像上完美地工作。
'''
print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))





'''
现在，让我们像在正常应用中那样使用该探测器。 首先，我们

将从磁盘上加载它。
'''
detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
'''
这里英文解释 不想删，因为作者很高兴

我们可以看看我们学到的HOG过滤器。 它应该看起来像一张脸。 很整齐!
'''
win_det = dlib.image_window()
win_det.set_image(detector)

'''
现在，让我们在 faces 文件夹中的图像上运行检测器，并显示

结果。
'''
print("Showing detections on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

'''
接下来，假设你已经训练了多个探测器，并且你想把它们作为一个群体有效地运行。

有效地作为一个组来运行。 你可以按以下方式进行。
'''
detector1 = dlib.fhog_object_detector("detector.svm")
# In this example we load detector.svm again since it's the only one we have on
# hand. But in general it would be a different detector.
detector2 = dlib.fhog_object_detector("detector.svm")
# make a list of all the detectors you want to run.  Here we have 2, but you
# could have any number.
detectors = [detector1, detector2]
image = dlib.load_rgb_image(faces_folder + '/2008_002506.jpg')
[boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=1, adjust_threshold=0.0)
for i in range(len(boxes)):
    print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))

'''最后，请注意，你不必使用基于XML的输入到

train_simple_object_detector()。 如果你已经加载了你的训练

图片和物体的边界框，那么你就可以像下面这样调用它。

如下所示'''

# You just need to put your images into a list.
images = [dlib.load_rgb_image(faces_folder + '/2008_002506.jpg'),
          dlib.load_rgb_image(faces_folder + '/2009_004587.jpg')]
'''然后为每张图片制作一个矩形列表，给出盒子边缘的像素

框的边缘的位置'''
boxes_img1 = ([dlib.rectangle(left=329, top=78, right=437, bottom=186),
               dlib.rectangle(left=224, top=95, right=314, bottom=185),
               dlib.rectangle(left=125, top=65, right=214, bottom=155)])
boxes_img2 = ([dlib.rectangle(left=154, top=46, right=228, bottom=121),
               dlib.rectangle(left=266, top=280, right=328, bottom=342)])

'''
然后你把这些盒子的列表汇总成一个大的列表，然后调用

train_simple_object_detector()。
'''
boxes = [boxes_img1, boxes_img2]

detector2 = dlib.train_simple_object_detector(images, boxes, options)
# We could save this detector to disk by uncommenting the following.
#detector2.save('detector2.svm')

# Now let's look at its HOG filter!
win_det.set_image(detector2)
dlib.hit_enter_to_continue()

# Note that you don't have to use the XML based input to
# test_simple_object_detector().  If you have already loaded your training
# images and bounding boxes for the objects then you can call it as shown
# below.
print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(images, boxes, detector2)))
```

</details>

从这个程序能够知道，该如何训练和测试自己的模型。

上述代码笔者已经调试过。现在知道是要用一个叫`imglab`的工具给图片做标注，被标注的数据也就叫
训练集。然后再准备未被标注的数据，称为测试数据，再运行检测器加载训练数据然后对测试数据进行检测。

这里我想更进一步，测试的时候，笔者希望得到一个反馈，如果被测试数据是被训练集检测正确，则将这个
被测试数据也加入到训练集数据中，这样就可以让我们的训练集变得更加强大，从而在以后的检测中能更加
准确的检测。

从代码中的注释看到，人脸是对称的，这里通过代码中设置bool值来确定，所以，我们可以根据研究飞机是否
对称。首先第一步就是要确定飞机是否对称。然后，准备训练集的飞机图片，再用imglab对图片进行标注，
然后保存到训练集中，再使用测试数据测试训练集，测试完之后，反馈是否测试成功，测试成功则将被测试的
数据加入到训练集中。

这里要介绍一下`imglab` 这个软件的用法。这里有一篇博文有详细的介绍:https://blog.csdn.net/u010168781/article/details/91048497

由于我的图片已经下载到了`airplanes`这个文件夹中，所以，现在我开始执行民营，生成`mydataset.xml`
文件。然后再使用`imglab`对数据进行标注：

```shell
$ imglab -c ./airplanes/mydataset.xml ./airplanes
```
这里就会生成`mydataset.xml`文件，然后使用`imglab`对这个文件进行标注
```shell
$ imglab ./airplanes/mydataset.xml
```
这里会出现可视化窗体程序，然后根据上面的`imglab`的使用文章去标注，标注完之后记得点击`imglab`
的file菜单，单机保存。

这里是做好训练数据集，还要准备测试数据集，测试数据集也是经过标注的，所以必须再次下载一些飞机图片
这次我放到airplanes/test/下，然后再次运行

```shell
$ ./imglab -c ./airplanes/test/testdataset.xml ./airplanes/test
```

，这里会生成testdataset.xml文件，然后还要运行

```shell
$ ./imglab ./airplanes/test/testdataset.xml
```
来标注测试数据。


**前面两个步骤，已经准备好了训练数据集，和测试数据集**，现在，开始编写实际的程序去生成预测模型，
并且使用这个预测模型去检测未标注的的图片，并从图片中发现飞机的矩形。如果成功，意味着，我们对dlib
不仅仅是了解了人脸检测，还能将其应用到其它物体的检测。

这还要下载一点没有标注过的飞机图片放到airplanes/unknown文件夹下用来检测飞机在图片中的位置。

下面是完整的程序代码，记住，这并不是最最重要的，最重要的数据的标注。

<details>
<summary><mark>train_airplane_detector.py</mark></summary>

```python
import os
import sys
import glob

import dlib

if len(sys.argv) != 2:
    print('err')
    exit()
airplane_folder = sys.argv[1]


options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False

options.C = 5

options.num_threads = 2
options.be_verbose = True


training_xml_path = os.path.join(airplane_folder, "mydataset.xml")
testing_xml_path = os.path.join(airplane_folder, "test/testdataset.xml")

dlib.train_simple_object_detector(training_xml_path, "airplane_detector.svm", options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "airplane_detector.svm")))

print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "airplane_detector.svm")))

detector = dlib.simple_object_detector("airplane_detector.svm")

win_det = dlib.image_window()
win_det.set_image(detector)

print("Showing detections on the images in the airplanes/unknwon folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(os.path.join(airplane_folder, 'unknown'), "*.jpeg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
```

</details>

如果运行完毕没有框出飞机，则需要重新标注训练数据集和测试数据集。

这里笔者，再总结一遍这些步骤：

1. 准备训练数据集
2. 准备测试数据集
3. 标注训练数据集和测试数据集
4. 准备未知数据集
5. 运行程序，训练出模型数据集
6. 使用模型数据集对未知数据进行检测。
7. 如果没有检测出，则回到3，重新标注，再重新测试
8. 如果实在检测不出，再回到1，重新准备数据。

**可以看出，机器学习，如果只是调用库和api，这将会变成一种体力劳动，而非智力**。但是，渺小的众多程序员，由于
将大量的时间都放在了如何编写更好的代码和软件上，所以，从而这些比较重要的东西将会被忽略，例如，数学，算法。但是
，如果将大量的时间用在了数学和算法上，将会导致不会将数学和算法应用到实际的应用中。所以，从这里可以看出，程序员
和**算法工程师**都是有必要存在的角色，这两种角色能互补。

这里还能总结出，程序员和算法工程师的职责，程序员负责做流程，算法工程师负责提供算法库。这是最好的配合。
程序员能从成功的应用中获得价值和成就感，而算法工程师能了解到更深层次的原理而备受尊敬。但是说两者在世界中有没有
不可替代性？答案都是肯定的，都能被替代。所以，没有必要羡慕和感伤。一切都会变成历史，生命应该去寻找让自己内心
变得火热的事业去投入，这才是活着的意义。

而笔者，认为，酷是这个世界的神，他引领大众领悟活着的真谛。