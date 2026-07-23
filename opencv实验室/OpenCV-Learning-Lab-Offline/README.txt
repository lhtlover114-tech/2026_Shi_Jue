OpenCV 图片实验室 · 离线教学版
================================

一、如何启动
------------

1. 解压整个文件夹。
2. 确保下面两个文件始终放在同一目录：

   index.html
   opencv.js

3. 双击 index.html。
4. 等待页面右上角显示“OpenCV 已就绪 · 4.6.0”。
5. 点击“打开内置示例”，或选择你自己的图片。

整个网页不需要联网，也不需要安装 Node.js、Python 或前端框架。
图片只在浏览器内存中处理，不会上传。

二、包含的实验
--------------

1. 灰度化
   cv.cvtColor()：把 RGBA 彩色图转换为单通道灰度图。

2. 高斯模糊
   cv.GaussianBlur()：调整卷积核和 Sigma，观察细节与噪声如何被平滑。

3. 二值化
   cv.threshold()：普通、反向、Otsu 和反向 Otsu 四种模式。

4. Canny 边缘检测
   cv.Canny()：调整双阈值、预模糊、Sobel 孔径和 L2 梯度。

5. 形态学
   cv.erode()、cv.dilate()、cv.morphologyEx()：腐蚀、膨胀、开与闭运算。

6. 轮廓检测
   cv.findContours()：显示轮廓面积、周长、外接矩形和编号。

每次切换实验时，都从原始图片重新处理，不会把上一步结果作为输入。
右侧会同步显示当前参数对应的 JavaScript 代码、矩阵类型、处理耗时和资源释放数量。

三、常见问题
------------

1. 页面提示“OpenCV 加载失败”

   请确认 opencv.js 没有被删除或重命名，并且与 index.html 在同一个文件夹中。
   建议使用最新版桌面 Chrome、Edge 或 Firefox 打开，不要在聊天软件内置浏览器中运行。

2. 处理很慢或浏览器内存占用较高

   OpenCV 会按图片原始分辨率处理。手机拍摄的超大图片可能包含数千万像素，
   可以先缩小图片分辨率，或使用内置示例学习参数作用。

3. 为什么必须调用 delete()？

   cv.Mat、cv.MatVector 和结构元素存放在 WebAssembly 内存中。
   JavaScript 垃圾回收不能保证及时释放它们，所以页面使用 ResourceTracker 在每次处理后显式释放。

4. 为什么下载的是 PNG？

   Canvas 可以稳定导出 PNG，避免重复压缩造成学习时难以判断算法本身的影响。

四、文件说明
------------

index.html            页面、样式、教学内容和 OpenCV 调用代码
opencv.js             OpenCV.js 4.6.0 单文件运行时，WebAssembly 已内嵌
LICENSE-OpenCV.txt    OpenCV 使用的 Apache License 2.0
README.txt            本说明

OpenCV.js 文件 SHA-256：
fced3a671afb61d57b325d4bd380ef73296ecb24b147a7034e91523392e63605

版本说明：页面启动后会从 cv.getBuildInformation() 读取并显示运行时版本。
