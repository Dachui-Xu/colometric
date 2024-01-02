import cv2
from QCandyUi import CandyWindow
from colormetric import Ui_mainWindow
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QStringListModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

APP_NAME = "Colormetric V1.1.0"
class ColorUi(QMainWindow, Ui_mainWindow):
    imgNamepath =""
    grayIn = False
    def __init__(self):
        super(ColorUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle(APP_NAME)

        global grayIn
        grayIn = False
        pd.set_option('display.unicode.east_asian_width',True)
        columns = ['浓度', '数值']
        self.df = pd.DataFrame(columns=columns)
        self.init()


    def init(self):
        global imgNamepath
        imgNamepath = ""
        pic = cv_imread('image/img.png')
        self.showImageLeft(pic, 800, 800)
        #图片选择按键
        self.bt_choose.clicked.connect(self.imageChoose)
        self.bt_chooseimages.clicked.connect(self.openimage)
        #图片处理部分
        self.bt_HSV.clicked.connect(self.hsvProce)
        self.bt_RGB.clicked.connect(self.rgbProce)
        self.bt_gray.clicked.connect(self.grayProc)
        self.bt_colormetric.clicked.connect(self.colorProc)
        #数据处理部分
        self.bt_record.clicked.connect(self.recordData)
        self.bt_plot.clicked.connect(self.drawResult)
        self.bt_save.clicked.connect(self.saveResultToExcel)
        self.bt_import.clicked.connect(self.importData)
        self.bt_evaluate.clicked.connect(self.predicateResults)
        #数据清洗
        self.bt_delect.clicked.connect(self.delectLine)
        # self.br_merge.clicked.connect(self.mergeCon)
        self.bt_claer.clicked.connect(self.clearDf)
        #菜单栏
        self.actionimage_f.triggered.connect(self.openimage)
        self.actionimage_flie.triggered.connect(self.imageChoose)
        self.actionimport_excel.triggered.connect(self.importData)
        self.actionsave_as_excel.triggered.connect(self.saveResultToExcel)
        self.actionrecord.triggered.connect(self.recordData)
        self.actionregression_line.triggered.connect(self.drawResult)
        self.actionconcentration.triggered.connect(self.predicateResults)
        self.actionHSV.triggered.connect(self.hsvProce)
        self.actionRGB.triggered.connect(self.rgbProce)
        self.actionGARy.triggered.connect(self.grayProc)
        self.actionColormetric.triggered.connect(self.colorProc)
        self.actionhsv_table.triggered.connect(self.hsv_table)
    '''打开图片'''
    #打开多个图
    def openimage(self):
        self.path, imgType = QtWidgets.QFileDialog.getOpenFileNames(self, "多文件选择", " ", "所有文件 (*);;图片文件 (*.jpg);;图片文件 (*.png)")
        slm = QStringListModel()
        slm.setStringList(self.path)
        print(self.path)
        self.listView.setModel(slm)
        self.listView.clicked.connect(self.onClickedListView)
    def onClickedListView(self, qModelIndex):
        global imgNamepath
        # QMessageBox.information(self, "QListView", "您选择了："+self.path[qModelIndex.row()])
        imgNamepath = self.path[qModelIndex.row()]
        self.lineEdit.setText(imgNamepath)
        self.loadImage(imgNamepath)
        #print(self.list[item.row()])
    #打开单一图
    def imageChoose(self):
        global imgNamepath
        #dir.setDirectory("./")
        imgNamepath, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;*.tif;*.png;;All Files(*)")
        self.print.clear()
        if imgNamepath == "":
            return 0
        #image = QtGui.QPixmap(imgNamepath).scaled(self.label_3.width(), self.label_3.height())
        #dir.setNameFilter('*.jpg'|'*.png')

            #self.lineEdit.setText(dir.selectFiles()[0])
        self.lineEdit.setText(imgNamepath)
        self.loadImage(imgNamepath)


    def loadImage(self,path):
        image = cv_imread(path)
        self.showImageLeft(image, 800, 800)

    ''' 以下是图片处理的函数，左侧展示的是原始图像，右侧是处理图，opencv的图RGB是BGR所红蓝颜色会有不同
        hsv、rgb、灰度、y=sqrt(r^2+g^2+b^2)'''
    def hsvProce(self):
        self.print.clear()
        self.print_ave.clear()
        z = []
        global imgNamepath
        if(imgNamepath==""):
            image = cv_imread('image/img.png')
        else:image = cv_imread(imgNamepath)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        def getpos(event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                #print(HSV[y, x][0]," ",HSV[y, x][1]," ",HSV[y,x][2])
                self.printfln(str(HSV[y, x][0]).rjust(3,'0'))
                self.printf("    ")
                self.printf(str(HSV[y, x][1]).rjust(3,'0'))
                self.printf("   ")
                self.printf(str(HSV[y, x][2]).rjust(3,'0'))
                z.append(HSV[y, x][1])
            if event == cv2.EVENT_RBUTTONDOWN:
                per_image_Hmean = []
                per_image_Smean = []
                per_image_Vmean = []
                cv2.namedWindow("ROI", 2)
                l = cv2.selectROI('ROI', HSV, False, False)
                imCrop = HSV[int(l[1]):int(l[1] + l[3]), int(l[0]):int(l[0] + l[2])]
                cv2.namedWindow("cut_image", 2)
                cv2.resizeWindow("cut_image", 300, 300)
                cv2.imshow("cut_image", imCrop)
                per_image_Hmean.append(np.mean(imCrop[:, :, 0]))
                per_image_Smean.append(np.mean(imCrop[:, :, 1]))
                per_image_Vmean.append(np.mean(imCrop[:, :, 2]))
                h = np.mean(per_image_Hmean)
                s = np.mean(per_image_Smean)
                v = np.mean(per_image_Vmean)
                self.printfln(str(int(h)).rjust(3,'0'))
                self.printf("    ")
                self.printf(str(int(s)).rjust(3,'0'))
                self.printf("   ")
                self.printf(str(int(v)).rjust(3,'0'))
                # self.printf("  ")
                # self.printf(str(round(s, 3)))
                z.append(s)
        self.printfln("颜色 饱和度 亮度 (均值为饱和度)")
        #print("颜色 饱和度 亮度")
        self.showImageRight(HSV,800,800)
        # 创建一个名字叫做 image_win的窗口
        cv_imageWin(image, getpos, "image_win")
        self.average(z)

    def rgbProce(self):
        self.print.clear()
        self.print_ave.clear()
        z = []
        global imgNamepath
        if (imgNamepath == ""):
            image = cv_imread('image/img.png')
        else:
            image = cv_imread(imgNamepath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        def getpos(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                #print(HSV[y, x][0]," ",HSV[y, x][1]," ",HSV[y,x][2])
                self.printfln(str(image[y, x][0]).rjust(3,'0'))
                self.printf("  ")
                self.printf(str(image[y, x][1]).rjust(3,'0'))
                self.printf("  ")
                self.printf(str(image[y, x][2]).rjust(3,'0'))
                q = (int(image[y, x][0])+int(image[y, x][1])+int(image[y, x][2]))/3
                self.printf("  ")
                self.printf(str(round(q,3)))
                z.append(q)
            if event == cv2.EVENT_RBUTTONDOWN:
                per_image_Rmean = []
                per_image_Gmean = []
                per_image_Bmean = []
                cv2.namedWindow("ROI", 2)
                l = cv2.selectROI('ROI', rgb, False, False)
                imCrop = image[int(l[1]):int(l[1] + l[3]), int(l[0]):int(l[0] + l[2])]
                cv2.namedWindow("cut_image", 2)
                cv2.resizeWindow("cut_image", 300, 300)
                cv2.imshow("cut_image", imCrop)
                per_image_Rmean.append(np.mean(imCrop[:, :, 0]))
                per_image_Gmean.append(np.mean(imCrop[:, :, 1]))
                per_image_Bmean.append(np.mean(imCrop[:, :, 2]))
                r = np.mean(per_image_Rmean)
                g = np.mean(per_image_Gmean)
                b = np.mean(per_image_Bmean)
                self.printfln(str(int(r)).rjust(3,'0'))
                self.printf("  ")
                self.printf(str(int(g)).rjust(3,'0'))
                self.printf("  ")
                self.printf(str(int(b)).rjust(3,'0'))
                q=(int(r)+int(g)+int(b))/3
                self.printf("  ")
                self.printf(str(round(q, 3)))
                z.append(q)


        self.printfln("红   绿   蓝  均值")
        self.showImageRight(rgb,800,800)
        # 创建一个名字叫做 image_win的窗口
        cv_imageWin(rgb, getpos, "image_win")
        self.average(z)

    def grayProc(self):

        self.print.clear()
        self.print_ave.clear()
        global imgNamepath
        global grayIn
        grayIn = True
        z = []
        if (imgNamepath == ""):
            image = cv_imread('image/img.png')
        else:
            image = cv_imread(imgNamepath)
        Gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        def getpos(event, x, y, flags, param):
             if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                self.printfln(str(Gray[y,x]))
                z.append(Gray[y,x])
             if event == cv2.EVENT_RBUTTONDOWN:
                per_image_Rmean = []
                per_image_Gmean = []
                per_image_Bmean = []
                cv2.namedWindow("ROI", 2)
                l = cv2.selectROI('ROI', rgb, False, False)
                imCrop = rgb[int(l[1]):int(l[1] + l[3]), int(l[0]):int(l[0] + l[2])]
                cv2.namedWindow("cut_image", 2)
                cv2.resizeWindow("cut_image", 300, 300)
                cv2.imshow("cut_image", imCrop)
                per_image_Rmean.append(np.mean(imCrop[:, :,0]))
                per_image_Gmean.append(np.mean(imCrop[:, :, 1]))
                per_image_Bmean.append(np.mean(imCrop[:, :, 2]))
                r = np.mean(per_image_Rmean)
                g = np.mean(per_image_Gmean)
                b = np.mean(per_image_Bmean)
                gary = int(r)*0.299 + int(g)*0.587 + int(b)*0.114
                self.printfln(str(round(gary,3)))
                z.append(gary)
        self.showImageRight(image, 800, 800)
        self.printfln("灰度值")
        # # 创建一个名字叫做 image_win的窗口
        cv_imageWin(rgb, getpos, "image_win")
        # 均值计算
        self.average(z)

    def colorProc(self):
        self.print.clear()
        self.print_ave.clear()
        global imgNamepath
        z = []

        if (imgNamepath == ""):
            image = cv_imread('image/img.png')
        else:
            image = cv_imread(imgNamepath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        def getpos(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                b = int(rgb[y,x][0])
                g = int(rgb[y,x][1])
                r = int(rgb[y,x][2])
                q = np.sqrt(r*r+g*g+b*b)
                z.append(q)
                self.printfln(str(q))
            if event == cv2.EVENT_RBUTTONDOWN:
                per_image_Rmean = []
                per_image_Gmean = []
                per_image_Bmean = []
                l = cv2.selectROI('ROI', rgb, False, False)
                # x,y,w,h=r
                #print(r)
                imCrop = rgb[int(l[1]):int(l[1] + l[3]), int(l[0]):int(l[0] + l[2])]
                cv2.namedWindow("cut_image", 2)
                cv2.resizeWindow("cut_image", 300, 300)
                cv2.imshow("cut_image", imCrop)
                per_image_Bmean.append(np.mean(imCrop[:, :, 0]))
                per_image_Gmean.append(np.mean(imCrop[:, :, 1]))
                per_image_Rmean.append(np.mean(imCrop[:, :, 2]))
                r = np.mean(per_image_Rmean)
                g = np.mean(per_image_Gmean)
                b = np.mean(per_image_Bmean)
                q = np.sqrt(r * r + g * g + b * b)
                z.append(q)
                self.printfln(str(q))

        self.printfln("sqrt(r^2+g^2+b^2)")
        self.showImageRight(rgb, 800, 800)
        # 创建一个名字叫做 image_win的窗口
        cv_imageWin(rgb, getpos, "image_win")
        #均值计算
        self.average(z)

    '''数组均值计算，显示在均值的文本浏览器中'''
    def average(self,z):
        sum = 0
        # 均值计算
        if len(z) == 0:
            self.printfln("未获得数据")
            self.print_ave.insertPlainText(str(0))  # 显示均值，并保留两位小数
            self.cursot = self.print_ave.textCursor()
            self.print_ave.moveCursor(self.cursot.End)
            QtWidgets.QApplication.processEvents()
        else:
            for a in z:
                sum += a
            ave = sum / len(z)
            self.print_ave.insertPlainText(str(round(ave, 3)))  # 显示均值，并保留两位小数
            self.cursot = self.print_ave.textCursor()
            self.print_ave.moveCursor(self.cursot.End)
            QtWidgets.QApplication.processEvents()

    '''左右图片显示方法 ，和文本浏览器显示'''
    ##图片显示在左侧
    def showImageLeft(self,img,sizeX,siezeY):
        global grayIn
        img = cv2.resize(img, (sizeX,siezeY))
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        if(grayIn==True):frame = QImage(img, x, y, QImage.Format_Grayscale16)
        else:frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.scene.clearSelection()
        self.item.setSelected(True)
        self.iv_original.setScene(self.scene)

    ##图片显示在右侧
    def showImageRight(self,img,sizeX,siezeY):
        global grayIn
        img = cv2.resize(img, (sizeX,siezeY))
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        if(grayIn):frame = QImage(img, x, y, QImage.Format_Grayscale16)
        else:frame = QImage(img, x, y, QImage.Format_RGB888)
        grayIn = False
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.scene.clearSelection()
        self.item.setSelected(True)
        self.iv_proce.setScene(self.scene)
    '''文本浏览器打印字符'''
    #左侧文本浏览器
    def printfln(self, mes):
            self.print.append(mes)  # 在指定的区域显示提示信息
            self.cursot = self.print.textCursor()
            self.print.moveCursor(self.cursot.End)
            QtWidgets.QApplication.processEvents()

    def printf(self, mes):
        self.print.insertPlainText(mes)  # 在指定的区域显示提示信息
        self.cursot = self.print.textCursor()
        self.print.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents()
    #右侧文本浏览器
    def printfResultln(self,mes):
        self.result.append(mes)  # 在指定的区域显示提示信息
        self.cursot = self.result.textCursor()
        self.result.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents()
    def printfResult(self,mes):
        self.result.insertPlainText(mes)  # 在指定的区域显示提示信息
        self.cursot = self.result.textCursor()
        self.result.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents()
    #结果文本浏览器
    def printEvaluate(self,result):
        self.evaluate.clear()
        self.evaluate.append(str(round(result[0][0], 3)))  # 在指定的区域显示提示信息
        self.cursot = self.evaluate.textCursor()
        self.evaluate.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents()

    '''数据处理（ui右下部分）'''
    def recordData(self):
        concentration = self.le_concentration.text()
        average= self.print_ave.toPlainText()
        if concentration=="" or average=="":
            QMessageBox.critical(None,"数据不完整","(@_@)请您检查浓度是否输入，均值是否计算",QMessageBox.Ok)
            return
        #插入数据
        df_insert = pd.DataFrame({'浓度':[float(concentration)],'数值':[float(average)]})
        self.df = pd.concat([self.df,df_insert],ignore_index=True)#合并df
        self.df = self.df.sort_values(by='浓度',ascending=True)
        self.df.reset_index(drop=True, inplace=True)#更新index
        #print(self.df)
        self.result.clear()
        self.printfResultln(str(self.df))

    #回归曲线绘制
    def drawResult(self):
        if str(pd.isnull(self.df['浓度']))=="Series([], Name: 浓度, dtype: bool)":
            QMessageBox.critical(None, "无数据！！", "您还没获得测量数据呢(●'◡'●)", QMessageBox.Ok)
            return
        plt.scatter(self.df['浓度'],self.df['数值'], label='origin', color='red')
        plt.show()
        #线性回归
        clf = linear_model.LinearRegression()
        x1 = pd.DataFrame(self.df['浓度'])
        y1 = pd.DataFrame(self.df['数值'])
        clf.fit(x1,y1) #拟合模型
        R2 = clf.score(x1,y1,sample_weight=None)
        k = clf.coef_ #回归系数
        b = clf.intercept_ #截距
        #线性回归图
        x = np.linspace(0, x1.max(), num=100)
        y = clf.predict(x.reshape(-1, 1))
        plt.plot(x, y, label='LinearRegression', linewidth=2)
        plt.legend(['origin', 'LinearRegression'])
        #回归曲线函数显示
        self.printfResultln('回归曲线：')
        self.printfResultln('y = ')
        self.printfResult(str(round(float(k),3)))
        self.printfResult(' x + ')
        self.printfResult(str(round(float(b),3)))
        self.printfResultln('R^2 = ')
        self.printfResult(str(round(float(R2),3)))

    #保存文件
    def saveResultToExcel(self):
        if str(pd.isnull(self.df['浓度']))=="Series([], Name: 浓度, dtype: bool)":
            select = QMessageBox.information(None,"保存","没有获得任何数据确定要保存吗？",QMessageBox.Yes|QMessageBox.No)
            if select == QMessageBox.No:
                return
        fileName_choose, filetype = QFileDialog.getSaveFileName(self,
                                                                "文件保存",
                                                                "",
                                                                "*.xlsx;;*.txt;;All Files (*);;")

        if filetype == '*.xlsx':
            self.df.to_excel(fileName_choose)
        if filetype =='*.txt':
            result = self.result.toPlainText()
            file = open(fileName_choose, 'w')
            file.write(result)
            file.close()

        if fileName_choose == "":
            self.printfResultln('取消保存')
        else:
            self.printfResultln('您保存的文件为:')
            self.printfResultln(fileName_choose)
    #导入数据
    def importData(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "打开excel", "", "*.xlsx;;All Files(*)")
        self.print.clear()
        if fileName == "":
            self.result.clear()
            self.printfResultln("您未选择任何文件~😔")
            return
        if fileName != "" and filetype != "*.xlsx":
            QMessageBox.critical(None, "错误文件格式", "╮(￣▽ ￣)╭请导入excel（*.xlsx）格式文件", QMessageBox.Ok)
            return
        else:
            self.df = pd.read_excel(fileName)
            self.result.clear()
            self.printfln("导入成功")
            self.printfResultln(str(self.df))
    #预测结果
    def predicateResults(self):
        if str(pd.isnull(self.df['浓度']))=="Series([], Name: 浓度, dtype: bool)":
            QMessageBox.critical(None, "无法估计~", "您还没获得测量数据呢(●'◡'●)", QMessageBox.Ok)
            return
        self.print.clear()
        self.print_ave.clear()
        self.le_concentration.clear()
        global imgNamepath, getpos
        if (imgNamepath == ""):
            image = cv_imread('image/img.png')
        else:
            image = cv_imread(imgNamepath)
        index = self.lv_evaluate.currentText()
        # 线性回归,反向拟合
        clf = linear_model.LinearRegression()
        x1 = pd.DataFrame(self.df['浓度'])
        y1 = pd.DataFrame(self.df['数值'])
        clf.fit(y1, x1)  # 拟合模型
        if(index=="比色法估计"):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            def getpos(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                    b = int(rgb[y, x][0])
                    g = int(rgb[y, x][1])
                    r = int(rgb[y, x][2])
                    q = np.sqrt(r * r + g * g + b * b)
                    x0=np.array([[q]])
                    result=clf.predict(x0)
                    self.printEvaluate(result)
        if (index == "HSV估计"):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            def getpos(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                    x0 = np.array([[HSV[y, x][1]]])
                    result = clf.predict(x0)
                    self.printEvaluate(result)
        if (index == "RGB估计"):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            def getpos(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                    q = (int(image[y, x][0])+int(image[y, x][1])+int(image[y, x][2]))/3
                    x0 = np.array([[q]])
                    result = clf.predict(x0)
                    self.printEvaluate(result)
        if(index == "灰度估计"):
            Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            def getpos(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                    x0 = np.array([[Gray[y, x]]])
                    result = clf.predict(x0)
                    self.printEvaluate(result)
        cv_imageWin(image, getpos, "evaluate")
    def hsv_table(self):
        image = cv_imread("image/hsv-table.png")
        cv2.imshow('image', image)
    def delectLine(self):
        index1 = self.le_delect.text()

        if(index1==""):
            QMessageBox.critical(None, "没有数据", "请输入数据", QMessageBox.Ok)
            return
        if (self.df.empty):
            QMessageBox.critical(None, "无效表格", "表格中无数据", QMessageBox.Ok)
            return
        index = int(index1)
        if(index>self.df.index.values[-1]):
            QMessageBox.critical(None, "索引越界", "请检查删除的行是否正确", QMessageBox.Ok)
            return
        else:self.df.drop(index=index,inplace=True)
        self.result.clear()
        if (self.df.empty): self.printfResultln("没有数据啦~~")
        else:
            self.df.reset_index(drop=True, inplace=True)  # 更新index
            self.printfResultln(str(self.df))
    # def mergeCon(self):
    #     self.df.drop_duplicates(['浓度'],keep=False)  # 去除所有重复行
    #     self.result.clear()
    #     self.df.reset_index(drop=True, inplace=True)  # 更新index
    #     self.printfResultln(str(self.df))
    def clearDf(self):
        self.df.drop(self.df.index, inplace=True)
        self.df = self.df.drop(index=self.df.index)
        self.result.clear()
        self.df.reset_index(drop=True, inplace=True)  # 更新index
        self.printfResultln(str(self.df))

'''中文路径，opencv常用函数 '''
## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
'''openCV创建一个图片窗口'''
def cv_imageWin(image,pointFun,name):
    # 创建一个名字叫做 evaluate的窗口
    cv2.namedWindow(name, 4)
    cv2.imshow(name, image)
    cv2.setMouseCallback(name, pointFun)
    cv2.waitKey(0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui_color = ColorUi()
    #ui_color = CandyWindow.createWindow(ui_color, 'blueDeep')
    ui_color.show()
    sys.exit(app.exec_())