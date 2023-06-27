import Rbeast as rb
import pwlf
import pandas as pd
import scipy
import scipy.io as scio
from scipy.io import loadmat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from osgeo import gdal
import itertools
import h5py
import pickle
from matplotlib.font_manager import FontProperties
# def read_img(filename):
#         dataset = gdal.Open(filename)
#         im_width = dataset.RasterXSize #栅格矩阵的列数
#         im_height = dataset.RasterYSize #栅格矩阵的行数
#         im_geotrans = dataset.GetGeoTransform() #仿射矩阵
#         im_proj = dataset.GetProjection() #地图投影
#         im_data = dataset.ReadAsArray(0,0,im_width,im_height)
#         del dataset
#         return im_proj,im_geotrans,im_data
def read_img(img_path):
    """读取遥感数据信息"""
    dataset = gdal.Open(img_path)
    img_width = dataset.RasterXSize #栅格矩阵的列数
    img_height = dataset.RasterYSize #栅格矩阵的行数
    adf_GeoTransform = dataset.GetGeoTransform() #仿射矩阵
    img_Proj = dataset.GetProjection() #地图投影
    img_data = dataset.ReadAsArray(0,0,img_width,img_height)
    del dataset
    return img_width, img_height, adf_GeoTransform, img_Proj, img_data

def arr2img(img_width, img_height, adf_GeoTransform, img_Proj, save_path, arr):
    # 保存为jpg格式
    #plt.imsave(save_path, arr)
    # 保存为TIF格式
    
    if 'int8' in arr.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in arr.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName("GTiff")
    datasetnew = driver.Create(save_path, img_width, img_height, 1, gdal.GDT_Float32)
    datasetnew.SetGeoTransform(adf_GeoTransform)
    datasetnew.SetProjection(img_Proj)
    band = datasetnew.GetRasterBand(1)
    band.WriteArray(arr)
    del datasetnew
    del driver

def beast_threshold(metaTime, Data, o, Prob = 0.5, DeltaH = 10, numImg = 3):
    O_Time = o.time
    Y_data = o.trend.Y
    cp = o.trend.cp
    cpPr = o.trend.cpPr
    Tncp_median = o.trend.ncp_median
    TCP = cp[:, :, 0]
    TPR = cpPr[:, :, 0]
    cpCI = o.trend.cpCI
    cpAbrupt = o.trend.cpAbruptChange
    cpCI_Start = cpCI[:,:,:,0]
    cpCI_End = cpCI[:,:,:,1]
    index_Prob = np.argwhere(cpPr >= Prob)
    lenProb = len(index_Prob[:,-1])
    [M,N,Q] = np.shape(cp)
    numImg_arr = np.zeros([M,N,Q])
    for pixel in range(lenProb):
        xProb = index_Prob[pixel,0]
        yProb = index_Prob[pixel,1]
        zProb = index_Prob[pixel,2]
        Start = cpCI_Start[xProb,yProb,zProb]
        End = cpCI_End[xProb,yProb,zProb]
        indexTime = np.argwhere((metaTime >= Start) & (metaTime <= End))
        Data_indexTime = Data[xProb,yProb,indexTime]
        numImg_arr[xProb,yProb,zProb]  = sum(~np.isnan(Data_indexTime))
    
    index_Threshold = np.where((cpPr >= Prob) & (abs(cpAbrupt) >= DeltaH) & (numImg_arr >= numImg) )
    cp_Thre = cp[index_Threshold].reshape([-1,1])
    cpPr_Thre = cpPr[index_Threshold].reshape([-1,1])
    cpAbrupt_Thre = cpAbrupt[index_Threshold].reshape([-1,1])
    #out_Threshold = np.concatenate((np.array(index_Threshold),cp_Thre,cpPr_Thre,cpAbrupt_Thre), axis=0)
    return index_Threshold

def beast_segmentslp(metaTime, Data, o, Prob = 0.5):
    O_Time = o.time
    Y_data = o.trend.Y
    ncp_mode = o.trend.ncp_mode
    cp = o.trend.cp
    cpPr = o.trend.cpPr
    Tncp_median = o.trend.ncp_median
    TCP = cp[:, :, 0]
    TPR = cpPr[:, :, 0]
    cpCI = o.trend.cpCI
    cpAbrupt = o.trend.cpAbruptChange
    cpCI_Start = cpCI[:,:,:,0]
    cpCI_End = cpCI[:,:,:,1]
    index_Prob = np.argwhere(cpPr >= Prob)
    lenProb = len(index_Prob[:,-1])
    [M,N,Q] = np.shape(cp)
    numImg_arr = np.zeros([M,N,Q])
    #只关注二维索引
    index_segment = np.argwhere(TPR >= Prob)
    lenSegment = len(index_segment[:,-1])
    #Time_and_Slope_arr = np.zeros([M,N])
    TimeSlope_cell = np.empty((M,N),dtype=object)
    SignSlope_arr = np.zeros((M,N))*np.nan
    for pixel in range(lenSegment):
        xProb = index_Prob[pixel,0]
        yProb = index_Prob[pixel,1]
        pixel_ncp_mode =  int(ncp_mode[xProb,yProb])
        print('ncp_mode:',pixel_ncp_mode)
        pixel_Y = Data[xProb,yProb,:]
        pixel_cpPr = cpPr[xProb,yProb,0:pixel_ncp_mode]
        pixel_cp = cp[xProb, yProb, 0:pixel_ncp_mode]

        #真实突变点时刻前后的置信区间设置为一段，而对于未通过阈值的点并不单独设置区间
        index_ProbHigh = np.argwhere(pixel_cpPr >= Prob)
        index_ProbLow = np.argwhere(pixel_cpPr < Prob)
        Start_ProbHigh = cpCI_Start[xProb,yProb,index_ProbHigh].reshape(-1)
        End_ProbHigh = cpCI_End[xProb,yProb,index_ProbHigh].reshape(-1)
        Time_ProbLow = pixel_cp[index_ProbLow].reshape(-1)
        Time_minmax = np.array([np.min(O_Time),np.max(O_Time)])
        Time_List0 = np.concatenate((Time_minmax,Time_ProbLow,Start_ProbHigh,End_ProbHigh))
        Time_List = np.sort(Time_List0)
        print(Time_List)
        #统计每个分段的斜率
        slope_List = np.zeros([len(Time_List)-1,])
        #slp_sign_List = np.zeros([len(Time_List)-1,])
        slp_sign_List = ''
        for i_time in range(len(Time_List)-1):
            index_left = np.where(metaTime >= Time_List[i_time])[0][0]
            index_right = np.where(metaTime <= Time_List[i_time+1])[0][-1]
            dataT = metaTime[index_left:index_right+1]
            dataY = pixel_Y[index_left:index_right+1]
            dataTm = np.ma.masked_array(dataT, mask=np.isnan(dataY)).compressed()
            dataYm = np.ma.masked_array(dataY, mask=np.isnan(dataY)).compressed()
            #print(len(dataTm), len(dataYm))
            print(dataTm, dataYm)
            if len(dataTm) == 0:
                continue
            slope, intercept, r_value, p_value, std = scipy.stats.linregress(dataTm,dataYm)
            slope_List[i_time] = slope
            if (slope >= 0.05):
                #slp_sign_List[i_time] = 1
                slp_sign_List = slp_sign_List + '3'
            elif (slope <= 0.05):
                #slp_sign_List[i_time] = -1
                slp_sign_List = slp_sign_List + '1'
            else:
                #slp_sign_List[i_time] = 0
                slp_sign_List = slp_sign_List + '2'
        
        Time_NewList = Time_List[0:-1]
        Time_and_Slope = np.vstack((Time_NewList,slope_List))
        TimeSlope_cell[xProb,yProb] = Time_and_Slope
        SignSlope_arr[xProb,yProb] = int(slp_sign_List)
    return TimeSlope_cell,SignSlope_arr
    #return Time_List
    # for pixel in range(lenProb):
    #     xProb = index_Prob[pixel,0]
    #     yProb = index_Prob[pixel,1]
    #     zProb = index_Prob[pixel,2]
    #     Start = cpCI_Start[xProb,yProb,zProb]
    #     End = cpCI_End[xProb,yProb,zProb]
    #     indexTime = np.argwhere((metaTime >= Start) & (metaTime <= End))
    #     Data_indexTime = Data[xProb,yProb,indexTime]
    #     numImg_arr[xProb,yProb,zProb]  = sum(~np.isnan(Data_indexTime))
    
def beast_segmentslp_trendData(metaTime, Data, o, Prob = 0.5):
    O_Time = o.time
    Y_data = o.trend.Y
    ncp_mode = o.trend.ncp_mode
    cp = o.trend.cp
    cpPr = o.trend.cpPr
    slp = o.trend.slp
    Tncp_median = o.trend.ncp_median
    TCP = cp[:, :, 0]
    TPR = cpPr[:, :, 0]
    cpCI = o.trend.cpCI
    cpAbrupt = o.trend.cpAbruptChange
    cpCI_Start = cpCI[:,:,:,0]
    cpCI_End = cpCI[:,:,:,1]
    index_Prob = np.argwhere(cpPr >= Prob)
    lenProb = len(index_Prob[:,-1])
    [M,N,Q] = np.shape(cp)
    numImg_arr = np.zeros([M,N,Q])
    #只关注二维索引
    index_segment = np.argwhere(TPR >= Prob)
    lenSegment = len(index_segment[:,-1])
    #Time_and_Slope_arr = np.zeros([M,N])
    TimeSlope_cell = np.empty((M,N),dtype=object)
    #SignSlope_arr = np.zeros((M,N),dtype=str)*np.nan
    SignSlope_arr = np.ones([M,N]) * (-9999)
    SignSlope_arr = SignSlope_arr.astype(int)
    SignSlope_arr = SignSlope_arr.astype(dtype=str)
    for pixel in range(lenSegment):
        xProb = index_Prob[pixel,0]
        yProb = index_Prob[pixel,1]
        pixel_ncp_mode =  int(ncp_mode[xProb,yProb])
        #print('ncp_mode:',pixel_ncp_mode)
        pixel_Y = Data[xProb,yProb,:]
        pixel_cpPr = cpPr[xProb,yProb,0:pixel_ncp_mode]
        pixel_cp = cp[xProb, yProb, 0:pixel_ncp_mode]
        pixel_slp = slp[xProb,yProb,:]
        #真实突变点时刻前后的置信区间设置为一段，而对于未通过阈值的点并不单独设置区间
        index_ProbHigh = np.argwhere(pixel_cpPr >= Prob)
        index_ProbLow = np.argwhere(pixel_cpPr < Prob)
        Start_ProbHigh = cpCI_Start[xProb,yProb,index_ProbHigh].reshape(-1)
        #Start_ProbHigh = O_Time[np.argmin(abs(O_Time))]
        End_ProbHigh = cpCI_End[xProb,yProb,index_ProbHigh].reshape(-1)
        Time_ProbLow = pixel_cp[index_ProbLow].reshape(-1)
        Time_minmax = np.array([np.min(O_Time),np.max(O_Time)])
        Time_List0 = np.concatenate((Time_minmax,Time_ProbLow,Start_ProbHigh,End_ProbHigh))
        Time_List = np.sort(Time_List0)
        T_trend = Time_List * np.nan
        for k in range(len(Time_List)):
            T = Time_List[k]
            T_trend[k] = O_Time[np.argmin(abs(O_Time-T))]
        #使用趋势线中的时间数据与斜率数据
        Time_List = T_trend
        #print(Time_List)
        #统计每个分段的斜率
        slope_List = np.zeros([len(Time_List)-1,])
        #slp_sign_List = np.zeros([len(Time_List)-1,])
        slp_sign_List = ''
        for i_time in range(len(Time_List)-1):
            #index_left = np.where(metaTime >= Time_List[i_time])[0][0]
            #index_right = np.where(metaTime <= Time_List[i_time+1])[0][-1]
            #dataT = metaTime[index_left:index_right+1]
            #dataY = pixel_Y[index_left:index_right+1]
            #寻找指定分段的左右区间
            index_left = np.where(O_Time >= Time_List[i_time])[0][0]
            index_right = np.where(O_Time <= Time_List[i_time+1])[0][-1]
            dataT = O_Time[index_left:index_right+1]
            dataSlp = pixel_slp[index_left:index_right+1]
            #dataTm = np.ma.masked_array(dataT, mask=np.isnan(dataY)).compressed()
            #dataYm = np.ma.masked_array(dataY, mask=np.isnan(dataY)).compressed()
            #print(len(dataT), len(dataSlp))
            #测试
            if (xProb == 39 & xProb == 138):
                print(dataT,dataSlp)
            if len(dataT) == 0:
                continue
            #slope, intercept, r_value, p_value, std = scipy.stats.linregress(dataTm,dataYm)
            slope = np.nanmean(dataSlp)
            slope_List[i_time] = slope
            if (slope >= 0.0005):
                #slp_sign_List[i_time] = 1
                slp_sign_List = slp_sign_List + '+'
            elif (slope <= 0.0005):
                #slp_sign_List[i_time] = -1
                slp_sign_List = slp_sign_List + '-'
            else:
                #slp_sign_List[i_time] = 0
                slp_sign_List = slp_sign_List + '0'
        
        Time_NewList = Time_List[0:-1]
        Time_and_Slope = np.vstack((Time_NewList,slope_List))
        TimeSlope_cell[xProb,yProb] = Time_and_Slope
        #SignSlope_arr[xProb,yProb] = int(slp_sign_List)
        SignSlope_arr[xProb,yProb] = slp_sign_List
    return TimeSlope_cell,SignSlope_arr



def data_threshold(data,Threshold):
    [M,N,Q] = np.shape(data)
    data_New = np.zeros([M,N,Q])*(np.nan)
    for num in range(len(Threshold[0])):
         x = Threshold[0][num]
         y = Threshold[1][num]
         data_New[x,y,:] = data[x,y,:]
    return data_New
  
def img_threshold(data,Threshold):
    [M,N] = np.shape(data)
    data_New = np.zeros([M,N])*(np.nan)
    for num in range(len(Threshold[0])):
         x = Threshold[0][num]
         y = Threshold[1][num]
         data_New[x,y] = data[x,y]
    return data_New  
#raster=gdal.Open("SRTM_pm150033_16.tif")
#数据预处理
tifpath = 'interest_glacier_Raster.tif'
[img_width, img_height, adf_GeoTransform, img_Proj, I_glacier] = read_img(tifpath)
[x_glacier,y_glacier] = np.where(I_glacier != 255);

# glacier_raster = scio.loadmat('SRTM_pm150033_16.tif')
# Kyagar = glacier_raster['I_glaicer']
# [x_glacier,y_glacier] = np.where(Kyagar != 255)
num = len(x_glacier)

# dataFile = 'DEM_TIME_DATA.mat'
# #data = loadmat(dataFile)
# data = h5py.File(dataFile,'r')
# Height0 = data['Height_ALL']
# H0 = Height0[:]
# Height = H0.swapaxes(0,2)

dataFile = 'DEM_TIME_DATA.mat'
data = loadmat(dataFile)
#data = h5py.File(dataFile,'r')
Height0 = data['Height_ALL']
Height = Height0
#H0 = Height0[:]
#Height = H0.swapaxes(0,2)

#Height[Kyagar == 255] = np.nan
[M,N,Q] = np.shape(Height)
#Hnew = Height.reshape(1,-1,Q)
#Hnew = np.zeros([1,num,Q])
Hnew = np.zeros([M,N,Q])
Hnew.fill(np.nan)
for i in range(0,num):
    x = x_glacier[i]
    y = y_glacier[i]
    HT = Height[x,y,:]
    Hnew[x,y,:]= HT

#file_alltime = data['file_alltime'].reshape(-1)
file_alltime = data['file_alltime']

Height = Hnew
Htime = np.array(file_alltime).reshape(-1)
Htime = Htime[1:]
#Htime = Htime[:]
#Htime = Htime[1:]
# print(Height)
# print(Htime)
Prob = 0.5
DeltaH = 20
numImg = 2
#BEAST参数设置
# create an empty object to stuff the attributes: "metadata  = lambda: None" also works
metadata = rb.args()
metadata.isRegular = False   # data is irregularly-spaced
# times of individulal images/data points: the unit here is fractional year (e.g., 2004.232)
metadata.time = Htime
# regular interval used to aggregate the irregular time series (1/2 = 1/2 year = 6 month)
metadata.deltaTime = 1/2
# the period is 1.0 year, so freq= 1.0 /(1/2) = 2 data points per period
metadata.period = 2
# the dimension of the input ndvi is (484,10,20): which dim refers to the time. whichDimIsTime is a 1-based index
metadata.whichDimIsTime = 3
extra = rb.args()
extra.computeTrendSlope = True
#prior超参数
prior = rb.args()
prior.trendMinOrder = 0
prior.trendMaxOrder = 1
# #BEAST运行
# o_1st = rb.beast123(Height, metadata, prior, [], extra)
# #ncp_mode = np.where(Kyagar != 255,o_1st.trend.ncp_mode,np.nan)
# ncp_mode = o_1st.trend.ncp_mode
# dist_ncp = pd.DataFrame(ncp_mode[~np.isnan(ncp_mode)])
# # ax = dist_ncp.plot.kde()
# # plt.show()
# # plt.hist(ncp_mode)
# # plt.show()
# Threshold_1st = beast_threshold(Htime, Height, o_1st, Prob, DeltaH, numImg)
#Height_New = np.zeros([M,N,Q])*(np.nan)
# for num in range(len(Threshold[0])):
#      x = Threshold[0][num]
#      y = Threshold[1][num]
#      z = Threshold[2][num]
#      Height_New[x,y,z] = Height[x,y,z]
# Height_2nd = data_threshold(Height,Threshold_1st)
# o_2nd = rb.beast123(Height_2nd, metadata, [], [], extra)
# [Threshold_2nd] = beast_threshold(Htime, Height_2nd, o_2nd, Prob, DeltaH, numImg)

# Height_3rd = data_threshold(Height_2nd,Threshold_2nd)
# o_3rd = rb.beast123(Height_3rd, metadata, [], [], extra)
# [Threshold_3rd] = beast_threshold(Htime, Height_3rd, o_3rd, Prob, DeltaH, numImg)
# Height_4th = data_threshold(Height_3rd,Threshold_3rd)
# o_4th = rb.beast123(Height_4th, metadata, [], [], extra)

#读取已有beast结果文件
result_file = 'Kyagar_20230524.dict2'
f_read = open(result_file, 'rb')
o_1st = pickle.load(f_read)

o = o_1st
# result_file = 'Kyagar_20230524.dict2'
# fw=open(result_file, 'wb')
# pickle.dump(o,fw)
Threshold_1st = beast_threshold(Htime, Height, o_1st, Prob, DeltaH, numImg)
O_Time = o.time
Y_data = o.trend.Y
cp = o.trend.cp
cpPr = o.trend.cpPr
Tncp_median = o.trend.ncp_median
ncp_mode = o.trend.ncp_mode
ncp_mode_list = ncp_mode[~np.isnan(ncp_mode)]
#BEAST参数设置
# create an empty object to stuff the attributes: "metadata  = lambda: None" also works
metadata = rb.args()
metadata.isRegular = False   # data is irregularly-spaced
# times of individulal images/data points: the unit here is fractional year (e.g., 2004.232)
metadata.time = O_Time
# regular interval used to aggregate the irregular time series (1/2 = 1/2 year = 6 month)
metadata.deltaTime = 1/2
# the period is 1.0 year, so freq= 1.0 /(1/2) = 2 data points per period
metadata.period = 2
# the dimension of the input ndvi is (484,10,20): which dim refers to the time. whichDimIsTime is a 1-based index
metadata.whichDimIsTime = 3
extra = rb.args()
extra.computeTrendSlope = True
#o_2nd = rb.beast123(Y_data,metadata, [], [], extra)
# TCP = img_threshold(cp[:, :, 0],Threshold_1st)
# TPR = img_threshold(cpPr[:, :, 0],Threshold_1st)
cpCI = o.trend.cpCI
cpAbrupt = o.trend.cpAbruptChange

[M,N,Q] = np.shape(Height)
TCP = np.zeros([M,N])*np.nan
TPR = np.zeros([M,N])*np.nan
TCPAbrupt = np.zeros([M,N])*np.nan
for i in range(len(Threshold_1st[0])):
  x = Threshold_1st[0][i]
  y = Threshold_1st[1][i]
  TCP[x,y] = cp[x,y,0]
  TPR[x,y] = cpPr[x,y,0]
  TCPAbrupt[x,y] = cpAbrupt[x,y,0]
 
X = 39
Y = 138
[TimeSlope_cell,SignSlope_arr] = beast_segmentslp_trendData(Htime, Height, o_1st, Prob)
#print(TimeSlope_cell)
norm = matplotlib.colors.Normalize(vmin=2012,vmax=2018) 
rb.plot(o[X,Y])
#print(TimeSlope_cell[X,Y])
plt.scatter(Htime,Height[X,Y,:])
SignSlope = SignSlope_arr[SignSlope_arr != '-9999']
np.place(SignSlope_arr, SignSlope_arr == '-9999', np.nan) 
unique, frequency = np.unique(SignSlope, return_counts = True)
index_unique = unique[np.where(frequency>10)]
index_freq = frequency[np.where(frequency>10)]
k=10
index_freq_max10 = index_freq.argsort()[-k:][::-1]
unique_max10 = index_unique[index_freq_max10]
freq_max10 = index_freq[index_freq_max10]
percentage_max10 = freq_max10/len(SignSlope)
print(unique_max10)
print(freq_max10)
font_set = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=12)
plt.figure(figsize=(10, 6),dpi=300)
plt.bar(unique_max10,percentage_max10, width=0.8)
plt.ylabel('频率',fontproperties=font_set)

[M,N] = np.shape(SignSlope_arr)
SignSlope_Class = np.zeros([M,N])*np.nan
for i in range(M):
    for j in range(N):
        pixel = SignSlope_arr[i,j]
        if( '-+-' in pixel):
            SignSlope_Class[i,j] = 1 #可能的下部接收区
        elif('+-+' in pixel):
            SignSlope_Class[i,j] = 2 #可能的上部蓄积区
plt.imshow(SignSlope_Class,cmap='jet')
plt.colorbar()
plt.show()
