import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
scaler_filename = "mms.save"
def mms_fit(im,path):
    #calculate mms
    mmScaler=MinMaxScaler(copy=False)
    img_temp = im.reshape(im.shape[0], -1)
    mmScaler.fit(img_temp)    
    joblib.dump(mmScaler,path+scaler_filename)
    return

def mms_trans(im,path):
    #calculate mms
    mmScaler=joblib.load(path+scaler_filename)
    img_temp = im.reshape(im.shape[0], -1)
    img_temp = mmScaler.transform(img_temp)
    mms_img = img_temp.reshape(im.shape[0],im.shape[1],im.shape[2],im.shape[3])    
    return mms_img