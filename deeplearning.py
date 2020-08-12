from torch.utils.data import Dataset, DataLoader
import torch

class Splitter:
  """**Kwargs takes inputs like val_pct,classes"""
  def classes_folder(path:str,**kwargs):
    import os
    import numpy as np
    import pandas as pd
    classes,df=[],pd.DataFrame()
    try:
        classes=kwargs["classes"]
    except Exception:
      for i in os.listdir(path):
        classes.append(i)
    classes.sort()
    label=[x for x in range(len(classes))]
    files=os.listdir(os.path.join(path,classes[0]))
    df["Image Index"]=files
    df["Image Labels"]=label[0]
    for i in range(1,len(label)):
      files=os.listdir(os.path.join(path,classes[i]))
      df_adder=pd.DataFrame({'Image Index':files,'Image Labels':label[i]})
      df=pd.concat([df,df_adder],axis=0,ignore_index=True)
    try:
      val_size=int(kwargs["val_pct"]*len(df))
      idx=np.random.permutation(len(df))
      return df.iloc[idx[val_size:]],df.iloc[idx[:val_size]],classes
    except Exception:
      idx=np.random.permutation(len(df))
      return df.iloc[idx],classes


class ImageGenerator(Dataset):
  from pandas import DataFrame
  from torchvision import transforms
  """takes input path, classes, dataframe and transforms and create a pytorch dataset"""
  def __init__(self,path:str,classes:list,df:DataFrame,tf:transforms):
    self.path=path
    self.df=df
    self.classes=classes
    self.c=len(classes)
    self.tf=tf
  def __len__(self):
    return len(df)
  def __getitem__(self,idx):
    from PIL import Image
    from os import path
    import torch
    obj=self.df.iloc[idx].to_numpy()
    img=path.join(self.path,path.join(self.classes[int(obj[1])],obj[0]))
    return self.tf(Image.open(img)),torch.tensor(int(obj[1]),dtype=torch.int)