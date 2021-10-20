import numpy as np
import os

class DataLoader:
    def __init__(self, path, mask=""):
        self.datas = {}
        
        for data_file in os.listdir(path):
            file_name = os.fsdecode(data_file)
            if file_name.endswith(".csv") and mask in file_name:
                k = int(file_name.split("_k_")[1].split("_")[0])
                data_type = file_name.split(f"_k_{k}_")[1].split(".csv")[0]
                data = np.genfromtxt(os.path.join(path, file_name), delimiter=',')
                if k not in self.datas.keys():
                    self.datas[k] = {}
                self.datas[k][data_type] = data

    def merge(self):
        datas_n = 30 #len(self.datas.keys())
        datas_merged = {}
        
        for keys, values in self.datas.items():
            for key, value in values.items():
                if key not in datas_merged.keys():
                    if "eval" in key:
                        datas_merged[key] = {}
                    else:
                        datas_merged[key] = np.zeros((datas_n, ) + value.shape)
                datas_merged[key][keys] = value
        return datas_merged



