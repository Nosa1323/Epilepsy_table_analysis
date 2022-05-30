import re
import glob
import pandas as pd
import numpy as np
import os

def csv_to_xlsx():
    csv_search_string = '*/**.csv'
    csv_filepaths = glob.glob(csv_search_string)
    df = pd.DataFrame()
    for csv in csv_filepaths:
        out = csv.split('.')[0]+'.xlsx'
        if 'GS'in csv: 
            df = pd.read_csv(csv, sep=',', decimal = ',', encoding='latin1',engine='python')
        else:
            df = pd.read_csv(csv, sep=';', decimal = ',', encoding='latin1',engine='python')
        df = df.to_excel(out, float_format="%.4f")
    return (df)

def file_extraction(search_string): 
    filepaths = glob.glob(search_string) 
    df = pd.DataFrame()
    for find_files in filepaths:
        param = re.split ('_',find_files)
        add_df = pd.read_excel(find_files, usecols = [2, 5])
        add_df ['animal_num'] = param [0][-1]  
        add_df ['exp_group'] = param [3]
        add_df ['slice_num'] =param [1][-1]
        add_df ['cell_num'] =param [2][-1] 
        add_df ['protein'] = param[-1][:-5]
        df = pd.concat ([df, add_df], ignore_index=True)
    df.index.names = ['id'] 
    df.rename(columns={'Volume (unit)':'volume', 'SurfaceArea':'surface_area'}, inplace=True)
    df.index += 1
    return(df)

def file_folder_creator(save_path):
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    return 'Folder created'

def stat_out(stat, index):
    test_name = []
    p_val = []
    u_stat = []
    for i in stat:
        test_name.append(re.split(',',str(i))[0])
        p_val.append(re.split('=|,| ',str(i))[5])
        u_stat.append(re.split('=|,| ',str(i))[7])
        
    dt = dict(zip(['p_val', 'u_stat'], [p_val, u_stat]))
    return(pd.DataFrame(dt, index = index))