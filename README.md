# epilepsy_table_analysis
A script designed to facilitate routine tables procedures for the analysis of tables obtained from fluorescent images

**The aim of this script** is obtaining figure and statitical pivot tables based on input data. Aditionaly, it provides a great opportunity to improve pandas knowledge.

**Input data**

There are 3 set of data (data, data2, data3).

- ***data*** contain tables connected with glutamine synthase and glutamate transporter characteristics in control group ('Контроль') and experimental group ('ЭС'):
Volume, Surface Area, XYZ mass, Feret diameter.
For our analysis we work only with Volume and Surface Area. Further, the dataset contain table neuro_count.xlsx with number of neurons in different hippocampal zones in 
observed groups and rec_count.xlsx data from electrophysiology analysis such as amplitude and decay time.  
Electrophysiology data are attended only in this dataset.

- ***data2*** 
has a similar to 'data' tables except for lack of electrophysiology data and the proteins of interest . Cx43 and s100b were used here.

- ***data3*** 
contains the same set of tables where protein of interest is GFAP.

**Result**

There are four scenarios for available data analysis:
1. 'GLT1_GS_graph_stat'. It grabs files named 'GS'/'GLT' from folder 'data', create figures and save statistical data into excel-file
2. 'Cell_count_table_analysis'. It upload file 'neuro_count.xlsx' from folder 'data', then create figures and save statistical data into excel-file
3. 'recordings calculate'. It grabs file named 'rec_count.xlsx' from folder 'data', create figures and save statistical data into excel-file
4. '4_Cx43_s100b_graph_stat' and '5_GFAP_GLT1_graph_stat' scenarios work with  'data2' and 'data3' respectively

**Dependencies**

```
Python 3.8.8
glob2 0.7
matplotlib 3.5.0
numpy 1.21.2
pandas 1.3.5
seaborn 0.11.2
scipy 1.7.3
```
