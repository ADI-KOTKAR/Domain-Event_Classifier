# -*- coding: utf-8 -*-
#%%
from Functions import *
import xlwt 
from xlwt import Workbook 

#%%
# Workbook is created 
wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1',cell_overwrite_ok=True) 
hi = classifyType("sample_input.csv")
#print(hi)

#%%
import csv
s = open('Dataset/CCMLEmployeeData.csv')
checkIt = csv.reader(s)

for c in checkIt:
    for event in hi:
        event_domain = event['domain'].replace("'","").replace("[","").replace("]","").replace("*","")
        event_type = event['type'].replace("'","").replace("[","").replace("]","").replace("*","")
        event_name = event['event'].replace("'","").replace("[","").replace("]","").replace("*","")
        if(c[1] == event_domain and c[2] == event_type):
            #print(c[0]," - ",c[1]," : ",c[2])
            event['emp'].append(c[0])
        elif (c[1] == event_domain and c[3] == event_type):
            #print(c[0]," - ",c[1]," : ",c[3])
            event['emp'].append(c[0])

#%%
for event,i in zip(hi,range(0,len(hi))):
    event_domain = event['domain'].replace("'","").replace("[","").replace("]","").replace("*","")
    event_type = event['type'].replace("'","").replace("[","").replace("]","").replace("*","")
    event_name = event['event'].replace("'","").replace("[","").replace("]","").replace("*","")
    event_emp = event['emp']
    joined_string = ", ".join(event_emp)
    #print(joined_string)
    sheet1.write(i, 0, event_name)
    sheet1.write(i, 1, joined_string)
    
wb.save('output.xls') 


       










