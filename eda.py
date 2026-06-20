import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
import datetime
import re

DATA_FOLDER = 'data/'
pd.set_option('display.max_columns', 10, 'display.max_rows', 150, 'display.width', 1000) # type: ignore

df = pd.read_csv(DATA_FOLDER + 'Highway-Rail_Grade_Crossing_Accident_Data__Form_57__20240925.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Maintainance Incident Number': 'Maintenance Incident Number'})

list(df['State Name'].unique())

########### check if narratives contains some keywords ############
df[df['Narrative'].str.contains('DERAIL').fillna(False)]['Narrative']
df[df['Railroad Name'].str.contains('AMTRAK').fillna(False)]

############## Accidents that occurred at the same place around the same time #############
df_gb = df.groupby(['Date', 'City Name'])
df_gn_num = df_gb.apply(lambda df: df.shape[0])
df_gn_num[df_gn_num != 1]

################################ news samples #######################################
# https://www.nytimes.com/1982/03/15/nyregion/train-kills-9-teen-agers-on-li-as-van-goes-past-crossing-gate.html
df[df['Date'] == '1982-03-14'][['State Name', 'County Name', 'Date', 'Highway User', 'Train Speed', 'Total Killed Form 57', 'Railroad Code']].loc[29508:]
# https://www.nytimes.com/1984/07/12/us/amtrak-train-hits-truck-carrying-gas-2-are-killed.html
df[df['Date'] == '1984-07-11'][['State Name', 'County Name', 'Highway Name', 'Highway User', 'Train Speed', 'Total Killed Form 57', 'Railroad Code']]
# https://www.nytimes.com/1975/09/24/archives/school-principal-and-5-girls-killed-in-cartrain-crash.html
df[df['Date'] == '1975-09-22'][['State Name', 'County Name', 'Date', 'Highway User', 'Train Speed', 'Total Killed Form 57', 'Railroad Code']].iloc[-1:]

################################################################################
list_col_nm = ['Railroad Code', 'Railroad Name', 'Report Year', 
               'Incident Number', 'Incident Year', 'Incident Month', 
               'Other Railroad Code', 'Other Railroad Name', 'Other Incident Number', 
               'Other Incident Year', 'Other Incident Month', 
               'Maintenance Incident Railroad Code', 'Maintenance Railroad Name', 
               'Maintainance Incident Number', 'Maintenance Incident Year', 'Maintenance Incident Month', 
               'Grade Crossing ID', 'Date', 'Time', 'Month', 'Day', 'Hour', 'Minute', 'AM/PM', 
               'Nearest Station', 'Division', 'Subdivision', 'County Code', 'County Name', 
               'State Code', 'State Name', 'City Name', 'Highway Name', 
               'Public/Private Code', 'Public/Private', 'Highway User Code', 'Highway User', 
               'Estimated Vehicle Speed', 'Vehicle Direction Code', 'Vehicle Direction', 
               'Highway User Position Code', 'Highway User Position', 
               'Equipment Involved Code', 'Equipment Involved', 
               'Railroad Car Unit Position', 'Equipment Struck Code', 'Equipment Struck', 
               'Hazmat Involvement Code', 'Hazmat Involvement', 
               'Hazmat Released by Code', 'Hazmat Released by', 'Hazmat Released Name', 
               'Hazmat Released Quantity', 'Hazmat Released Measure', 'Temperature', 
               'Visibility Code', 'Visibility', 'Weather Condition Code', 'Weather Condition', 
               'Equipment Type Code', 'Equipment Type', 'Track Type Code', 'Track Type', 
               'Track Name', 'Track Class', 'Number of Locomotive Units', 'Number of Cars', 
               'Train Speed', 'Estimated/Recorded Speed', 'Train Direction Code', 'Train Direction', 
               'Crossing Warning Expanded Code 1', 'Crossing Warning Expanded Code 2', 'Crossing Warning Expanded Code 3', 'Crossing Warning Expanded Code 4', 'Crossing Warning Expanded Code 5', 'Crossing Warning Expanded Code 6', 'Crossing Warning Expanded Code 7', 'Crossing Warning Expanded Code 8', 'Crossing Warning Expanded Code 9', 'Crossing Warning Expanded Code 10', 'Crossing Warning Expanded Code 11', 'Crossing Warning Expanded Code 12', 
               'Crossing Warning Expanded 1', 'Crossing Warning Expanded 2', 'Crossing Warning Expanded 3', 'Crossing Warning Expanded 4', 'Crossing Warning Expanded 5', 'Crossing Warning Expanded 6', 'Crossing Warning Expanded 7', 'Crossing Warning Expanded 8', 'Crossing Warning Expanded 9', 'Crossing Warning Expanded 10', 'Crossing Warning Expanded 11', 'Crossing Warning Expanded 12', 
               'Signaled Crossing Warning Code', 'Signaled Crossing Warning', 
               'Crossing Warning Explanation Code', 'Crossing Warning Explanation', 
               'Roadway Condition Code', 'Roadway Condition', 
               'Crossing Warning Location Code', 'Crossing Warning Location', 
               'Warning Connected To Signal', 'Crossing Illuminated', 
               'User Age', 'User Gender', 'User Struck By Second Train', 
               'Highway User Action Code', 'Highway User Action', 'Driver Passed Vehicle', 
               'View Obstruction Code', 'View Obstruction', 
               'Driver Condition Code', 'Driver Condition', 'Driver In Vehicle', 
               'Crossing Users Killed', 'Crossing Users Injured', 
               'Vehicle Damage Cost', 'Number Vehicle Occupants', 
               'Employees Killed', 'Employees Injured', 
               'Number People On Train', 'Form 54 Filed', 
               'Passengers Killed', 'Passengers Injured', 
               'Video Taken', 'Video Used', 
               'Special Study 1', 'Special Study 2', 'Narrative', 
               'Total Killed Form 57', 'Total Injured Form 57', 
               'Railroad Type', 'Joint Code', 'Total Killed Form 55A', 'Total Injured Form 55A', 
               'District', 'Whistle Ban Code', 'Whistle Ban', 'Report Key']

df_cal = df[df['State Name'] == 'CALIFORNIA'].reset_index(drop=True)

df_cal['Highway Name'].value_counts()
df_cal['City Name'].value_counts()
df_cal['County Name'].value_counts()
df_cal['Division'].value_counts()
df_cal['Subdivision'].value_counts()
# df_cal['County Name'].value_counts()

df_cal['Highway User'].value_counts()
df_cal['Highway User Position'].value_counts()
df_cal['Equipment Struck'].value_counts()
df_cal['Highway User Action'].value_counts()
df_cal['Crossing Users Killed'].value_counts()
df_cal['Crossing Users Injured'].value_counts()
df_cal['Total Killed Form 57'].value_counts()


""" """ """ """ """ 'Report Year', 'Incident Year' """ """ """ """ """
assert (df['Report Year'].astype(str).str[2:].astype(int) == df['Incident Year']).all()
###
df.drop('Report Year', axis=1)

""" """ """ """ """ 'Incident Year|Month', 'Other Incident Year|Month' (if exists) """ """ """ """ """
df_temp1 = df[~df['Other Incident Month'].isna()] # 242283 of 246849 is na
(df_temp1['Other Incident Month'] == df_temp1['Incident Month']).sum()
### two columns are the same, but just keep

""" """ """ """ """ 'Date', 'Incident Year|Month', 'Month|Day' """ """ """ """ """
df_temp3 = df[~df['Date'].isna()]
assert (df_temp3['Date'].dt.year.astype(str).str[2:4].astype(int) == df_temp3['Incident Year']).all()
assert (df_temp3['Date'].dt.month == df_temp3['Incident Month']).all()
(df_temp3['Date'].dt.month != df_temp3['Month']).sum() # 1 not matched
assert (df_temp3['Date'].dt.day == df_temp3['Day']).all()
###
df.dropna(subset=['Date']).drop(['Incident Year', 'Incident Month', 'Month', 'Day'], axis=1)

""" """ """ """ """ 'Time', 'Hour|Minute|AM/PM' """ """ """ """ """
df_temp4 = pd.to_datetime(df['Time'])
assert (df['Time'].str[:2].astype(int) == df['Hour']).all()
assert (df['Time'].str[3:5].astype(int) == df['Minute']).all()
(df['Time'].str[6:8] != df['AM/PM']).sum() # 31 not matched
###
df.drop(['Hour', 'Minute'], axis=1)[df['Time'].str[6:8] == df['AM/PM']]

""" """ """ """ """ 'County Code', 'County Name' """ """ """ """ """
df_temp5 = df_cal[~df_cal[['County Name', 'County Code']].isna().all(axis=1)]
df_temp5['County Code'].value_counts()
df_temp5['County Name'].value_counts()
# significantly unmatched

""" """ """ """ """ 'State Code', 'State Name' """ """ """ """ """
df_temp6 = df[~df[['State Code', 'State Name']].isna().all(axis=1)]
dict_state = dict(zip(df_temp6['State Code'].unique(), df_temp6['State Name'].unique()))
(df_temp6['State Name'] != df_temp6['State Code'].map(dict_state)).sum() # 2 unmatched
###
df.dropna(subset=['State Name', 'State Code'], how='all')

""" """ """ """ """ etc """ """ """ """ """
df['Nearest Station']
df[~df['Division'].isna()]['Division']
df[~df['Subdivision'].isna()]['Subdivision']
df['County Name'].isna().sum()
df['State Name'].isna().sum()
df['City Name'].isna().sum()
df['Highway Name'].isna().sum()

""" """ """ """ """ 'Public/Private Code', 'Public/Private' """ """ """ """ """
assert ((df['Public/Private Code'] == 'Y') == (df['Public/Private'] == 'Public')).all()
###
df.drop('Public/Private Code', axis=1)

""" """ """ """ """ 'Highway User Code', 'Highway User' """ """ """ """ """
df_temp7 = df[~df[['Highway User Code', 'Highway User']].isna().all(axis=1)]
dict_hghw_usr = dict(zip(df_temp7['Highway User Code'].unique(), df_temp7['Highway User'].unique()))
assert (df_temp7['Highway User'] == df_temp7['Highway User Code'].map(dict_hghw_usr)).all() # 5 unmatched
###
df.drop('Highway User Code', axis=1)

""" """ """ """ """ 'Vehicle Direction Code', 'Vehicle Direction' """ """ """ """ """
df_temp8 = df[~df[['Vehicle Direction Code', 'Vehicle Direction']].isna().all(axis=1)]
df_temp8 = df_temp8[~(df_temp8['Vehicle Direction Code'].isin(['A', 0]))]
df_temp8['Vehicle Direction Code'] = df_temp8['Vehicle Direction Code'].astype(int)
dict_vh_drct = dict(zip(df_temp8['Vehicle Direction Code'].unique(), df_temp8['Vehicle Direction'].unique()))
assert (df_temp8['Vehicle Direction'] == df_temp8['Vehicle Direction Code'].map(dict_vh_drct)).all()
###
df['Vehicle Direction Code'].replace({'A': np.nan, 0: np.nan, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0})

""" """ """ """ """ 'Highway User Position Code', 'Highway User Position' """ """ """ """ """
df_temp9 = df[~df[['Highway User Position Code', 'Highway User Position']].isna().all(axis=1)]
dict_hghw_usr_pst = dict(zip(df_temp9['Highway User Position Code'].unique(), df_temp9['Highway User Position'].unique()))
assert (df_temp9['Highway User Position'] == df_temp9['Highway User Position Code'].map(dict_hghw_usr_pst)).all()
###
df.drop('Highway User Position Code', axis=1)

""" """ """ """ """ 'Equipment Involved Code', 'Equipment Involved' """ """ """ """ """
df_temp10 = df[~df[['Equipment Involved Code', 'Equipment Involved']].isna().all(axis=1)]
df_temp10['Equipment Involved Code'] = df_temp10['Equipment Involved Code'].replace({x: str(x) for x in range(1, 10)})
dict_eqpm_invlv = dict(zip(df_temp10['Equipment Involved Code'].unique(), df_temp10['Equipment Involved'].unique()))
assert (df_temp10['Equipment Involved'] == df_temp10['Equipment Involved Code'].map(dict_eqpm_invlv)).all()
###
df.drop('Equipment Involved Code', axis=1)
df['Equipment Involved Code'].replace({x: str(x) for x in range(1, 10)})

""" """ """ """ """ 'Equipment Struck Code', 'Equipment Struck' """ """ """ """ """
df_temp11 = df[~df[['Equipment Struck Code', 'Equipment Struck']].isna().all(axis=1)]
df_temp11['Equipment Struck Code'].value_counts()
df_temp11['Equipment Struck'].value_counts()
dict_eqpm_strck = dict(zip(df_temp11['Equipment Struck Code'].unique(), df_temp11['Equipment Struck'].unique()))
(df_temp11['Equipment Struck'] != df_temp11['Equipment Struck Code'].map(dict_eqpm_strck)).sum() # 3 na
###

""" """ """ """ """ 'Hazmat Involvement Code', 'Hazmat Involvement' """ """ """ """ """
df_temp12 = df[~df[['Hazmat Involvement Code', 'Hazmat Involvement']].isna().all(axis=1)]
df_temp12['Hazmat Involvement Code'].value_counts()
df_temp12['Hazmat Involvement'].value_counts()
dict_hzmt_invlvm = dict(zip(df_temp12['Hazmat Involvement Code'].unique(), df_temp12['Hazmat Involvement'].unique()))
(df_temp12['Hazmat Involvement'] != df_temp12['Hazmat Involvement Code'].map(dict_hzmt_invlvm)).sum() # 13 na
###

""" """ """ """ """ 'Hazmat Released by Code', 'Hazmat Released by' """ """ """ """ """
df_temp13 = df[~df[['Hazmat Released by Code', 'Hazmat Released by']].isna().all(axis=1)]
df_temp13['Hazmat Released by Code'].value_counts()
df_temp13['Hazmat Released by'].value_counts()
dict_hzmt_rls = dict(zip(df_temp13['Hazmat Released by Code'].unique(), df_temp13['Hazmat Released by'].unique()))
(df_temp13['Hazmat Released by'] != df_temp13['Hazmat Released by Code'].map(dict_hzmt_rls)).sum() # 1881 na
###

""" """ """ """ """ 'Visibility Code', 'Visibility' """ """ """ """ """
df_temp14 = df[~df[['Visibility Code', 'Visibility']].isna().all(axis=1)]
dict_vsbl = dict(zip(df_temp14['Visibility Code'].unique(), df_temp14['Visibility'].unique()))
assert (df_temp14['Visibility'] != df_temp14['Visibility Code'].map(dict_vsbl)).all() # 1881 na
###
df.drop('Visibility Code', axis=1)

""" """ """ """ """ 'Weather Condition Code', 'Weather Condition' """ """ """ """ """
df_temp15 = df[~df[['Weather Condition Code', 'Weather Condition']].isna().all(axis=1)]
df_temp15['Weather Condition Code'].value_counts()
df_temp15['Weather Condition'].value_counts()
dict_wthr_cndtn = dict(zip(df_temp15['Weather Condition Code'].unique(), df_temp15['Weather Condition'].unique()))
(df_temp15['Weather Condition'] != df_temp15['Weather Condition Code'].map(dict_wthr_cndtn)).sum() # 8 na
###

""" """ """ """ """ 'Equipment Type Code', 'Equipment Type' """ """ """ """ """
df_temp16 = df[~df[['Equipment Type Code', 'Equipment Type']].isna().all(axis=1)]
df_temp16['Equipment Type Code'] = df_temp16['Equipment Type Code'].replace({x: str(x) for x in range(1, 10)})
dict_eqpmt_tp = dict(zip(df_temp16['Equipment Type Code'].unique(), df_temp16['Equipment Type'].unique()))
assert (df_temp16['Equipment Type'] == df_temp16['Equipment Type Code'].map(dict_eqpmt_tp)).all()
###
df.drop('Equipment Type Code', axis=1)

""" """ """ """ """ 'Track Type Code', 'Track Type' """ """ """ """ """
df_temp17 = df[~df[['Track Type Code', 'Track Type']].isna().all(axis=1)]
dict_trck_tp = dict(zip(df_temp17['Track Type Code'].unique(), df_temp17['Track Type'].unique()))
(df_temp17['Track Type'] != df_temp17['Track Type Code'].map(dict_trck_tp)).sum() # 2 not matched
###

""" """ """ """ """ 'Train Direction Code', 'Train Direction' """ """ """ """ """
df['Train Direction Code'] = df['Train Direction Code'].replace({'A': np.nan, 0: np.nan, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0})
df_temp18 = df[~df[['Train Direction Code', 'Train Direction']].isna().all(axis=1)]
dict_trn_drct = dict(zip(df_temp18['Train Direction Code'].unique(), df_temp18['Train Direction'].unique()))
assert (df_temp18['Train Direction'] == df_temp18['Train Direction Code'].map(dict_trn_drct)).all() # 2 not matched
###
df.drop('Train Direction Code', axis=1)

""" """ """ """ """ 'Signaled Crossing Warning Code', 'Signaled Crossing Warning' """ """ """ """ """
df_temp19 = df[~df[['Signaled Crossing Warning Code', 'Signaled Crossing Warning']].isna().all(axis=1)]
df_temp19['Signaled Crossing Warning Code'] = df_temp19['Signaled Crossing Warning Code'].replace({str(x): float(x) for x in range(1, 10)})
dict_sgnl_wrn = dict(zip(
    df_temp19['Signaled Crossing Warning Code'].value_counts().index.values,
    df_temp19['Signaled Crossing Warning'].value_counts().index.values
))
(df_temp19['Signaled Crossing Warning'] != df_temp19['Signaled Crossing Warning Code'].map(dict_sgnl_wrn)).sum() # 4 na
###
df.drop('Signaled Crossing Warning Code', axis=1)

""" """ """ """ """ 'Crossing Warning Explanation Code', 'Crossing Warning Explanation' """ """ """ """ """
df_temp20 = df[~df[['Crossing Warning Explanation Code', 'Crossing Warning Explanation']].isna().all(axis=1)]
df_temp20['Crossing Warning Explanation Code'] = df_temp20['Crossing Warning Explanation Code'].replace({str(x): float(x) for x in range(1, 10)})
dict_crss_wrn_expln = dict(zip(
    df_temp20['Crossing Warning Explanation Code'].value_counts().index.values,
    df_temp20['Crossing Warning Explanation'].value_counts().index.values
))
assert (df_temp20['Crossing Warning Explanation'] == df_temp20['Crossing Warning Explanation Code'].map(dict_crss_wrn_expln)).all()
###
df.drop('Crossing Warning Explanation Code', axis=1)

""" """ """ """ """ 'Crossing Warning Explanation Code', 'Crossing Warning Explanation' """ """ """ """ """
df_temp20 = df[~df[['Crossing Warning Explanation Code', 'Crossing Warning Explanation']].isna().all(axis=1)]
df_temp20['Crossing Warning Explanation Code'] = df_temp20['Crossing Warning Explanation Code'].replace({str(x): float(x) for x in range(1, 10)})
dict_crss_wrn_expln = dict(zip(
    df_temp20['Crossing Warning Explanation Code'].value_counts().index.values,
    df_temp20['Crossing Warning Explanation'].value_counts().index.values
))
assert (df_temp20['Crossing Warning Explanation'] == df_temp20['Crossing Warning Explanation Code'].map(dict_crss_wrn_expln)).all()
###
df.drop('Crossing Warning Explanation Code', axis=1)



(df_cal[['Date', 'Incident Year', 'Incident Month']].isna()).all(axis=1)

(~df_cal[['Date', 'Incident Year', 'Incident Month']].isna()).any(axis=1)
# df_cal.dropna(subset=['Date'])
# df_cal['Date'].isna().sum()
