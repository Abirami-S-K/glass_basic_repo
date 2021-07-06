import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()
X = glass_df.iloc[:, :-1]
y = glass_df['GlassType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
feature_cols=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
@st.cache()
def prediction(model,feat_col):
  glass_type=model.predict([feat_col])
  glass_type=glass_type[0]
  if glass_type == 1 :
    return "building windows float processed".upper()
  elif glass_type == 2 :
    return "building windows float non processed".upper()
  elif glass_type == 3 :
    return "vehicle windows float processed".upper()
  elif glass_type == 4 :
    return "vehicle windows non float processed".upper()
  elif glass_type == 5:
    return "containers".upper()
  elif glass_type == 6 :
    return "tableware".upper()
  else :
    return "headlamp".upper()
st.title('Glass Type Prediction web app')
st.sidebar.title('Glass Type Prediction web app')
if st.sidebar.checkbox('Show raw data'):
  st.subheader('glass type data set')
  st.dataframe(glass_df)
st.sidebar.subheader('Visualisation Selector')
plot_list=st.sidebar.multiselect('Select the Charts/Plots:',('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))
if 'Line Chart' in plot_list:
  st.subheader('line chart')
  st.line_chart(glass_df) 	
if 'Area Chart' in plot_list:
  st.subheader('Area chart')
  st.area_chart(glass_df)
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

if 'Correlation Heatmap' in plot_list:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize=(10,5))
  sns.heatmap(glass_df.corr(),annot=True)
  st.pyplot()
 
 
if 'Count Plot' in plot_list:
  st.subheader('Count plot')
  plt.figure(figsize=(10,5))  
  sns.countplot(x=glass_df['GlassType'])
  st.pyplot()
   
if 'Pie Chart' in plot_list:
  st.subheader('pie chart')
  pie_data = glass_df['GlassType'].value_counts()
  plt.pie(pie_data, labels=pie_data.index, autopct='%1.2f%%', startangle=30)
  st.pyplot()
if 'Box Plot' in plot_list:
  st.subheader('box plot')
  column=st.sidebar.selectbox('select feature',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(10,5))
  sns.boxplot(glass_df[column])
  st.pyplot()
