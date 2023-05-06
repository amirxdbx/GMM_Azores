import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import os
import joblib

     
@st.cache(allow_output_mutation=True)

def model():
    model=pickle.load(open(f'Tuned_models/All_targets.sav', 'rb'))
    return model
       
model='Xgboost'
st.title("""
Ground motion model 
This app predicts the **geometric mean of ground motion intensities** 
""")

st.sidebar.image("logo.png",width=30)
st.sidebar.title('Define your input')

Mw = st.sidebar.slider("Mw",min_value=5.0, value=6.0,max_value=6.8,step=0.1, help="Please enter a value between 5.0 and 6.8")
RJB = st.sidebar.slider("RJB",min_value=0, value=30,max_value=148,step=1, help="Please enter a value between 0 and 148 km")
FD = st.sidebar.slider("Focal Depth",min_value=5, value=10,max_value=17.35,step=1, help="Please enter a value between 5 and 17.35 km")

    
x=pd.DataFrame({'Mw':[Mw],'Focal Depth':[FD],'RJB':[RJB]})
st.title('Summary of your inputs:')
st.write(x)
st.sidebar.markdown("Made by [Amirhossein Mohammadi](https://www.linkedin.com/in/amir-hossein-mohammadi-86729957/)")
st.sidebar.markdown("---")

###############################################################
st.title('Outputs:')
PGA_model=model()
PGA=np.exp(PGA_model.predict(x)[0][0])
PGV=np.exp(PGA_model.predict(x)[0][1])
st.text('PGA= '+ str(np.round(PGA,2)) +'  cm/s2')
st.text('PGV= '+ str(np.round(PGV,2)) +'  cm/s')

PSAs=np.exp(PGA_model.predict(x)[0][2:])


PSAs= pd.DataFrame()
PSAs['PSAs']=PSAs
PSAs['T']=['0.03','0.05','0.075','0.1','0.15','0.2','0.25','0.3','0.4','0.5','0.75','1.0','1.5','2.0']
PSAs.sort_values(by=["T"], inplace = True) 
PSAs.reset_index(drop=True,inplace=True)

fig, ax = plt.subplots(figsize=(8,2))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(PSAs['T'],PSAs['PSAs'],color='k')
plt.xlabel('T (s)')
plt.ylabel(r'$PSA\ (cm/s^2)$')
plt.xlim(0.01,3.5)
plt.ylim(0,1000)
plt.grid(which='both')
plt.savefig('sprectra.png',dpi=600,bbox_inches='tight',pad_inches=0.05)
plt.gcf().subplots_adjust(bottom=0.15)

from PIL import Image
image = Image.open('sprectra.png')
st.image(image)

def convert_df(df):
    return df.to_csv().encode('utf-8')
csv = convert_df(PSAs)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='PSAs.csv',
    mime='text/csv',
)