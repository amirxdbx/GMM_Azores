import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import os

st.markdown(""" <style> .font {font-size:20px ; color: #000000;} 
</style> """, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)

def model():
    models=pickle.load(open(f'All_targets.sav', 'rb'))
    return models
       
st.title("""Backbone Ground Motion Model for Azores Plateau (KMSCL23)""")

st.markdown('<p class="font">This app predicts the PGA, PGV, and PSA (5% damping) for the horizontal componenet of ground motion records in bedrock</p>', unsafe_allow_html=True)

st.sidebar.title('Define your input')

Mw = st.sidebar.slider("Mw",min_value=5.0, value=6.7,max_value=6.8,step=0.1, help="Please enter a value between 5.0 and 6.8")
RJB = st.sidebar.slider("RJB",min_value=0, value=3,max_value=148,step=1, help="Please enter a value between 0 and 148 km")
FD = st.sidebar.slider("Focal Depth",min_value=5.0, value=10.0,max_value=17.3,step=0.1, help="Please enter a value between 5 and 17.3 km")

    
x=pd.DataFrame({'Mw':[Mw],'RJB':[RJB],'Focal Depth':[FD]})
st.title('Summary of your inputs:')
st.write('Mw= '+ str(x.Mw[0])+'; RJB= '+ str(x.RJB[0])+ ' km'+ '; Focal Depth= '+ str(x['Focal Depth'][0])+ ' km')

st.sidebar.image("logo.png",width=120)
st.sidebar.markdown("Made by [Amirhossein Mohammadi](https://www.linkedin.com/in/amir-hossein-mohammadi-86729957/)")
st.sidebar.markdown("---")

###############################################################
st.title('Outputs:')
models=model()
PGA=np.exp(models.predict(x)[0][0])
PGV=np.exp(models.predict(x)[0][1])
st.text('PGA= '+ str(np.round(PGA,2)) +' (cm/s^2)')
st.text('PGV= '+ str(np.round(PGV,2)) +'  cm/s')

PSAs=np.exp(models.predict(x)[0][2:])

PSAs_df= pd.DataFrame()
PSAs_df['PSAs']=PSAs
PSAs_df['T']=[0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0,1.5,2.0]
PSAs_df.sort_values(by=["T"], inplace = True) 
PSAs_df.reset_index(drop=True,inplace=True)

fig, ax = plt.subplots(figsize=(8,2))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(PSAs_df['T'],PSAs_df['PSAs'],color='k')
plt.xlabel('T (s)')
plt.ylabel(r'$PSA\ (cm/s^2)$')
plt.xlim(0.03,2)
plt.ylim(0,1000)
plt.grid(which='both')
plt.savefig('sprectra.png',dpi=600,bbox_inches='tight',pad_inches=0.05)
plt.gcf().subplots_adjust(bottom=0.15)

from PIL import Image
image = Image.open('sprectra.png')
st.image(image)

def convert_df(df):
    return df.to_csv().encode('utf-8')
csv = convert_df(PSAs_df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='PSAs.csv',
    mime='text/csv',
)
