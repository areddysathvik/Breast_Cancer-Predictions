import streamlit as st
import os

st.title('Downlad Full Project')
with open('areddysathvik/mini_Cancer-Predictions-/Streamlit/pages/BCP.zip','rb') as f:
    c = st.download_button('Download Zip', f, file_name='archive.zip')

if(c):
    st.success('Thanks For Downloading....')
    st.snow()