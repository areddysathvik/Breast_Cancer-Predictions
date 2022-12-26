import streamlit as st
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
st.set_page_config(page_title="About the data")

st.write(' # DATA DESCRIPTION')
st.write('___')
st.write(dataset.DESCR)