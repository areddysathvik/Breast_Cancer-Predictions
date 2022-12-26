import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer

st.set_page_config(layout='wide')
dataset = load_breast_cancer()
data = pd.DataFrame(dataset.data)
st.title('Correlation Matrix of all Independent features')
fig,axs = plt.subplots(figsize=(22,15))
sns.heatmap(data.corr(),ax=axs,cmap='Greens')
st.pyplot(fig)
st.write('---')
st.title('General Information')
st.table(data.describe())


st.write('---')


