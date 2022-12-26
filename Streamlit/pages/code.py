import streamlit as st
st.code("""
import streamlit as st
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

# Data
dataset = load_breast_cancer()
# creating dataframe for better visualization
data = pd.DataFrame(dataset.data)
data['target'] = dataset.target

# Independent(X) and dependent Features(y)
X = data.drop('target',axis=1)
y = data['target']


#decomposition of columns to visualise data
pca_model = PCA(2)
data_new = pca_model.fit_transform(X)


st.set_page_config(page_title='')
st.write("# Breast Cancer Prediction")
st.write('___')
st.write('##### Dataset')
st.dataframe(data.head(5))

with st.sidebar:
    st.write('# Basic Data INFO')
    st.write('___')
    st.write("#### Target Labels")
    st.write("1 - WDBC-Malignant")
    st.write("2 - WDBC-Benign")
    st.write("Select Penalty of the error")
    st.warning("Higher the C value,Greater the chances of overfitting")
    C = st.slider(label="C",min_value= 1,max_value= 10,step=1)


# UnScaled Data
# ML Model for Unscaled Data
model = SVC(C=C)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1024)
st.title("Support Vector Machine(Classifier)")
st.write("***")
model.fit(X_train,y_train)
st.write("## Without scaling the Data")
st.success(f"Training Score = {model.score(X_train,y_train)}")
y_pred = model.predict(X_test)
st.success(f"Test Data Score = {accuracy_score(y_test,y_pred)}")
fig,axs_sc = plt.subplots(nrows=1,ncols=2,figsize=(16,6))
axs_sc[0].scatter(data_new[:,0],data_new[:,1],c=y)
axs_sc[0].set_title("Decomposed Data(30 columns into 2)",fontdict={'fontsize':14},pad=40)
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,ax = axs_sc[1],cmap='Reds')
axs_sc[1].set_title("Confusion matrix(True values to Predicted Values)",fontdict={'fontsize':14},pad=40)
st.pyplot(fig)
st.write('---')


# After scaling
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)
x_train_sc,x_test_sc,y_train_sc,y_test_sc = train_test_split(x_scaled,y,test_size=0.3,random_state=1024)
model2 = SVC(C=C)
model2.fit(x_train_sc,y_train_sc)
st.write("## After scaling the Data")
st.success(f"Training Score = {model2.score(x_train_sc,y_train_sc)}")
y_pred_sc = model2.predict(x_test_sc)
st.success(f"Test Data Score = {accuracy_score(y_test_sc,y_pred_sc)}")
fig_sc,axs_sc = plt.subplots(nrows=1,ncols=2,figsize=(16,6))
axs_sc[0].scatter(data_new[:,0],data_new[:,1],c=y)
axs_sc[0].set_title("Decomposed Data(30 columns into 2)",fontdict={'fontsize':14},pad=40)
sns.heatmap(confusion_matrix(y_test_sc,y_pred_sc),annot=True,ax = axs_sc[1],cmap='Reds')
axs_sc[1].set_title("Confusion matrix(True values to Predicted Values)",fontdict={'fontsize':14},pad=40)

st.pyplot(fig_sc)



""")