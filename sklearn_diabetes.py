############ Import the required Libraries ##################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    st.title('Diabetes models hyper-parameter tuning Web App')
    st.markdown('Pick the classifier you prefer, setup the parameter for it')
    st.markdown('Select the metrics(multiple) and click "classify". It will plot the figure')
    
    st.subheader('Model Selection Panel')
    st.markdown("Choose your model and its parameters")
    #@st.cache(allow_output_mutation=True)
    @st.cache(persist=True)
    def load_data():# Function to load our dataset
        data = pd.read_csv("dataset/diabetes/diabetes.csv")
        return data

    @st.cache(persist=True)
    def split(df):# Split the data to 'train and test' sets
        req_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
        x = df[req_cols] # Features for our algorithm
        y = df.Outcome
        x = df.drop(columns=['Outcome'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot() 
    
    df=load_data() 
    class_names = ['Diabetec', 'Non-Diabetic']
    x_train, x_test, y_train, y_test = split(df)
    
    
    
    st.subheader("Select your Classifier")
    classifier = st.selectbox("Classifier", ("Decision Tree", "Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"))
    if classifier == 'Decision Tree':
        st.subheader("Model parameters")
        #choose parameters
        criterion= st.radio("Criterion(measures the quality of split)", ("gini", "entropy"), key='criterion')
        splitter = st.radio("Splitter (How to split at each node?)", ("best", "random"), key='splitter')
        metrics = st.multiselect("Select your metrics : ", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.subheader("Decision Tree Results")
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2)*100,"%")
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.subheader("Model Parameters")
        C = st.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.multiselect("Select your metrics?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2)*100,"%")
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Random Forest':
        st.subheader("Model Hyperparameters")
        n_estimators = st.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2)*100,"%")
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics) 
        if st.checkbox("Show raw data", False):
            st.subheader("Diabetes Raw Dataset")
            st.write(df) 

    if classifier == 'Support Vector Machine (SVM)':
        st.subheader("Model Hyperparameters")
        C = st.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.radio("Kernel",("rbf", "linear"), key='kernel')
        gamma = st.radio("Gamma (Kernel Coefficient", ("scale", "auto"), key = 'gamma')
        metrics = st.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))


        if st.button("Classfiy", key='classify'):
            st.subheader("Support Vector Machine (SVM Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
 
if __name__ == '__main__':
    main()