import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

st.title("Ml classification by Sourav")

st.write("""
# Explore different classifier
Which one is the best
""")

dataset_name = st.sidebar.selectbox("Select Dataset:-", ("Iris", "Brest Cancer", "Wine Dataset","Digit", "Diabetes"))
st.sidebar.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier:-", ("KNN", "SVM", "Random Forest","Decision Tree", "Logistic Regression", "Multinomial Naive Bayes","Gaussian naive bayes", "Bernoulli's naive bayes"))


def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data = datasets.load_iris()
    elif dataset_name=="Brest Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name=="Wine Dataset":
        data = datasets.load_wine()
    elif dataset_name=="Digit":
        data = datasets.load_digits()
    else:
        data = datasets.load_diabetes()


    X = data.data
    y = data.target
    return X,y

X,y = get_dataset(dataset_name)
st.write("Shape of the dataset", X.shape)
st.write("Number of Classes", len(np.unique(y)))


def add_parameter(clf_name):
    param =dict()
    if clf_name=="KNN":
        K = st.sidebar.slider("K", 1, 15)
        param["K"]=K
    elif clf_name=="SVM":
        C = st.sidebar.slider("C", 0.01, 10.00)
        kernel = st.sidebar.selectbox("Kernel",('linear', 'poly', 'rbf'))
        st.sidebar.write("Option Selected :-", kernel)
        param["kernel"] = kernel
        param["C"] = C
    elif clf_name=="Decision Tree":
        criteria = st.sidebar.selectbox("Criteria",("gini",'entropy','log loss'))
        st.sidebar.write("Option Selected :-", criteria)
        param['criteria'] = criteria
        param["splitter"] = st.sidebar.selectbox("Splitter",('best','random'))
        st.sidebar.write("Option Selected :-",param['splitter'])
        param["max depth"] = st.sidebar.slider("max_depth", 2,50)
    elif clf_name=="Multinomial Naive Bayes":
        param["alpha"] = st.sidebar.slider("Alpha",min_value=0.0,max_value=1.0)
    elif clf_name=="Bernoulli's naive bayes":
        param["alpha"] = st.sidebar.slider("Alpha",min_value=0.0,max_value=1.0)
    elif clf_name=="Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 2,100)
        param["n_estimators"] = n_estimators
        max_depth = st.sidebar.slider("max_depth", 2,15)
        param["max_depth"] = max_depth
    return param


param = add_parameter(classifier_name)

def get_classifier(clf_name,param):
    if clf_name=="KNN":
        clf = KNeighborsClassifier(n_neighbors=param["K"])
    elif clf_name=="SVM":
        clf = SVC(C=param["C"], kernel=param["kernel"])
    elif clf_name=="Decision Tree":
        clf=DecisionTreeClassifier(criterion=param['criteria'],splitter=param["splitter"],max_depth=param["max depth"])
    elif clf_name=="Multinomial Naive Bayes":
        clf=MultinomialNB(alpha=param['alpha'])
    elif clf_name=="Bernoulli's naive bayes":
        clf = BernoulliNB(alpha=param["alpha"])
    elif clf_name=="Gaussian naive bayes":
        clf = GaussianNB()
    elif clf_name=="Logistic Regression":
        clf =LogisticRegression(multi_class='ovr')
    else:
        clf = RandomForestClassifier(n_estimators=param["n_estimators"], max_depth=param["max_depth"],random_state=42)
    return clf

clf = get_classifier(classifier_name, param)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)
pca = PCA(n_components=2)
X_plot = pca.fit_transform(X)
x1 = X_plot[:,0]
x2 = X_plot[:,1]
fig,ax = plt.subplots()
ax.scatter(x1,x2,c = y, alpha=0.8, cmap = 'viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

st.pyplot(fig)
#Classification
if st.sidebar.button("Classify"):
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
        st.subheader(f"Accuracy score for {classifier_name} is :-")
        st.write(accuracy_score(y_test,y_pred))
        st.subheader(f"Confusionn Matrix for {classifier_name} is :-")
        st.write(confusion_matrix(y_test,y_pred))
        st.subheader("The Heatmap is :-")
        fig,ax = plt.subplots()
        ax =sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
        st.pyplot(fig)
        if classifier_name=="Decision Tree":
            fig,ax =plt.subplots()
            ax = plot_tree(clf)
            st.pyplot(fig)