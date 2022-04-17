from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import streamlit as st
import joblib

#import matplotlib.pyplot as plt

# Standard scaler #
from sklearn.preprocessing import StandardScaler

# column transformer #
from sklearn.compose import make_column_transformer

# Train test split #
from sklearn.model_selection import train_test_split

# KNN and cross validation #
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Logistic Regression #
from sklearn.linear_model import LogisticRegression

# SVM (SVC) #
from sklearn.svm import SVC

# Random forest #
from sklearn.ensemble import RandomForestClassifier

# Building a Pipeline #
from sklearn.pipeline import make_pipeline

# Count text #
from sklearn.feature_extraction.text import CountVectorizer

st.title("Spam Detection App")


# FUNCTIONS TO GET MODEL OUTPUT #
# convert list to df #
def make_list_to_df(list_to_convert, cols):
    df = pd.DataFrame(columns=cols)
    df.loc[len(df)] = list_to_convert

    return df

# creating a vector from user input


def vectorize_data(data_entry, vocabulary):
    # make sure the voca is a list
    vocab = list(vocabulary)
    pipe_user_data = make_pipeline(CountVectorizer(vocabulary=vocab))
    new_email = pipe_user_data.fit_transform([data_entry]).toarray()
    return new_email[0]


# LOADING DATA #
email = pd.read_csv("emails.csv")
# target and features #
X = email.drop(labels=["Email No.", "Prediction"], axis=1)
y = email["Prediction"]
# column transformer application #
column_trans = make_column_transformer(
    (StandardScaler(), X.columns)
)

column_trans.fit_transform(X)

# Train test split application #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# Model vocab #
vocab = email.columns[1:-1].tolist()

# Select algorithm #
# Using object notation
add_selectbox_algo = st.sidebar.selectbox(
    'Please select an algorithm',
    ('knn', 'Logistic Regression', 'SVM', 'Random Forests')
)
#st.write('You selected:', add_selectbox_algo)
if add_selectbox_algo == "knn":
    knn_value = st.sidebar.slider('Select a K value', 0, 15)
    knn = KNeighborsClassifier(knn_value)
    pipe_knn = make_pipeline(column_trans, knn)
    knn_acc = np.mean(cross_val_score(pipe_knn, X_train,
                      y_train, scoring='accuracy', cv=3))

    st.write("knn model accuracy: ", knn_acc)

elif add_selectbox_algo == "Logistic Regression":
    # Logistic Regression #
    lg = LogisticRegression()

    pipe_lg = make_pipeline(column_trans, lg)

    lg_acc = np.mean(cross_val_score(pipe_lg, X_train,
                     y_train, scoring="accuracy", cv=3))
    st.write("Logistic Regression model accuracy: ", lg_acc)

elif add_selectbox_algo == "SVM":
    #param_grid = {"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]}
    C_value = st.sidebar.selectbox(
        'Please select C value (c=10)',
        (0.1, 1, 10,)
    )
    gamma_value = st.sidebar.selectbox(
        'Please select gamma value (gamma="scale")',
        ("scale", "auto")
    )
    # best params: c=10 , gamma={'scale', 'auto'} #

    svm = SVC(C=C_value, gamma=gamma_value)
    pipe_svm = make_pipeline(column_trans, svm)

    svm_acc = np.mean(cross_val_score(pipe_svm, X_train,
                                      y_train, scoring='accuracy', cv=3))

    st.write("SVM model accuracy: ", svm_acc)

elif add_selectbox_algo == "Random Forests":
    # YET TO ADD HYPER PERAMETER TUNNİNG
    bootstrap_val = st.sidebar.selectbox(
        "Please select a bootstrap value",
        (True, False)
    )

    max_features_val = st.sidebar.selectbox(
        "Please select a max_features value",
        ("auto", "sqrt")
    )
    min_samples_leaf_val = st.sidebar.selectbox(
        "Please select a min_samples_leaf value",
        (1, 2, 4)
    )
    min_samples_split_val = st.sidebar.selectbox(
        "Please select a min_samples_split value",
        (2, 5, 10)
    )
    n_estimators_val = st.sidebar.selectbox(
        "Please select a n_estimators value",
        (200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000)
    )

    rfc_params = {
        'bootstrap': bootstrap_val,
        'max_features': max_features_val,
        'min_samples_leaf': min_samples_leaf_val,
        'min_samples_split': min_samples_split_val,
        'n_estimators': n_estimators_val
    }
    #rfc = RandomForestClassifier(rfc_params)
    rfc = RandomForestClassifier(n_estimators=200)
    pipe_rfc = make_pipeline(column_trans, rfc)

    rfc_acc = np.mean(cross_val_score(pipe_rfc, X_train,
                      y_train, scoring='accuracy', cv=3))

    st.write("Random Forests model accuracy: ", rfc_acc)

    # Form #
with st.form("my_form"):

    #submitted = st.form_submit_button("Submit")
    # if submitted:
    new_email = st.text_input('Please add email')
    submitted = st.form_submit_button("Submit")

    new_email_vector = vectorize_data(new_email, vocab)
    new_email_df = make_list_to_df(new_email_vector, vocab)
    if add_selectbox_algo == "knn":
        # do prediction using knn #

        if submitted:

            knn_model = pipe_knn.fit(X_train, y_train)

            result = knn_model.predict(new_email_df)
            if result:
                st.write(" ### The email is: Spam")
            else:
                st.write(" ### The email is: Ham")
    elif add_selectbox_algo == "Logistic Regression":

        if submitted:
            lg_model = pipe_lg.fit(X_train, y_train)
            result = lg_model.predict(new_email_df)
            if result:
                st.write(" ### The email is: Spam")
            else:
                st.write(" ### The email is: Ham")
    elif add_selectbox_algo == "SVM":

        if submitted:
            svm_model = pipe_svm.fit(X_train, y_train)
            result = svm_model.predict(new_email_df)
            if result:
                st.write(" ### The email is: Spam")
            else:
                st.write(" ### The email is: Ham")

    elif add_selectbox_algo == "Random Forests":
        if submitted:
            rfc_model = pipe_rfc.fit(X_train, y_train)
            result = rfc_model.predict(new_email_df)
            if result:
                st.write(" ### The email is: Spam")
            else:
                st.write(" ### The email is: Ham")

    # st.write("submitted!!")
