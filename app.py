import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

df = load_data()

def main():
    # Page selection
    pages = ["Home", "Data Exploration", "Prediction"]
    page = st.sidebar.selectbox("Choose a page", pages)
    if page == "Home":
        st.subheader("auteur : Thiara kanteye")
        st.title("Contexte du projet")
        st.write("Le projet de jeu de données Iris vise à fournir une expérience d'introduction "
                 "à l'analyse de données et à l'apprentissage automatique en utilisant un jeu de données "
                 "bien connu et simple. Le jeu de données Iris contient des mesures de quatre caractéristiques"
                 "(longueur des sépales, largeur des sépales, longueur des pétales et largeur des pétales) pour"
                 "150 fleurs d'iris de trois espèces différentes (setosa, versicolor et virginica). "
        )
        st.image('iris.webp')

    elif page == "Data Exploration":
        st.write("### Data Exploration")

        # Display dataset
        st.write("#### Dataset")
        st.write(df.head())

        # Display data summary
        st.write("#### Data Summary")
        st.write(df.describe())

        # Display data info
        st.write("#### Data Information")
        if st.checkbox("Afficher les valeurs manquantes") : 
            st.dataframe(df.isna().sum())
        
        if st.checkbox("Afficher les doublons") : 
            st.write(df.duplicated().sum())
        

        # Correlation heatmap
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

        # Pairplot
        st.write("#### Pairplot")
        fig = sns.pairplot(df, hue='target', diag_kind='kde')
        st.pyplot(fig)

        # Interactive Boxplot
        st.write("#### Boxplot")
        fig = px.box(df, y=df.columns[:-1])
        st.plotly_chart(fig)


    elif page == "Prediction":
        st.write("### Make a Prediction")

        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train models
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        log_reg_pred = log_reg.predict(X_test)
        log_reg_acc = accuracy_score(y_test, log_reg_pred)

        tree_clf = DecisionTreeClassifier()
        tree_clf.fit(X_train, y_train)
        tree_pred = tree_clf.predict(X_test)
        tree_acc = accuracy_score(y_test, tree_pred)

        forest_clf = RandomForestClassifier()
        forest_clf.fit(X_train, y_train)
        forest_pred = forest_clf.predict(X_test)
        forest_acc = accuracy_score(y_test, forest_pred)

        # Streamlit app
        st.title("Machine Learning Models")
        model_choice = st.sidebar.selectbox("Choose Model", ("Logistic Regression", "Decision Tree", "Random Forest"))

        if model_choice == "Logistic Regression":
            st.write("### Logistic Regression Model")
            st.write(f"Accuracy: {log_reg_acc:.2f}")
        elif model_choice == "Decision Tree":
            st.write("### Decision Tree Model")
            st.write(f"Accuracy: {tree_acc:.2f}")
        elif model_choice == "Random Forest":
            st.write("### Random Forest Model")
            st.write(f"Accuracy: {forest_acc:.2f}")

        # Display feature importances for tree-based models
        if model_choice in ["Decision Tree", "Random Forest"]:
            if model_choice == "Decision Tree":
                feature_importances = tree_clf.feature_importances_
            else:
                feature_importances = forest_clf.feature_importances_

            importance_df = pd.DataFrame({
                'Feature': df.columns[:-1],
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            st.write("### Feature Importances")
            st.write(importance_df)

        # Allow user to make predictions
        st.write("### Make a Prediction")
        input_data = []
        for feature in df.columns[:-1]:
            value = st.number_input(f"Input {feature}", value=0.0)
            input_data.append(value)

        input_data = scaler.transform([input_data])

        if st.button("Predict"):
            if model_choice == "Logistic Regression":
                prediction = log_reg.predict(input_data)[0]
            elif model_choice == "Decision Tree":
                prediction = tree_clf.predict(input_data)[0]
            else:
                prediction = forest_clf.predict(input_data)[0]
            
            st.write(f"Prediction: {prediction}")
            
    
    
    
if __name__ == "__main__":
    main()


#mon apli streamlit