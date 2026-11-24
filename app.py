import streamlit as st
import pandas as pd
import pickle

# Hide Streamlit branding and toolbar
st.markdown("""
<style>

/* Hide top header */
header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0%;
}

/* Hide toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide deploy button */
[data-testid="stDeployButton"] {
    display: none !important;
}

/* Hide main menu */
#MainMenu {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Hide decoration */
[data-testid="stDecoration"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


# Load datasets
df_biased = pd.read_csv(
    "datasets/biased_recruitment_dataset.csv"
)

df_balanced = pd.read_csv(
    "datasets/balanced_fair_recruitment_dataset.csv"
)

# Page configuration
st.set_page_config(
    page_title="AI Bias Demonstration",
    page_icon="⚖️",
    layout="wide"
)

/* Hide Streamlit status widget */
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Hide bottom-right floating controls */
[data-testid="stBottom"] {
    display: none !important;
}

/* Hide Streamlit footer */
footer {
    display: none !important;
}



# Load trained models
with open("model/biased_model.pkl", "rb") as f:
    biased_model, biased_columns = pickle.load(f)

with open("model/balanced_model.pkl", "rb") as f:
    balanced_model, balanced_columns = pickle.load(f)


# Application title
st.title("⚖️ AI Bias Demonstration")
st.write(
    "Explore how training data can influence "
    "Machine Learning predictions."
)

st.divider()

# Candidate input
st.subheader("Candidate Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    experience = st.slider(
        "Work Experience (Years)",
        0, 10, 3
    )

with col2:
    aptitude = st.slider(
        "Aptitude Score",
        0, 100, 85
    )

    communication = st.slider(
        "Communication Score",
        0, 100, 85
    )


# Prepare input data
input_data = pd.DataFrame([{
    "Gender": gender,
    "Experience_Years": experience,
    "Aptitude_Score": aptitude,
    "Communication_Score": communication
}])

st.divider()

# Prediction button
if st.button("Predict Candidate Selection", type="primary"):

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        X_biased = pd.get_dummies(
            input_data,
            columns=["Gender"],
            dtype=int
        ).reindex(columns=biased_columns, fill_value=0)

        prediction1 = biased_model.predict(X_biased)[0]

        st.markdown("### Model 1: Biased Dataset")

        if prediction1 == "Yes":
            st.success("Candidate Selected")
        else:
            st.error("Candidate Rejected")

    with col2:
        X_balanced = pd.get_dummies(
            input_data,
            columns=["Gender"],
            dtype=int
        ).reindex(columns=balanced_columns, fill_value=0)

        prediction2 = balanced_model.predict(X_balanced)[0]

        st.markdown("### Model 2: Balanced Dataset")

        if prediction2 == "Yes":
            st.success("Candidate Selected")
        else:
            st.error("Candidate Rejected")

    st.divider()

    if prediction1 != prediction2:
        st.warning(
            "Different predictions! "
            "The training data influenced the model decisions."
        )
    else:
        st.info(
            "Both models produced the same prediction "
            "for this candidate."
        )


st.divider()

st.header("📊 Explore Training Datasets")

tab1, tab2 = st.tabs([
    "Biased Dataset",
    "Balanced Dataset"
])

with tab1:
    st.subheader("Intentionally Biased Dataset")

    st.write("Total Records:", len(df_biased))

    st.dataframe(
        df_biased,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Biased Dataset",
        df_biased.to_csv(index=False),
        "biased_recruitment_dataset.csv",
        "text/csv"
    )

with tab2:
    st.subheader("Balanced Dataset")

    st.write("Total Records:", len(df_balanced))

    st.dataframe(
        df_balanced,
        use_container_width=True,
        hide_index=True
    )
