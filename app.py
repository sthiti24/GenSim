import streamlit as st
import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

def main():
    st.title("Synthetic Data Generator")
    st.write("Upload a CSV file and select the number of synthetic records to generate.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        real_data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(real_data.head())

        sensitive_columns = st.multiselect(
            "Select sensitive columns to be protected",
            options=real_data.columns.tolist()
        )

        n_records = st.number_input("Number of synthetic records to generate", min_value=1, value=100)

        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                synthcity = Plugins()
                syn_model = synthcity.get("dpgan", n_iter=10)
                loader = GenericDataLoader(real_data, sensitive_columns=sensitive_columns)
                syn_model.fit(loader)
                synthetic_data = syn_model.generate(count=n_records).dataframe()

                st.write("Synthetic Data Preview:")
                st.write(synthetic_data.head())

                csv = synthetic_data.to_csv(index=False)
                st.download_button(
                    label="Download synthetic data as CSV",
                    data=csv,
                    file_name='synthetic_data.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()
