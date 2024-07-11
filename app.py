import streamlit as st
import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    """Preprocess the data by encoding non-numeric columns."""
    # Identify non-numeric columns
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    
    # Initialize a label encoder
    le = LabelEncoder()
    
    # Apply label encoding to non-numeric columns
    for column in non_numeric_columns:
        data[column] = le.fit_transform(data[column])
        
    return data

def main():
    st.title("Synthetic Data Generator")
    st.write("Upload a CSV file and select the number of synthetic records to generate.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            real_data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(real_data.head())

            # Preprocess the data
            processed_data = preprocess_data(real_data.copy())

            # Select sensitive columns (placeholder for future use, not used in CTGAN)
            sensitive_columns = st.multiselect(
                "Select sensitive columns to be protected",
                options=real_data.columns.tolist()
            )

            # Number of synthetic records to generate
            n_records = st.number_input("Number of synthetic records to generate", min_value=1, value=100)

            if st.button("Generate Synthetic Data"):
                with st.spinner("Generating synthetic data..."):
                    try:
                        # Initialize CTGAN
                        ctgan = CTGAN()
                        
                        # Train CTGAN
                        ctgan.fit(processed_data, epochs=5)
                        
                        # Generate synthetic data
                        synthetic_data = ctgan.sample(n_records)
                        
                        st.write("Synthetic Data Preview:")
                        st.write(synthetic_data.head())

                        csv = synthetic_data.to_csv(index=False)
                        st.download_button(
                            label="Download synthetic data as CSV",
                            data=csv,
                            file_name='synthetic_data.csv',
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Error during synthetic data generation: {e}")
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")

if __name__ == "__main__":
    main()
