import pandas as pd
import pdfplumber

def parse(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        tables = []
        for page in pdf.pages:
            tables.extend(page.extract_tables())
        # Create a DataFrame from the tables
        df = pd.DataFrame()
        for table in tables:
            df_table = pd.DataFrame(table[1:], columns=table[0])
            df = pd.concat([df, df_table], ignore_index=True)
        # Convert the 'Debit Amt', 'Credit Amt', and 'Balance' columns to float
        df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce')
        df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce')
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
        return df