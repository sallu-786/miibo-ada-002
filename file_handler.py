#This file is for handling different types of file formats such as pdf,ppt,txt,docx etc
import pandas as pd
def get_excel_text(file):
    df = pd.read_excel(file)
    rows = []
    for index, row in df.iterrows():
        row_text = " ".join(str(cell) for cell in row)  # Join all cells in the row
        rows.append((row_text, index + 2))  # store text with row number (1-based index)
    return rows

def get_csv_text(file):
    df = pd.read_csv(file)
    rows = []
    for index, row in df.iterrows():
        row_text = row.to_string(index=False)
        rows.append((row_text, index + 2))  # store text with row number (1-based index)
    return rows
