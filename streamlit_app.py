import streamlit as st
from datetime import date
import pandas as pd
from prophet import Prophet 
from prophet.plot import plot_plotly 
from plotly import graph_objs as go
import numpy as np
import plotly.express as px

# Load data from dataset
def loadData(dataset):
    df = pd.read_csv(f'datasets/{dataset}')
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    #df['date'] = pd.to_datetime(df['date'], format= "%Y/%m/%d")
    return df

# Plot data
def plotData(df):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=df['date'], y=df['family']))
    figure.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

# Clean data in dataset



# Train data using train dataset


# Test data using test dataset


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.sidebar.header("Settings")
    setting = st.sidebar.selectbox("Setting", ("Raw Data", "Forecasting"))

    if setting == "Raw Data":
        st.title("Raw Data")
        df = loadData("train.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.strftime('%Y/%m/%d')
        categories = np.insert(df['family'].unique(), 0, "ALL")
        selectedCategory = st.selectbox("Select Category", categories)
        totalSalesDf = df.pivot(index=['id', 'date'], columns='family', values='sales')
        if selectedCategory == "ALL":
            totalSalesDf = totalSalesDf.groupby('date'[:10]).sum()
            st.write(totalSalesDf.head(15))
            #figure = go.Figure()
            #figure.add_trace(go.Scatter(x='date', y='family'))
            #figure.layout.update(xaxis_rangeslider_visible=True)
            
            #st.plotly_chart(x=[1,2,3], y=[3,4,5])
        else:
            totalSalesCatDf = totalSalesDf[selectedCategory].groupby('date'[:10]).sum()
            st.write(totalSalesCatDf.head(15))

            #figure = go.Figure()
            #figure.add_trace(go.Scatter(x=df['date'], y=df['family']))
            #figure.layout.update(xaxis_rangeslider_visible=True)
            #st.plotly_chart(totalSalesCatDf[selectedCategory])
            
            #fig2 = px.line(df, x=totalSalesCatDf['sales'], y=totalSalesCatDf['date'], height=500, width=1000, template="gridon")
            #st.plotly_chart(fig2, use_container_width=True)

    else:
        st.title("Retail Store Inventory and Demand Forecasting")
