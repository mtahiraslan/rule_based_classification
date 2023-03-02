import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Rules Based Classificiation on Customers Dataset",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
    )

image = Image.open('customer.jpg')

st.image(image, width=200)

st.title('Rule-Based Classification in a Customer Dataset')

st.markdown(
    """
    This app is to create new level-based customer definitions using some features of a game company's customers. 
    It creates segments according to these new customer definitions and estimates how much the new customers can earn according to these segments.
    
    * **Pyton libraries:** pandas, streamlit, PIL, matplotlib, plotly.express
    * **Data source:** persona.csv
    
    ***
    """
)

persona = 'RBC/persona.csv'
df = pd.read_csv(persona)


def check_dataframe(df, row_num=5):
    st.subheader("Shape of Dataset")
    st.write("No. of Rows:", df.shape[0], "No. of Columns:", df.shape[1])
    st.subheader("Types of Columns")
    st.write(df.dtypes)
    st.subheader("First 5 Rows")
    st.write(df.sample(row_num))
    st.subheader("Last 5 Rows")
    st.write(df.tail(row_num))
    st.subheader("No. of Null Values In The Dataset")
    st.write(pd.DataFrame(df.isnull().sum()))
    st.subheader("Summary Statistics of The Dataset")
    st.write(df.describe())


st.header('Basic Information of Dataset')
check_dataframe(df)


def category_summary(df, col_name):
    st.write(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))


for col in df.columns:
    st.header(f'{col.title()} Ratio')
    category_summary(df, col)


def grab_columns(df, categorical_th=10, cardinal_th=20):

    cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat_col = [col for col in df.columns if
                                      df[col].nunique() < categorical_th and df[col].dtypes in ["int64", "float64"]]
    cat_but_car_col = [col for col in df.columns if
                                     df[col].nunique() > cardinal_th and str(df[col].dtypes) in ["category", "object"]]
    cat_col = cat_col + num_but_cat_col
    cat_col = [col for col in cat_col if col not in cat_but_car_col]

    num_col = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_col = [col for col in num_col if col not in cat_col]

    st.header('Types of Columns in Dataset')
    st.write(f'No. of Categorical Columns: {len(cat_col)}')
    st.write(f'No. of Numerical Columns: {len(num_col)}')
    st.write(f'No. of Cardinal Columns: {len(cat_but_car_col)}')

    return cat_col, num_col, cat_but_car_col


cat_cols, num_cols, cat_but_car = grab_columns(df)

st.header("Categorical Variable Analysis")
st.subheader("Country")
fig = px.histogram(df, x="COUNTRY", color="COUNTRY", nbins=20)
st.plotly_chart(fig)
st.subheader("Source (OS)")
fig = px.histogram(df, x="SOURCE", color="SOURCE", nbins=20)
st.plotly_chart(fig)
st.subheader("Sex")
fig = px.histogram(df, x="SEX", color="SEX", nbins=20)
st.plotly_chart(fig)

st.header("Numeric Variable Analysis")
st.subheader("Age Distribution")
fig = px.histogram(df, x="AGE", nbins=20)
st.plotly_chart(fig)
st.subheader("Age and Price Distribution")
fig = px.scatter(df, x="AGE", y="PRICE", color="SEX")
st.plotly_chart(fig)

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values(
    "PRICE", axis=0, ascending=False)
agg_df = agg_df.reset_index()

st.subheader('Finding the average price by Country, Gender, Source, and Age')
st.write(agg_df)

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40, 70],
                           right=True, labels=["0_18", "19_23", "24_30", "31_40", "41_70"])
st.subheader('Converting age variable to categorical variable')
st.write(agg_df)

agg_df["CUSTOMERS_LEVEL_BASED"] = [('_'.join(x)).upper() for x in
                                   agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].values]
agg_df = agg_df.groupby("CUSTOMERS_LEVEL_BASED").agg({"PRICE": "mean"})
agg_df = agg_df.sort_values("PRICE", ascending=False)
st.subheader('Creating a customer_level_based column and finding the average price accordingly')
st.write(agg_df)

agg_df["SEGMENT"] = pd.qcut(agg_df['PRICE'], 4, ["D", "C", "B", "A"])
agg_df.reset_index(inplace=True)
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

st.subheader('Segmenting into four different segments according to the price column and determining the mean value according to the price')
st.write(agg_df)

st.sidebar.title("New Customer Information")
country = st.sidebar.selectbox("Select Country", sorted(df['COUNTRY'].str.upper().unique()))
source = st.sidebar.selectbox("Select Source (OS)", sorted(df['SOURCE'].str.upper().unique()))
sex = st.sidebar.selectbox("Select Gender", sorted(df['SEX'].str.upper().unique()))
age = st.sidebar.number_input("Enter Age", min_value=0, max_value=100, value=18)

if st.sidebar.button("Save"):
    # Save the selected filters to a dictionary
    filters = {"Country": country, "Source (OS)": source, "Gender": sex, "Age": age}
    # Print the filters and a success message
    st.sidebar.write("New user added!:", filters)
    st.sidebar.success("Data insertion saved successfully!")

    new_user_df = [[country, source, sex, age]]
    new_user_df = pd.DataFrame(new_user_df, columns=["COUNTRY", "SOURCE", "SEX", "AGE"])

    new_user_df["AGE_CAT"] = pd.cut(new_user_df["AGE"], [0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30','31_40', '41_70'])
    age_cat = new_user_df["AGE_CAT"][0]

    new_user = (country + "_" + source + "_" + sex + "_" + age_cat).upper()
    price = agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user].reset_index(drop=True)


    def new_customer(dataframe, new_user):
        st.subheader('Segment and price prediction:')
        st.info(f'New User Information: {new_user}')
        st.success("Mean Price for New Customer: " + str(format(price["PRICE"][0], ".2f")) + "$")
        st.success("Segment for New Customer: " + str(price["SEGMENT"][0]))


    new_customer(agg_df, new_user)



