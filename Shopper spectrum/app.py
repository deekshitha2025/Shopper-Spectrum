import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.sidebar.image(
    "https://img.icons8.com/fluency/240/shopping-cart-loaded.png",
    width=120
)
st.markdown("""
<style>

.main{
background-color:#F8F5FF;
}

section[data-testid="stSidebar"]{
background-color:#E8DDFE;
}

h1,h2,h3{
color:#5E548E;
}

.stButton>button{
background-color:#7B68EE;
color:white;
border-radius:10px;
height:45px;
font-size:16px;
}

.stMetric{
background:white;
padding:10px;
border-radius:10px;
box-shadow:2px 2px 10px lightgray;
}

div[data-testid="metric-container"]{
background:white;
padding:15px;
border-radius:15px;
box-shadow:2px 2px 8px #d3c4ff;
}

</style>
""",unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():

    df=pd.read_csv("online_retail.csv",encoding="latin1")

    df=df.dropna(subset=["CustomerID"])

    df=df[df["Quantity"]>0]

    df=df[df["UnitPrice"]>0]

    df=df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"]=df["Quantity"]*df["UnitPrice"]

    return df

df=load_data()

# ---------------- LOAD MODELS ----------------

kmeans=joblib.load("kmeans_model.pkl")

scaler=joblib.load("scaler.pkl")

similarity_df=joblib.load("similarity.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.title("🛒 Shopper Spectrum")

st.sidebar.markdown("## Customer Intelligence")

st.sidebar.write(
"Understand customer behaviour using Machine Learning."
)
st.sidebar.markdown("""
# 🛒 Shopper Spectrum

### Customer Intelligence Platform

Analyze customer purchasing behaviour using Machine Learning.

---

### 🎯 Objectives

✔ Customer Segmentation

✔ Product Recommendation

✔ Business Dashboard

✔ Sales Analytics

✔ Business Insights

✔ Customer Retention

✔ Inventory Optimization

✔ Marketing Strategy
""")
menu=st.sidebar.radio(
"Navigation",
[
"🏠 Home",
"📊 Business Dashboard",
"👥 Customer Segmentation",
"🛍 Product Recommendation",
"📈 Business Insights",
"🌍 Customer Analytics",
"📋 About Project",
"👨‍💻 About Developer"
]
)

st.sidebar.markdown("---")

st.sidebar.info("""
### 🎯 Project

Customer Segmentation

Product Recommendation

Business Analytics

Customer Intelligence

Sales Analysis

Machine Learning
""")

st.sidebar.markdown("---")

st.sidebar.success("""
### 💻 Technologies

Python

Pandas

NumPy

Scikit-Learn

Matplotlib

Streamlit

Joblib
""")

st.sidebar.markdown("---")

st.sidebar.warning("""
### 🤖 Machine Learning

✔ RFM Analysis

✔ KMeans Clustering

✔ Collaborative Filtering

✔ Cosine Similarity
""")

st.sidebar.markdown("---")

st.sidebar.write("""
### 📈 Business Goals

✔ Customer Retention

✔ Better Marketing

✔ Increase Revenue

✔ Inventory Planning

✔ Sales Forecasting

✔ Customer Satisfaction
""")

st.sidebar.markdown("---")

st.sidebar.caption("Developed using Streamlit")
st.sidebar.markdown("---")

st.sidebar.info("""
💡 **Business Quote**

"Understanding your customers is the first step toward growing your business."
""")
# =====================================================
# HOME
# =====================================================

if menu=="🏠 Home":

    st.title("🛒 Shopper Spectrum")

    st.subheader("Customer Intelligence & Product Recommendation System")

    st.markdown("---")

    left,right=st.columns([2,1])

    with left:

        st.markdown("""
### 📌 Project Overview

Shopper Spectrum is an E-Commerce Analytics application developed to identify customer purchasing behaviour.

The application performs

✔ Customer Segmentation

✔ Product Recommendation

✔ Sales Analytics

✔ Business Intelligence

using Machine Learning algorithms.

The recommendation engine uses Item-Based Collaborative Filtering while customer segmentation is performed using KMeans clustering.
""")

    with right:

        st.metric("Customers",df["CustomerID"].nunique())

        st.metric("Products",df["Description"].nunique())

        st.metric("Orders",df["InvoiceNo"].nunique())

        st.metric("Revenue",f"${df['TotalPrice'].sum():,.0f}")

    st.markdown("---")

    a,b,c=st.columns(3)

    a.success("""
### 👥 Customer Segmentation

• RFM Analysis

• KMeans Clustering

• High Value Customers

• At Risk Customers
""")

    b.info("""
### 🛍 Recommendation

• Similar Products

• Cosine Similarity

• Personalized Shopping

• Cross Selling
""")

    c.warning("""
### 📊 Business Dashboard

• KPI Cards

• Revenue

• Top Countries

• Monthly Sales
""")

# =====================================================
# BUSINESS DASHBOARD
# =====================================================

elif menu=="📊 Business Dashboard":

    st.title("📊 Business Dashboard")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Customers",df["CustomerID"].nunique())

    c2.metric("Products",df["Description"].nunique())

    c3.metric("Orders",df["InvoiceNo"].nunique())

    c4.metric("Revenue",f"${df['TotalPrice'].sum():,.0f}")

    st.markdown("---")

    left,right=st.columns(2)

    with left:

        st.subheader("🌍 Top 5 Countries")

        country=df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head()

        fig,ax=plt.subplots(figsize=(6,6))

        colors=["#CDB4DB","#FFC8DD","#BDE0FE","#A2D2FF","#D8B4F8"]

        ax.pie(country,
               labels=country.index,
               autopct="%1.1f%%",
               colors=colors,
               startangle=90)

        st.pyplot(fig)

    with right:

        st.subheader("🏆 Top Selling Products")

        product=df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)

        fig,ax=plt.subplots(figsize=(8,5))

        ax.barh(product.index,product.values,color="#B39DDB")

        ax.invert_yaxis()

        st.pyplot(fig)

    st.markdown("---")

    st.subheader("📈 Monthly Revenue")

    monthly=df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()

    fig,ax=plt.subplots(figsize=(12,5))

    ax.plot(monthly.index.astype(str),
            monthly.values,
            marker="o",
            linewidth=3,
            color="#7B68EE")

    plt.xticks(rotation=45)

    st.pyplot(fig)
    # =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif menu=="👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.markdown("Predict the customer segment using **Recency, Frequency and Monetary (RFM)** values.")

    st.markdown("---")

    c1,c2,c3=st.columns(3)

    with c1:
        recency=st.number_input(
            "📅 Recency (Days)",
            min_value=0,
            value=30
        )

    with c2:
        frequency=st.number_input(
            "🛒 Frequency",
            min_value=0,
            value=5
        )

    with c3:
        monetary=st.number_input(
            "💰 Monetary",
            min_value=0.0,
            value=1000.0
        )

    st.markdown("")

    if st.button("Predict Customer Segment"):

        scaled=scaler.transform([[recency,frequency,monetary]])

        cluster=kmeans.predict(scaled)[0]

        labels={
            0:"Occasional",
            1:"At-Risk",
            2:"High-Value",
            3:"Regular"
        }

        segment=labels[cluster]

        st.success(f"Customer Segment : {segment}")

        if segment=="High-Value":

            st.info("""
### 🏆 High-Value Customer

✔ Recent Purchases

✔ Frequent Purchases

✔ Highest Spending

**Business Recommendation**

Reward these customers with VIP offers, loyalty points and exclusive discounts.
""")

        elif segment=="Regular":

            st.info("""
### ⭐ Regular Customer

✔ Good Purchase History

✔ Moderate Spending

✔ Loyal Customer

**Business Recommendation**

Provide seasonal offers and personalized product recommendations.
""")

        elif segment=="Occasional":

            st.warning("""
### 🛍 Occasional Customer

✔ Purchases Sometimes

✔ Average Spending

✔ Needs Engagement

**Business Recommendation**

Offer coupons and email campaigns to encourage repeat purchases.
""")

        else:

            st.error("""
### ⚠ At-Risk Customer

✔ Long Time Since Last Purchase

✔ Low Frequency

✔ Low Spending

**Business Recommendation**

Run retention campaigns with discounts and personalized communication.
""")

    st.markdown("---")

    st.subheader("📊 Customer Segment Overview")

    segment_table=pd.DataFrame({

        "Customer Segment":[
            "High-Value",
            "Regular",
            "Occasional",
            "At-Risk"
        ],

        "Description":[
            "Frequent & High Spending",
            "Steady Purchasers",
            "Occasional Buyers",
            "Inactive Customers"
        ]

    })

    st.table(segment_table)
    # =====================================================
# PRODUCT RECOMMENDATION
# =====================================================

elif menu=="🛍 Product Recommendation":

    st.title("🛍 Product Recommendation")

    st.write("Find similar products using Item-Based Collaborative Filtering.")

    st.markdown("---")

    product=st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):

        if product in similarity_df.index:

            recommendations=similarity_df[product].sort_values(
                ascending=False
            )[1:6]

            st.success("Top 5 Recommended Products")

            for item in recommendations.index:

                st.markdown(f"""
<div style="
background:#EFE7FF;
padding:15px;
border-radius:10px;
margin-bottom:10px;
border-left:6px solid #7B68EE;
">

### ⭐ {item}

</div>
""",unsafe_allow_html=True)

        else:

            st.error("Product not found.")

    st.markdown("---")

    st.subheader("💡 Recommendation Benefits")

    col1,col2=st.columns(2)

    with col1:

        st.success("""
### Business Benefits

✔ Increase Cross Selling

✔ Improve Customer Experience

✔ Personalized Shopping

✔ Better Sales
""")

    with col2:

        st.info("""
### Machine Learning

Recommendation Technique

✔ Item-Based Collaborative Filtering

Similarity Measure

✔ Cosine Similarity
""")
# =====================================================
# BUSINESS INSIGHTS
# =====================================================

elif menu=="📈 Business Insights":

    st.title("📈 Business Insights")

    st.markdown("Business recommendations generated from customer purchasing behaviour.")

    st.markdown("---")

    revenue=df["TotalPrice"].sum()

    avg_order=df.groupby("InvoiceNo")["TotalPrice"].sum().mean()

    top_country=df.groupby("Country")["TotalPrice"].sum().idxmax()

    top_product=df.groupby("Description")["Quantity"].sum().idxmax()

    col1,col2=st.columns(2)

    with col1:

        st.metric("💰 Total Revenue",f"${revenue:,.2f}")

        st.metric("🌍 Top Country",top_country)

    with col2:

        st.metric("🛒 Average Order Value",f"${avg_order:.2f}")

        st.metric("🏆 Best Selling Product",top_product)

    st.markdown("---")

    st.success("""
### 📌 Key Business Insights

✔ Focus marketing campaigns on High-Value customers.

✔ Re-engage At-Risk customers with personalized offers.

✔ Recommend similar products to improve cross-selling.

✔ Maintain sufficient inventory for top-selling products.

✔ Launch loyalty programs to improve customer retention.

✔ Use customer segmentation for targeted promotions.
""")

    st.warning("""
### 📈 Suggested Business Actions

• Offer festive discounts

• Send personalized emails

• Reward repeat customers

• Recommend complementary products

• Optimize inventory levels

• Monitor inactive customers
""")
# =====================================================
# CUSTOMER ANALYTICS
# =====================================================

elif menu=="🌍 Customer Analytics":

    st.title("🌍 Customer Analytics")

    st.markdown("---")

    st.subheader("Top 10 Countries by Revenue")

    country=df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)

    fig,ax=plt.subplots(figsize=(9,5))

    ax.bar(country.index,country.values,color="#CDB4DB")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Top 10 Products")

    products=df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)

    fig,ax=plt.subplots(figsize=(10,5))

    ax.barh(products.index,products.values,color="#A2D2FF")

    ax.invert_yaxis()

    st.pyplot(fig)

    st.markdown("---")

    st.info("""
Customer Analytics helps businesses understand:

✔ Customer purchasing behaviour

✔ Country-wise sales

✔ Product demand

✔ Sales trends

✔ Revenue contribution
""")
# =====================================================
# ABOUT PROJECT
# =====================================================

elif menu=="📋 About Project":

    st.title("📋 About Shopper Spectrum")

    st.markdown("""
## 🛒 Shopper Spectrum

### Customer Segmentation and Product Recommendation in E-Commerce

This project analyzes customer purchasing behaviour using transaction data from an online retail store.

The application performs:

✔ Customer Segmentation using RFM Analysis

✔ K-Means Clustering

✔ Product Recommendation using Collaborative Filtering

✔ Business Dashboard

✔ Business Insights

✔ Customer Analytics

---

### Technologies Used

• Python

• Pandas

• NumPy

• Matplotlib

• Scikit-Learn

• Streamlit

• Joblib

---

### Machine Learning

✔ K-Means Clustering

✔ Collaborative Filtering

✔ Cosine Similarity

✔ StandardScaler

✔ RFM Analysis
""")
# =====================================================
# ABOUT DEVELOPER
# =====================================================

elif menu=="👨‍💻 About Developer":

    st.title("👨‍💻 About Developer")

    st.markdown("""
## Developer Information

**Project Name**

🛒 Shopper Spectrum

Customer Segmentation & Product Recommendation

---

### Skills Demonstrated

✔ Python Programming

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Customer Segmentation

✔ K-Means Clustering

✔ Recommendation System

✔ Streamlit Development

✔ Business Intelligence

---

### Project Objective

To help businesses understand customer purchasing behaviour, identify valuable customer segments, and recommend products using Machine Learning techniques.

---

### Thank You!

Thank you for exploring the Shopper Spectrum project.
""")

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
font-size:18px;
color:#6A5ACD;
padding:15px;'>

Shopper Spectrum 

Customer Intelligence | Business Analytics | Machine Learning

</div>
""",
unsafe_allow_html=True)