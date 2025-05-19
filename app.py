import os
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from insightforge_helpers import (
    df,
    advanced_summary,
    plot_product_category_sales,
    plot_sales_trend,
    plot_product_performance_over_time,
    plot_regional_sales,
    plot_customer_segmentation,
    agent_chain,
    generate_rag_insight,
    evaluate_model,
    model_monitor
)

# Streamlit App: InsightForge Business Intelligence Assistant
st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")
st.title("InsightForge: Business Intelligence Assistant 🤖💡")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "AI Assistant", "Model Performance"])

if page == "Home":
    st.header("Home")
    st.write("Welcome to InsightForge, your AI-powered Business Intelligence Assistant.")
    st.write("Use the sidebar to navigate through different sections of the application.")

elif page == "Data Analysis":
    st.header("Data Analysis")

    st.subheader("Sales Summary")
    summary_df = advanced_summary()
    st.dataframe(summary_df)

    st.subheader("Sales Distribution by Product Category")
    fig_cat = plot_product_category_sales()
    st.pyplot(fig_cat)

    st.subheader("Daily Sales Trend")
    fig_trend = plot_sales_trend()
    st.pyplot(fig_trend)

    st.subheader("Product Performance Over Time")
    fig_perf = plot_product_performance_over_time()
    st.pyplot(fig_perf)

    st.subheader("Sales by Region")
    fig_reg = plot_regional_sales()
    st.pyplot(fig_reg)

    st.subheader("Customer Segmentation")
    fig_seg = plot_customer_segmentation()
    st.pyplot(fig_seg)

elif page == "AI Assistant":
    st.header("AI Sales Analyst")

    ai_mode = st.radio("Choose AI Mode:", ["Standard", "RAG Insights"])
    user_input = st.text_input("Ask a question about the sales data:")
    if user_input:
        start_time = datetime.now()
        if ai_mode == "Standard":
            response = agent_chain.run(
                input=user_input,
                chat_history="",
                agent_scratchpad=""
            )
            st.subheader("AI Response")
            st.write(response)
        else:
            response = generate_rag_insight(user_input)
            st.subheader("RAG Insight")
            st.write(response)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        st.write(f"Execution time: {execution_time:.2f} seconds")
        model_monitor.log_interaction(user_input, execution_time)

elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Model Evaluation")
    if st.button("Run Model Evaluation"):
        qa_pairs = [
            {"question": "What is our total sales amount?", 
             "answer": f"The total sales amount is ${df['Sales'].sum():,.2f}."},
            {"question": "Which product category has the highest sales?", 
             "answer": f"The product category with the highest sales is {df.groupby('Product')['Sales'].sum().idxmax()}."},
            {"question": "What is our average customer satisfaction score?", 
             "answer": f"The average customer satisfaction score is {df['Customer_Satisfaction'].mean():.2f}."},
            # Add more question-answer pairs as needed
        ]
        eval_results = evaluate_model(qa_pairs)
        for r in eval_results:
            st.markdown(f"**Q:** {r['question']}")
            st.markdown(f"- **Predicted:** {r['predicted']}")
            st.markdown(f"- **Actual:** {r['actual']}")
            st.markdown(f"- **Correct:** {r['correct']}  \n---")
        accuracy = sum(r['correct'] for r in eval_results) / len(eval_results)
        st.write(f"**Model Accuracy:** {accuracy:.2%}")

    st.subheader("Execution Time Monitoring")
    if model_monitor.logs:
        timestamps = [datetime.fromisoformat(log['timestamp']) for log in model_monitor.logs]
        exec_times = [log['execution_time'] for log in model_monitor.logs]
        fig, ax = plt.subplots()
        ax.plot(timestamps, exec_times, marker='o')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Execution Time (s)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        avg_time = model_monitor.get_average_execution_time()
        st.write(f"Average Execution Time: {avg_time:.2f} seconds")
    else:
        st.write("No execution logs yet. Run an AI query or evaluation to begin logging.")
