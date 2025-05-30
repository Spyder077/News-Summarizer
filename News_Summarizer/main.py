import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Prompt template
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

# Create LLMChain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# News extraction function
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        return f"⚠️ Failed to fetch news: {e}"

# Summarization function
def summarize_news(url):
    article = extract_news(url)
    if article.startswith("⚠️ Failed"):
        return article
    summary = summarize_chain.run(article=article)
    return summary.strip()

# Page config
st.set_page_config(page_title="📰 News Summarizer", page_icon="🧠", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: #003366;
            margin-bottom: 20px;
        }
        .summary-box {
            background-color: #f8f9fa;
            border-left: 5px solid #4CAF50;
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            color: gray;
            margin-top: 30px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stTextInput>div>input {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🧠 AI-Powered News Summarizer</div>', unsafe_allow_html=True)
st.markdown("Paste a news article URL below and get a short, intelligent summary powered by **Gemini AI**.")

# Input
user_url = st.text_input("🔗 Enter News URL")

# Process
if st.button("📄 Summarize"):
    if user_url:
        with st.spinner("⏳ Processing the article..."):
            result = summarize_news(user_url)

        if result.startswith("⚠️ Failed"):
            st.error(result)
        else:
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.markdown(f"### 📝 Summary\n{result}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a valid URL.")

# Footer
st.markdown('<div class="footer">Built with ❤️ using Streamlit, LangChain & Gemini | © 2025</div>', unsafe_allow_html=True)
