import streamlit as st
from main import summarizer_bart

st.title('Text summarizer based on BART :robot_face:')

text_block = """\
The integration of Artificial Intelligence (AI) in customer engagement strategies has marked a
new era in the business-consumer relationship. AI technologies are transforming how
businesses interact with their customers, offering personalized experiences and enhancing
customer satisfaction. One of the key applications of AI in customer engagement is through chatbots and 
virtual assistants. These AI-powered tools can handle a wide range of customer service inquiries,
providing instant responses and support 24/7. They are programmed to learn from interactions,
enabling them to deliver more accurate and helpful responses over time. This not only improves
the efficiency of customer service but also allows human agents to focus on more complex
queries. Another significant impact of AI is in the realm of data analysis and customer insights.
AI algorithms can process vast amounts of data from various customer touchpoints and extract
meaningful patterns and trends. This data-driven approach enables businesses to understand
customer behavior and preferences in-depth, leading to more targeted and effective marketing
strategies. Personalization is also a major benefit of AI in customer engagement. By analyzing
past interactions and purchases, AI can help businesses tailor their communications and
recommendations to individual customers. This level of personalization enhances the customer
experience, increases brand loyalty, and can lead to higher conversion rates.
AI is also reshaping customer engagement through predictive analytics. By forecasting future
customer behaviors and trends, businesses can proactively address potential issues and seize
opportunities. This forward-looking approach helps in maintaining a competitive edge and
staying relevant in the market. However, the use of AI in customer engagement also presents
challenges, particularly in terms of privacy and ethical considerations. Businesses must ensure
that customer data is handled responsibly, and AI interactions are transparent and secure.
In conclusion, AI is playing a crucial role in revolutionizing customer engagement. Its ability to
provide personalized, efficient, and data-driven interactions is not only enhancing the customer
experience but also driving business growth and innovation. As AI technology continues to
evolve, its impact on customer engagement is expected to grow even further, shaping the future
of business-customer relationships.
"""

text = st.text_area('Input text', value=text_block, height=300)
min_length = st.slider('Minimum word length', 15, 30, 20)
max_length = st.slider('Maximum word length', 30, 300, 150)

# On click of the button, start the summarization process
if st.button("Summarize"):

    # Call the long-running process function
    with st.spinner("Running... Please wait."):
        sb = summarizer_bart(text=text,
                             min_length=min_length,
                             max_length=max_length)

        # Display the result after the function is finished
        summ1 = sb.pipeline_summarizer()[0]['summary_text']
        summ2 = sb.tokenizing_summarizer()

        with st.container():
            s1 = st.text_area('Summary 1', value=summ1, height=100)
            s2 = st.text_area('Summary 2', value=summ2, height=100)

    st.success("Completed successfully!")







