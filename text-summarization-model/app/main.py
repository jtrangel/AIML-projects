from transformers import pipeline, BartForConditionalGeneration, BartTokenizer, BartConfig
from ktrain.text.summarization import TransformerSummarizer

class summarizer_bart():
    def __init__(self, text: str, min_length: int, max_length: int):
        """
        Summarizer model making use of the pre-trained BART model.

        :param text: Input block of text to be summarized
        :type text: str
        :param min_length: Minimum word length of summary
        :type min_length: int
        :param max_length: Maximum word length of summary
        :type max_length: int

        For a given set of text, returns a summarized version.
        """
        self.model = "philschmid/bart-large-cnn-samsum"
        self.model2 = "facebook/bart-large-cnn"
        self.text = text
        self.min_length = min_length
        self.max_length = max_length

    def pipeline_summarizer(self) -> str:
        """
        Summarizer that utilizes the transformers pipeline method for use of the `facebook/bart-large-cnn-samsum` model.
        Fine-tuned on the CNN daily mail and Samsum dataset.

        :return: Summarized text based on input params
        :rtype: str
        """
        summarizer = pipeline("summarization", model=self.model)
        return summarizer(self.text, min_length=self.min_length, max_length=self.max_length, do_sample=False)

    def tokenizing_summarizer(self) -> str:
        """
        Summarizer using tokenization for use of the `facebook/bart-large-cnn` model.
        Fine-tuned on CNN daily mail.

        :return: Summarized text based on input params
        :rtype: str
        """
        # Tokenizer and model loading for bart-large-cnn
        tokenizer = BartTokenizer.from_pretrained(self.model2)
        model = BartForConditionalGeneration.from_pretrained(self.model2)

        # Transmitting the encoded inputs to the model.generate() function
        inputs = tokenizer.batch_encode_plus([self.text], return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'],
                                     num_beams=4,
                                     min_length=self.min_length,
                                     max_length=self.max_length,
                                     early_stopping=True)

        # Decoding and printing the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    ## Basically the same as the second method, packaged differently.
    # def ktrain_summarizer(self, text: str) -> str:
    #     """
    #     BART based text summarization that uses the `facebook/bart-large-cnn` model specifically.
    #     Fine tuned on CNN daily mail.
    #
    #     :return: Summarized text based on input params
    #     :rtype: str
    #     """
    #     ts = TransformerSummarizer()
    #     return ts.summarize(text, min_length=self.min_length, max_length=self.max_length)

if __name__ == '__main__':
    text_block = """ The integration of Artificial Intelligence (AI) in customer engagement strategies has marked a
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

    sm = summarizer_bart(text=text_block, min_length=30, max_length=200)

    summ1 = sm.pipeline_summarizer()[0]['summary_text']
    summ2 = sm.tokenizing_summarizer()

    print(summ1)
    print()
    print(summ2)
    print()
