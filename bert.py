from summarizer import Summarizer,TransformerSummarizer


def run_bert(body):
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(body, min_length=60))
    
    return bert_summary
