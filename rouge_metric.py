from rouge_score import rouge_scorer
from nltk.translate import bleu_score

def calculate_rouge_scores(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores

def calculate_bleu_score(reference, prediction):
    return bleu_score.sentence_bleu([reference.split()], prediction.split())

def run_rouge_metric(reference_text, predicted_text):
    # Hitung ROUGE scores
    rouge_scores = calculate_rouge_scores(reference_text, predicted_text)
    return [
        rouge_scores['rouge1'].precision, 
        rouge_scores['rouge1'].recall, 
        rouge_scores['rouge1'].fmeasure,
        rouge_scores['rouge2'].precision, 
        rouge_scores['rouge2'].recall, 
        rouge_scores['rouge2'].fmeasure,
        rouge_scores['rougeL'].precision, 
        rouge_scores['rougeL'].recall, 
        rouge_scores['rougeL'].fmeasure
    ]



