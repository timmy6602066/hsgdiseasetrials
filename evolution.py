!pip install psycopg2
!pip install groq
!pip install sacrebleu bert-score textstat

import psycopg2
from groq import Groq
from bert_score import score
from textstat import flesch_kincaid_grade
import numpy as np
from collections import Counter

# Groq API Key
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Function to execute queries for a given NCT ID
def execute_queries(nct_id):
    conn = psycopg2.connect(
        host="aact-db.ctti-clinicaltrials.org",
        database="aact",
        user="timmy6602066",
        password="nzj6602066",
        port="5432"
    )

    cur = conn.cursor()

    queries = [
        "SELECT * FROM ctgov.browse_interventions WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.design_groups WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.design_outcomes WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.eligibilities WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.interventions WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.keywords WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.outcome_analyses WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.outcome_counts WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.outcome_measurements WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.outcomes WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.studies WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.study_references WHERE nct_id = '{nct_id}';",
        "SELECT * FROM ctgov.browse_conditions WHERE nct_id = '{nct_id}';",
        "SELECT description FROM ctgov.brief_summaries WHERE nct_id = '{nct_id}';",
        "SELECT description FROM ctgov.detailed_descriptions WHERE nct_id = '{nct_id}';"
    ]

    combined_results = []

    for query in queries:
        formatted_query = query.format(nct_id=nct_id)
        cur.execute(formatted_query)
        study_info = cur.fetchall()

        if study_info:
            for row in study_info:
                combined_results.append(str(row))

    cur.close()
    conn.close()

    return "\n".join(combined_results)

# Function to process the text using Groq
def process_with_groq(input_text):
    lm = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Explain the recommendation trial method or conclusion for patients from this trial. Then summarize the treatment details, precautions, and other relevant information about this condition for the patient based on the experimental report. "
                    "Use simple sentences and easy-to-understand language (For people with a middle school education)."
                ),
            },
            {
                "role": "user",
                "content": input_text,
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    return lm.choices[0].message.content

# Function to evaluate text simplification metrics
def evaluate_ts_metrics(input_text, output_text, reference_texts):
    """
    Evaluate Text Simplification metrics.

    :param input_text: Original input text (combined_results).
    :param output_text: Simplified output text from the model.
    :param reference_texts: List of reference simplifications for comparison.
    :return: Dictionary with metric scores.
    """

    # FKGL Score for Original Text
    fkgl_original = flesch_kincaid_grade(input_text)

    # BERTScore
    bert_scores = score([output_text], [reference_texts[0]], lang="en")
    bert_precision, bert_recall, bert_f1 = bert_scores[0].mean().item(), bert_scores[1].mean().item(), bert_scores[2].mean().item()

    # FKGL Score for Simplified Text
    fkgl_simplified = flesch_kincaid_grade(output_text)

    # SARI Score
    sari_scores = sari_score([input_text], [output_text], reference_texts)

    # Compile results
    results = {
        "Original FKGL": fkgl_original,
        "Simplified FKGL": fkgl_simplified,
        "BERT Precision": bert_precision,
        "BERT Recall": bert_recall,
        "BERT F1": bert_f1,
        "SARI Preservation": sari_scores[0],
        "SARI Insertion": sari_scores[1],
        "SARI Deletion": sari_scores[2]
    }
    return results

# SARI Score Calculation (simplified)
def sari_score(orig_sentences, simplified_sentences, reference_sentences):
    def _sari_for_sentence(orig, simpl, references):
        preservation, insertion, deletion = 0, 0, 0
        orig_words = set(orig.split())
        simpl_words = set(simpl.split())
        
        # Calculate preservation, insertion, and deletion
        for word in orig_words:
            if word in simpl_words:
                preservation += 1
            else:
                deletion += 1
        
        for word in simpl_words:
            if word not in orig_words:
                insertion += 1
        
        num_references = len(references)
        reference_words = [set(ref.split()) for ref in references]
        
        # Calculate the preservation, insertion, and deletion ratios
        preservation_ratio = preservation / max(len(orig_words), 1)
        insertion_ratio = insertion / max(len(simpl_words), 1)
        deletion_ratio = deletion / max(len(orig_words), 1)
        
        return preservation_ratio, insertion_ratio, deletion_ratio
    
    preservation_scores = []
    insertion_scores = []
    deletion_scores = []
    
    for orig, simpl, ref in zip(orig_sentences, simplified_sentences, reference_sentences):
        preservation, insertion, deletion = _sari_for_sentence(orig, simpl, ref)
        preservation_scores.append(preservation)
        insertion_scores.append(insertion)
        deletion_scores.append(deletion)
    
    # Average over all sentences
    sari_score = np.mean(preservation_scores), np.mean(insertion_scores), np.mean(deletion_scores)
    
    return sari_score

# Main function
if __name__ == "__main__":
    # List of NCT IDs
    nct_ids = [
        "NCT05105191", "NCT00782431", "NCT05105191", "NCT04059991", "NCT00556062",
        "NCT01622491", "NCT05371327", "NCT02482662", "NCT03841591",
        "NCT04738591", "NCT04738591", "NCT01545791"
    ]

    reference_texts = [
        "This is a sample simplified summary text for comparison purposes."
    ]

    all_metrics = {}

    for nct_id in nct_ids:
        query_results = execute_queries(nct_id)

        if query_results:
            simplified_output = process_with_groq(query_results)
            metrics = evaluate_ts_metrics(query_results, simplified_output, reference_texts)
            all_metrics[nct_id] = metrics
        else:
            print(f"No data found for NCT ID: {nct_id}")

    # Print Evaluation Metrics
    print("\nEvaluation Metrics for all NCT IDs:")
    for nct_id, metrics in all_metrics.items():
        print(f"\nNCT ID: {nct_id}")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.4f}")

        # Print FKGL Difference
        fkgl_diff = metrics["Original FKGL"] - metrics["Simplified FKGL"]
        print(f"FKGL Difference (Original - Simplified): {fkgl_diff:.4f}")

    # Calculate average metrics
    total_nct = len(all_metrics)
    average_metrics = {
        "Original FKGL": sum(metrics["Original FKGL"] for metrics in all_metrics.values()) / total_nct,
        "Simplified FKGL": sum(metrics["Simplified FKGL"] for metrics in all_metrics.values()) / total_nct,
        "BERT Precision": sum(metrics["BERT Precision"] for metrics in all_metrics.values()) / total_nct,
        "BERT Recall": sum(metrics["BERT Recall"] for metrics in all_metrics.values()) / total_nct,
        "BERT F1": sum(metrics["BERT F1"] for metrics in all_metrics.values()) / total_nct,
        "SARI Preservation": sum(metrics["SARI Preservation"] for metrics in all_metrics.values()) / total_nct,
        "SARI Insertion": sum(metrics["SARI Insertion"] for metrics in all_metrics.values()) / total_nct,
        "SARI Deletion": sum(metrics["SARI Deletion"] for metrics in all_metrics.values()) / total_nct,
    }

    # Print Average Metrics
    print("\nAverage Metrics Across All NCT IDs:")
    for metric, score in average_metrics.items():
        print(f"{metric}: {score:.4f}")



#Average Metrics Across All NCT IDs:
#Original FKGL: 15.9800
#Simplified FKGL: 12.4100
#BERT Precision: 0.7620
#BERT Recall: 0.8153
#BERT F1: 0.7877
#SARI Preservation: 0.1770
#SARI Insertion: 0.5468
#SARI Deletion: 0.8230

