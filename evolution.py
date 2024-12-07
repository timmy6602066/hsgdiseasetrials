!pip install psycopg2
!pip install groq
!pip install sacrebleu bert-score textstat

import psycopg2
from groq import Groq
from bert_score import score
from textstat import flesch_kincaid_grade

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

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
        model="mixtral-8x7b-32768",
    )

    return lm.choices[0].message.content

def evaluate_ts_metrics(input_text, output_text, reference_texts):
    """
    Evaluate Text Simplification metrics.
    
    :param input_text: Original input text (combined_results).
    :param output_text: Simplified output text from the model.
    :param reference_texts: List of reference simplifications for comparison.
    :return: Dictionary with metric scores.
    """

    # BERTScore
    bert_scores = score([output_text], [reference_texts[0]], lang="en")
    bert_precision, bert_recall, bert_f1 = bert_scores[0].mean().item(), bert_scores[1].mean().item(), bert_scores[2].mean().item()

    # FKGL Score
    fkgl = flesch_kincaid_grade(output_text)

    # Compile results
    results = {
        "BERT Precision": bert_precision,
        "BERT Recall": bert_recall,
        "BERT F1": bert_f1,
        "FKGL": fkgl
    }
    return results

if __name__ == "__main__":
    # List of NCT IDs
    nct_ids = [
        "NCT05105191", "NCT00782431","NCT05105191","NCT04059991","NCT00556062","NCT01622491","NCT05371327","NCT02482662","NCT03841591",
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

    # Calculate average metrics
    total_nct = len(all_metrics)
    average_metrics = {
        "BERT Precision": sum(metrics["BERT Precision"] for metrics in all_metrics.values()) / total_nct,
        "BERT Recall": sum(metrics["BERT Recall"] for metrics in all_metrics.values()) / total_nct,
        "BERT F1": sum(metrics["BERT F1"] for metrics in all_metrics.values()) / total_nct,
        "FKGL": sum(metrics["FKGL"] for metrics in all_metrics.values()) / total_nct,
    }

    # Print Evaluation Metrics
    print("\nEvaluation Metrics for all NCT IDs:")
    for nct_id, metrics in all_metrics.items():
        print(f"\nNCT ID: {nct_id}")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.4f}")
    
    # Print Average Metrics
    print("\nAverage Metrics Across All NCT IDs:")
    for metric, score in average_metrics.items():
        print(f"{metric}: {score:.4f}")


# Evaluation Metrics for all NCT IDs:

# NCT ID: NCT05105191
# BERT Precision: 0.7486
# BERT Recall: 0.8144
# BERT F1: 0.7802
# FKGL: 12.5000

# NCT ID: NCT00782431
# BERT Precision: 0.7686
# BERT Recall: 0.8030
# BERT F1: 0.7854
# FKGL: 12.0000

# NCT ID: NCT04059991
# BERT Precision: 0.7546
# BERT Recall: 0.8113
# BERT F1: 0.7819
# FKGL: 11.3000

# NCT ID: NCT00556062
# BERT Precision: 0.7320
# BERT Recall: 0.8020
# BERT F1: 0.7654
# FKGL: 10.1000

# NCT ID: NCT01622491
# BERT Precision: 0.7735
# BERT Recall: 0.8152
# BERT F1: 0.7938
# FKGL: 12.3000

# NCT ID: NCT05371327
# BERT Precision: 0.7583
# BERT Recall: 0.8105
# BERT F1: 0.7835
# FKGL: 13.1000

# NCT ID: NCT02482662
# BERT Precision: 0.7563
# BERT Recall: 0.8132
# BERT F1: 0.7837
# FKGL: 10.2000

# NCT ID: NCT03841591
# BERT Precision: 0.7720
# BERT Recall: 0.8086
# BERT F1: 0.7899
# FKGL: 12.8000

# NCT ID: NCT04738591
# BERT Precision: 0.7492
# BERT Recall: 0.8185
# BERT F1: 0.7823
# FKGL: 15.0000

# NCT ID: NCT01545791
# BERT Precision: 0.7391
# BERT Recall: 0.8153
# BERT F1: 0.7754
# FKGL: 13.7000

# Average Metrics Across All NCT IDs:
# BERT Precision: 0.7552
# BERT Recall: 0.8112
# BERT F1: 0.7821
# FKGL: 12.3000

