import json
import requests
import re
import pandas as pd
from groq import Groq
from tabulate import tabulate
from pymongo import MongoClient
from flask import Flask, render_template, request
from disease_type import predict_disease, obtain_disease_name


app = Flask(__name__)
app.config['safe'] = None
app.config['disease_name'] = None

@app.route('/')
def index():
    return render_template('index.html')

MONGO_URI = "mongodb://47.109.42.124:27017"
DATABASE_NAME = "patientDB"
COLLECTION_NAME = "patientInfo"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# GROQ_API_KEY = "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"
# groq_client = Groq(api_key=GROQ_API_KEY)


@app.route('/api/submitPatientData', methods=['POST'])
def api_submit_patient_data():
    try:
        # patient_info = request.get_json()
        data = dict()
        name = request.form.get('name', '')
        data['name'] = name
        age = request.form.get('age', '')
        data['age'] = int(age)
        gender = request.form.get('gender', '')
        data['gender'] = gender
        chronicConditions = request.form.get('chronicConditions', '')
        data['chronicConditions'] = chronicConditions
        pastSurgeries = request.form.get('pastSurgeries', '')
        data['pastSurgeries'] = pastSurgeries
        familyHistory = request.form.get('familyHistory', '')
        data['familyHistory'] = familyHistory
        currentMedications = request.form.get('currentMedications', '')
        data['currentMedications'] = currentMedications
        previousMedications = request.form.get('previousMedications', '')
        data['previousMedications'] = previousMedications
        allergies = request.form.get('allergies', '')
        data['allergies'] = allergies
        currentSymptoms	= request.form.get('currentSymptoms', '')
        data['currentSymptoms'] = currentSymptoms
        symptomDuration	= request.form.get('symptomDuration', '')
        data['symptomDuration'] = symptomDuration
        symptomSeverity	= request.form.get('symptomSeverity', '')
        data['symptomSeverity'] = symptomSeverity
        diagnosis = request.form.get('diagnosis', '')
        data['diagnosis'] = diagnosis

        # print("Received data:", data)  # Debug print
        predicted_disease = predict_disease(data)
        
        print("Predicted disease:", predicted_disease)  # Debug print
    
        collection.insert_one(data)
        print("Data inserted to MongoDB")  # Debug print

        disease_name = obtain_disease_name(predicted_disease)
        print(f"Our prediction: {disease_name}")


        app.config['safe'] = predicted_disease
        app.config['disease_name'] = disease_name

        return render_template('nextpage.html',
                               predicted_disease=predicted_disease,
                               disease_name=disease_name)
    except Exception as e:
        print("Error:", str(e))  # Debug print error message
        from flask import jsonify
        return jsonify({'error': str(e)})


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        patient_info = {
            'name': request.form['name'],
            'age': request.form['age'],
            'gender': request.form['gender'],
            'currentSymptoms': request.form['currentSymptoms'],
            'symptomDuration': request.form['symptomDuration'],
            'symptomSeverity': request.form['symptomSeverity'],
            'chronicConditions': request.form.get('chronicConditions', ''),
            'pastSurgeries': request.form.get('pastSurgeries', ''),
            'familyHistory': request.form.get('familyHistory', ''),
            'currentMedications': request.form.get('currentMedications', ''),
            'previousMedications': request.form.get('previousMedications', ''),
            'allergies': request.form.get('allergies', ''),
            'diagnosis': request.form.get('diagnosis', ''),
        }

        disease_name = predict_disease(patient_info)
        print(f"Our prediction: {disease_name}")

        return render_template("nextpage.html", disease_name=disease_name)

    return render_template("index.html")


@app.route('/ntc_ids', methods=["POST"])
def nextpage():
    if request.method == 'POST':
        disease_name = request.form.get('disease_name', '')
        from disease_type import ntc_ids
        ntcids = ntc_ids(disease_name)
        safe = app.config.get('safe')
        disease_name = app.config.get('disease_name')
        return render_template("nextpage.html", ntcids=ntcids, safe=safe, disease_name=disease_name)


@app.route('/detail', methods=["GET"])
def detail():
    nctId = request.args.get("nctId")
    print(nctId, 'ntcid')
    from fetch_trials_output import main    
    result = main(nctId)

    result = re.sub(r"(\w{4}:)", r"<strong>\1</strong>", result)
    formatted_result = "<p>" + result.replace('\n', '</p><p>') + "</p>"

    return f"<html><body>{formatted_result}</body></html>"


if __name__ == "__main__":
    app.run(debug=True, port=8080, use_reloader=False)

