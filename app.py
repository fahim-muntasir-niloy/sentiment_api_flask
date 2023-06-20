from flask import Flask, render_template, request, jsonify
from setfit import SetFitModel

app = Flask(__name__)

model = SetFitModel.from_pretrained("./model/miniLM")       # load the model from directory
label_mapping = {
    2: "Negative",
    0: "Neutral",
    1: "Positive"
}

# checking in json

@app.route('/analyze',methods=['POST'])
def analyze():
    if not request.json or 'text' not in request.json:      # response error returned as 400
        return jsonify({"Error":"Response not given"}), 400
    
    text = request.json['text']
    prediction_output = model([text]).item()        # return 0/1/2
    prediction = label_mapping[prediction_output]   # returns pos neg neutral
    return jsonify({'sentiment': prediction})



# website Page


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        text_to_analyze = request.form.get('sentence')
        
        if text_to_analyze.strip():
            prediction_output = model([text_to_analyze]).item()
            prediction = label_mapping[prediction_output]
            return render_template("index.html", sentence=prediction)
        
        
        else:
            error_msg = "Enter text to analyze 🤖"
            return render_template("index.html", error_msg=error_msg)
    
    
    return render_template("index.html")



# run website

if __name__ == "__main__":
    app.run(debug=True, port=5000)