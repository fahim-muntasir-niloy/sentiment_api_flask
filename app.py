from flask import Flask, render_template, request, jsonify
from setfit import SetFitModel

app = Flask(__name__)

model = SetFitModel.from_pretrained("./model/miniLM")
label_mapping = {
    2: "Negative",
    0: "Neutral",
    1: "Positive"
}

# cheking in json

@app.route('/prediction',methods=['POST'])
def prediction():
    text = request.json['text']
    prediction_output = model([text]).item()   # return 0/1/2
    prediction = label_mapping[prediction_output]  # returns pos neg neutral
    return jsonify({'prediction': prediction})






@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        text_to_analyze = request.form['sentence']
        
        if text_to_analyze.strip(): 
            return render_template("index.html", sentence=text_to_analyze)
        else:
            error_msg = "Enter text to analyze 🤖"
            return render_template("index.html", error_msg=error_msg)
    
    
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)