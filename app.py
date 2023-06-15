from flask import Flask, render_template, request

app = Flask(__name__)

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