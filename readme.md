# Sentiment Analysis web-app in Flask 😊😐☹️
This web app is built on custom [Setfit](https://huggingface.co/StatsGary/setfit-ft-sentinent-eval) sentiment analysis model, where the base model is from `Hugging Face Transformers library` and later custom trained on the `twitter sentiment dataset` for custom class response.

## Model training 🤖
The Model used here is an efficient and prompt-free framework for few-shot fine-tuning of Sentence Transformers. It achieves high accuracy with little labeled data. 
As github doesn't support large files normally, the trained model folder is not uploaded. However, running the `model.py` file in the `model` folder would create a directory called `miniLM` which will contain the trained model. You can modify the parameters for training.

```cmd
cd model
python model.py
```

*This has to be run only once for creating the model and the miniLM folder.*

## Starting the Flask Server 🌶️
Running the `app.py` file would start the flask server at port 5000.
```cmd
python app.py
```


## Testing API endpoint in *Postman* 🧮
To check the API in postman, an endpoint at [http://127.0.0.1:5000/analyze](http://127.0.0.1:5000/analyze) was created. This simply took a json input in the body as raw json data, and returned the sentiment as json output.

```py
# input
{
    "text" : "Today is a boring day"
}
```

```py
# ouput
{
    "sentiment" : "Negative"
}
```

There is also the `predict.py` file which can send request to the "/analyze" directory and print the sentiment output.

## Web-App 🕸️
After starting the server, simply go to [http://127.0.0.1:5000/prediction](http://127.0.0.1:5000/prediction) for the web UI, where entering any text would print the sentiment output as positive/negative/neutral on the page.



# References 
1. StatsGary/Setfit Model: https://huggingface.co/StatsGary/setfit-ft-sentinent-eval
2. Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
