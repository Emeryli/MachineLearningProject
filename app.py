# import
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

# create a flask instance
application = Flask(__name__)
app = application

# route for index page
@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # create an instance with the data from user

        
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        # use created function to build a dictionary 
        custom_df=data.get_data_as_dataframe()
        # create prediction pipeline instance
        predict_pipeline=PredictionPipeline()
        # get the result
        results=predict_pipeline.predict(custom_df)
        # return the template
        return render_template('home.html',
                                gender=request.form.get('gender'),
                                ethnicity=request.form.get('ethnicity'),
                                parental_level_of_education=request.form.get('parental_level_of_education'),
                                lunch=request.form.get('lunch'),
                                test_preparation_course=request.form.get('test_preparation_course'),
                                reading_score=request.form.get('reading_score'),
                                writing_score=request.form.get('writing_score'),
                                results=round(float(results[0]),1),          
        )
    
# test
if __name__ == "__main__":
    app.run('0.0.0.0')
    