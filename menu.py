from flask import Flask, render_template, request
import json
import model


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    print('HI')
    age=request.form.get("age")
    gender=request.form.get("gender")
    family=request.form.get("family")
    car = request.form.get("car")
    bike = request.form.get("bike")
    revenue = request.form.get("revenue")
    if (gender == 'M'):
        gender = 0
    elif (gender == 'F'):
        gender = 1
    if (car == 'yes' or car == 'Yes'):
        car = 1
    elif (car == 'no' or car == 'No'):
        car = 0
    if (bike == 'yes' or bike == 'Yes'):
        bike = 1
    elif (bike == 'no' or bike == 'No'):
        bike = 0
    data = {
        'age': age,
        'gender': gender,
        'family': family,
        'car':car,
        'bike':bike,
        'revenue':revenue
    }
    prediction_var = model.return_preds(model.my_model,data)[0]
    result_dummy = 'JBSDKJNEF'
    if model.return_preds(model.my_model,data)[0] == 1:
        result = 'You will use public transportation'
    elif model.return_preds(model.my_model,data)[0] == 0:
        result = 'You will not use public transportation'
    else:
        result = ''

    return render_template('index.html', result = prediction_var, result_dummy = result_dummy)

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = '65432')