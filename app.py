# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, jsonify, request
from model import classify
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/klasifikasi',  methods=['GET'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    category = ["Belanja", "Makanan & Minuman", "Kesehatan & Kecantikan", "Pendidikan", "Transportasi", "Pinjaman & Sosial", "Sosial", "Hiburan", "Investasi"]
    name_product  = request.args.get('name_product')
    predict = classify(name_product)
    return jsonify({"name_product" : name_product, "category" : category[predict]})
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()