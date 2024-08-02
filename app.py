from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import threadpoolctl

app = Flask(__name__)

threadpoolctl.threadpool_limits(1, "blas")

data = pd.read_csv('amazon.csv')
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data = data.dropna(subset=['rating'])
data = data.groupby(['user_id', 'product_id']).agg({'rating': 'mean'}).reset_index()
user_item_matrix = data.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
user_item_matrix = user_item_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
user_item_matrix_csr = csr_matrix(user_item_matrix.values)

model = implicit.als.AlternatingLeastSquares(factors=100, iterations=15, regularization=0.1)
model.fit(user_item_matrix_csr)

@app.route('/')
def index():
    return "Welcome to the Product Recommendation System!"

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    user_id = int(user_id)
    if user_id >= user_item_matrix.shape[0]:
        return jsonify({"error": "User ID not found"}), 404
    recommendations = model.recommend(user_id, user_item_matrix_csr[user_id], N=10)
    recommended_product_ids = recommendations[0]
    return jsonify({"recommended_products": recommended_product_ids.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
