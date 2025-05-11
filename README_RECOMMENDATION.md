# Product Recommendation API Integration

This integration adds a product recommendation system to your agricultural application that suggests products based on:

1. The user's planted crops and regions
2. Their purchase history
3. AI-powered product matching for plant protection

## Endpoint

```
GET /recommend/{customer_id}
```

Where `{customer_id}` is the unique identifier for the customer.

## How It Works

The recommendation system uses multiple data sources and techniques:

1. **User Land Data**: Retrieves information about the user's lands, regions, and planted crops
2. **Purchase History**: Analyzes past purchases using matrix factorization (SVD)
3. **AI-Powered Matching**: Uses the LLM to identify products suitable for the user's crops
4. **Category Boosting**: Increases relevance of products in categories the user has purchased before

## Dependencies

The system requires:
- FastAPI
- Pandas and NumPy for data processing
- SciPy and Scikit-learn for the recommendation algorithm
- Access to:
  - User land/region/plant data API (port 3000)
  - Product and order data API (port 3000)
  - Ollama LLM API (port 11434) with the "ayansh03/fasal-mitra" model

## Testing

To test the endpoint, start your FastAPI application:

```bash
python main.py
```

Then access the endpoint:

```
http://localhost:8000/recommend/{customer_id}
```

Replace `{customer_id}` with an actual customer ID from your system.

## Example Response

```json
[
  {
    "_id": "65f9b2e9dff8a6d988f5d1e5",
    "name": "Organic Neem Oil Spray",
    "description": "Natural pesticide for organic farming",
    "category": "pesticides",
    "price": 12.99,
    "stockQuantity": 200,
    "image": "neem_oil.jpg"
  },
  ...
]
```

## Troubleshooting

If you encounter issues:

1. Ensure all API endpoints at port 3000 are accessible
2. Check that the Ollama LLM service is running 
3. Verify the database has product and order data
4. Check server logs for detailed error messages 