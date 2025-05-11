from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import requests
import re
import json
import traceback
import logging
from typing import List, Dict, Any, Optional

# Import the plant protection keywords for smart product matching
from plant_protection_keywords import plant_protection_keywords

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def is_ollama_available():
    """Check if the Ollama service is available."""
    try:
        response = requests.get("http://192.168.43.232:11434/api/version", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_user_land_info(user_id: str):
    """Fetch user's land information and planted plants."""
    try:
        logger.info(f"Fetching lands for user {user_id}")
        response = requests.get("http://192.168.43.232:3000/lands/all")
        response.raise_for_status()
        all_lands = response.json()
        logger.info(f"Retrieved {len(all_lands)} lands")
    except requests.RequestException as e:
        logger.error(f"Error fetching lands: {str(e)}")
        return None, None

    user_lands = [land for land in all_lands if land.get("user") == user_id]
    logger.info(f"Found {len(user_lands)} lands for user {user_id}")
    
    if not user_lands:
        return None, None

    region_names = []
    plant_ids = []

    for land in user_lands:
        for region_id in land.get("regions", []):
            try:
                logger.info(f"Fetching region {region_id}")
                region_resp = requests.get(f"http://192.168.43.232:3000/lands/region/{region_id}")
                region_resp.raise_for_status()
                region_data = region_resp.json()

                if name := region_data.get("name"):
                    region_names.append(name.lower())

                for plant_entry in region_data.get("plants", []):
                    if pid := plant_entry.get("plant"):
                        plant_ids.append(pid)
            except requests.RequestException as e:
                logger.error(f"Error fetching region {region_id}: {str(e)}")
                continue

    logger.info(f"Found {len(region_names)} regions and {len(plant_ids)} plants")
    return region_names, plant_ids

def get_plant_names(plant_ids):
    """Convert plant IDs to their corresponding names."""
    names = []
    for pid in plant_ids:
        try:
            logger.info(f"Fetching plant {pid}")
            resp = requests.get(f"http://192.168.43.232:3000/lands/plant/{pid}")
            resp.raise_for_status()
            plant_data = resp.json()
            if "name" in plant_data:
                names.append(plant_data["name"].lower())
        except requests.RequestException as e:
            logger.error(f"Error fetching plant {pid}: {str(e)}")
    logger.info(f"Retrieved plant names: {names}")
    return names

def query_ollama_for_useful_products(region_names, plant_names, products):
    """Query the LLM to identify useful products for the given plants."""
    if not region_names or not plant_names:
        logger.info("Skipping Ollama: No regions or plant names.")
        return []

    try:
        product_names = [products[pid]["name"] for pid in products if "name" in products[pid]]
        logger.info(f"Product names for LLM: {len(product_names)} products")

        prompt = (
            f"A user is planting {', '.join(plant_names)}. "
            f"Are any of these products: {json.dumps(product_names)} good for maintaining and protecting his crop from any diseases? "
            "Answer by only giving a list of the products that are useful."
        )

        logger.info(f"=== OLLAMA PROMPT ===")
        logger.info(prompt)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "ayansh03/fasal-mitra",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")
        logger.info(f"=== OLLAMA RAW RESPONSE ===")
        logger.info(response_text)

        matches = re.findall(r'\d+\.\s*(.+)', response_text)
        if not matches:
            matches = re.split(r'[\n,-]', response_text)

        cleaned = [re.sub(r'\s+', ' ', m.strip().lower()) for m in matches if m.strip()]
        logger.info(f"=== MATCHED PRODUCTS FROM AI ===")
        logger.info(f"{cleaned}")
        return cleaned

    except requests.RequestException as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in LLM processing: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def keyword_match_products(plant_names, products):
    """Simple keyword matching between plant names and products."""
    matched_products = []
    
    for plant_name in plant_names:
        plant_terms = plant_name.lower().split()
        
        # Try to match with crop keywords
        matched_crop = None
        for crop, keywords in plant_protection_keywords["crops"].items():
            if any(keyword in plant_name.lower() for keyword in keywords):
                matched_crop = crop
                break
        
        # Go through all products
        for product_id, product in products.items():
            if "name" not in product or "description" not in product:
                continue
                
            product_name = product["name"].lower()
            product_desc = product["description"].lower()
            
            # Check direct matches
            if any(term in product_name or term in product_desc for term in plant_terms):
                matched_products.append(product_name)
                continue
                
            # Check if product relates to any disease terms for the matched crop
            if matched_crop:
                for disease_type, terms in plant_protection_keywords["diseases"].items():
                    if any(term in product_name or term in product_desc for term in terms):
                        matched_products.append(product_name)
                        break
                        
            # Check if product has treatment keywords
            for treatment_type, terms in plant_protection_keywords["treatments"].items():
                if any(term in product_name or term in product_desc for term in terms):
                    matched_products.append(product_name)
                    break
    
    logger.info(f"Keyword matching found {len(matched_products)} products")
    return list(set(matched_products))  # Remove duplicates

# Initialize data
df_orders = None
products = None
user_item_matrix = None
user_factors = None
product_factors = None
svd = None

def load_recommendation_data():
    """Load order and product data, and prepare the recommendation model."""
    global df_orders, products, user_item_matrix, user_factors, product_factors, svd
    
    try:
        logger.info("Fetching orders data")
        try:
            orders_response = requests.get("http://192.168.43.232:3000/order", timeout=5)
            orders_response.raise_for_status()
            orders = orders_response.json()
            logger.info(f"Retrieved {len(orders)} orders")
        except requests.RequestException as e:
            logger.warning(f"Error fetching orders: {str(e)}")
            orders = []

        logger.info("Fetching products data")
        try:
            products_response = requests.get("http://192.168.43.232:3000/product", timeout=5)
            products_response.raise_for_status()
            products_list = products_response.json()
            logger.info(f"Retrieved {len(products_list)} products")
        except requests.RequestException as e:
            logger.error(f"Error fetching products: {str(e)}")
            return False
            
        # We need products to be available
        if not products_list:
            logger.error("No products found")
            return False
            
        # Process products data first (this is always needed)
        products = {str(p["_id"]): p for p in products_list}
        logger.info(f"Processed {len(products)} products")
        
        # If there are no orders, we'll still return true but won't create a recommendation model
        if not orders:
            logger.warning("No orders found - will use basic recommendation method")
            # Initialize empty dataframes to avoid None checks
            df_orders = pd.DataFrame(columns=["_id", "customerId", "productId", "quantity"])
            user_item_matrix = None
            user_factors = None
            product_factors = None
            svd = None
            return True

        # Process orders data
        try:
            logger.info("Processing orders data")
            df_orders = pd.DataFrame(orders)
            df_orders["_id"] = df_orders["_id"].astype(str)
            df_orders["customerId"] = df_orders["customerId"].astype(str)
            df_orders["orderItems"] = df_orders["orderItems"].apply(lambda x: x if isinstance(x, list) else [])
            df_orders = df_orders.explode("orderItems").reset_index(drop=True)
            df_orders = df_orders[df_orders["orderItems"].apply(lambda x: isinstance(x, dict))]

            df_order_items = df_orders["orderItems"].apply(pd.Series)
            df_orders = pd.concat([df_orders.drop(columns=["orderItems"]), df_order_items], axis=1)

            df_orders["productId"] = df_orders["productId"].astype(str)
            df_orders["quantity"] = pd.to_numeric(df_orders["quantity"], errors="coerce").fillna(0)
            
            logger.info(f"Processed {len(df_orders)} order items for {df_orders['customerId'].nunique()} customers")

            # Create recommendation model if there's enough data
            if not df_orders.empty and df_orders['customerId'].nunique() > 0 and df_orders['productId'].nunique() > 0:
                logger.info("Creating recommendation model")
                # Create user-item matrix and SVD model
                user_item_matrix = df_orders.pivot_table(
                    index="customerId", columns="productId", values="quantity", aggfunc="sum", fill_value=0
                )
                
                logger.info(f"Created user-item matrix of shape {user_item_matrix.shape}")
                
                if user_item_matrix.shape[1] > 1:  # Need at least 2 products for SVD
                    sparse_matrix = csr_matrix(user_item_matrix.values)
                    n_components = min(5, user_item_matrix.shape[1] - 1)
                    
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    user_factors = svd.fit_transform(sparse_matrix)
                    product_factors = svd.components_
                    user_factors = user_factors / np.linalg.norm(user_factors, axis=1, keepdims=True)
                    
                    logger.info(f"SVD model created with {n_components} components")
                else:
                    logger.warning("Not enough products for SVD model, using simple scoring")
            else:
                logger.warning("Not enough data for recommendation model, using simple scoring")
        
        except Exception as e:
            logger.error(f"Error processing order data: {str(e)}")
            logger.error(traceback.format_exc())
            # We can still continue with just product data
            df_orders = pd.DataFrame(columns=["_id", "customerId", "productId", "quantity"])
            user_item_matrix = None
            user_factors = None
            product_factors = None
            svd = None
        
        return True
    except Exception as e:
        logger.error(f"Error processing recommendation data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@router.get("/recommend/{customer_id}", response_model=List[Dict[str, Any]])
async def recommend_products(customer_id: str):
    """Generate personalized product recommendations based on user's land, plants, and purchase history."""
    try:
        logger.info(f"Generating recommendations for customer: {customer_id}")
        
        # Ensure data is loaded
        global df_orders, products, user_item_matrix, user_factors, product_factors
        
        if products is None:
            logger.info("Loading recommendation data")
            success = load_recommendation_data()
            if not success:
                logger.error("Failed to load recommendation data")
                raise HTTPException(status_code=500, detail="Failed to load recommendation data")
        
        # Verify we have product data
        if not products:
            logger.error("No product data available")
            raise HTTPException(status_code=500, detail="No product data available")
        
        # Initialize with default scores
        product_scores = pd.Series(1.0, index=[prod_id for prod_id in products.keys()])
        
        # APPROACH 1: Use order history if available
        has_order_history = False
        if user_item_matrix is not None and customer_id in user_item_matrix.index:
            logger.info("Found order history - using collaborative filtering")
            try:
                user_index = user_item_matrix.index.get_loc(customer_id)
                scores = np.dot(user_factors[user_index], product_factors)
                
                # Update scores with collaborative filtering results
                matrix_product_ids = list(user_item_matrix.columns)
                for i, prod_id in enumerate(matrix_product_ids):
                    if prod_id in product_scores.index:
                        product_scores[prod_id] = scores[i]
                
                # Boost products in categories the user has ordered before
                if df_orders is not None and customer_id in df_orders["customerId"].values:
                    logger.info("Boosting products in previously ordered categories")
                    try:
                        user_ordered_products = df_orders[df_orders["customerId"] == customer_id]["productId"].unique()
                        user_categories = set(
                            products.get(prod_id, {}).get("category", None)
                            for prod_id in user_ordered_products if prod_id in products
                        )
                        user_categories.discard(None)  # Remove None if present
                        
                        for prod_id in product_scores.index:
                            category = products.get(prod_id, {}).get("category")
                            if category in user_categories:
                                product_scores[prod_id] *= 1.25
                    except Exception as e:
                        logger.error(f"Error boosting category products: {str(e)}")
                
                has_order_history = True
            except Exception as e:
                logger.error(f"Error using collaborative filtering: {str(e)}")
                logger.error(traceback.format_exc())
        
        # APPROACH 2: Use plant data if available
        if not has_order_history:
            logger.info("No order history - checking for plant data")
            
            # Get user's land and plant information
            region_names, planted_plant_ids = get_user_land_info(customer_id)
            
            if planted_plant_ids:
                logger.info(f"Found plant data: {len(planted_plant_ids)} plants")
                plant_names = get_plant_names(planted_plant_ids)
                
                if plant_names:
                    # Try to use LLM for suggestions
                    ai_useful_names = set()
                    try:
                        logger.info("Querying LLM for product suggestions")
                        ai_useful_names = set(query_ollama_for_useful_products(region_names, plant_names, products))
                        
                        # If LLM fails or returns no suggestions, use keyword matching as backup
                        if not ai_useful_names:
                            logger.info("LLM returned no suggestions, using keyword matching")
                            ai_useful_names = set(keyword_match_products(plant_names, products))
                    except Exception as e:
                        logger.error(f"Error getting product suggestions: {str(e)}")
                        logger.info("Using keyword matching for products")
                        ai_useful_names = set(keyword_match_products(plant_names, products))
                    
                    # Apply boosting based on plant matches
                    if ai_useful_names:
                        logger.info(f"Found {len(ai_useful_names)} matching products for plants")
                        product_name_lookup = {
                            prod_id: re.sub(r'\s+', ' ', products[prod_id].get("name", "").strip().lower())
                            for prod_id in products
                            if "name" in products[prod_id]
                        }
                        
                        for prod_id, clean_name in product_name_lookup.items():
                            for ai_name in ai_useful_names:
                                if ai_name in clean_name or clean_name in ai_name:
                                    logger.info(f"Boosting product '{clean_name}'")
                                    product_scores[prod_id] *= 5
                                    break
        
        # Get final sorted product list
        logger.info("Sorting products by scores")
        sorted_products = []
        
        try:
            sorted_product_ids = product_scores.sort_values(ascending=False).index.tolist()
            
            sorted_products = [
                {
                    "_id": prod_id,
                    "name": products.get(prod_id, {}).get("name", "Unknown Product"),
                    "description": products.get(prod_id, {}).get("description", "No description available"),
                    "category": products.get(prod_id, {}).get("category", "unknown"),
                    "price": products.get(prod_id, {}).get("price", "Unknown Price"),
                    "stockQuantity": products.get(prod_id, {}).get("stockQuantity", 0),
                    "image": products.get(prod_id, {}).get("image", "No image available")
                }
                for prod_id in sorted_product_ids
            ]
        except Exception as e:
            logger.error(f"Error sorting products: {str(e)}")
            logger.error(traceback.format_exc())
            
            # APPROACH 3: Fallback to all products unsorted
            sorted_products = [
                {
                    "_id": prod_id,
                    "name": products.get(prod_id, {}).get("name", "Unknown Product"),
                    "description": products.get(prod_id, {}).get("description", "No description available"),
                    "category": products.get(prod_id, {}).get("category", "unknown"),
                    "price": products.get(prod_id, {}).get("price", "Unknown Price"),
                    "stockQuantity": products.get(prod_id, {}).get("stockQuantity", 0),
                    "image": products.get(prod_id, {}).get("image", "No image available")
                }
                for prod_id in products.keys()
            ]
        
        logger.info(f"Returning {len(sorted_products)} products")
        return sorted_products
        
    except Exception as e:
        error_msg = f"Error generating recommendations: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg) 