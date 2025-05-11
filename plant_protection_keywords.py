"""
Plant protection keywords for various crops, diseases, and treatments.
This module provides keywords that help identify relevant agricultural products for crop protection.
"""

plant_protection_keywords = {
    # Crops
    "crops": {
        "tomato": ["tomato", "tomatoes", "solanum lycopersicum"],
        "potato": ["potato", "potatoes", "solanum tuberosum"],
        "wheat": ["wheat", "triticum"],
        "rice": ["rice", "oryza sativa"],
        "corn": ["corn", "maize", "zea mays"],
        "cotton": ["cotton", "gossypium"],
        "carrot": ["carrot", "daucus carota"],
        "olive": ["olive", "olea europaea", "زيتون", "zitouna"],
        "fig": ["fig", "ficus carica", "تين", "karmous", "karmousa"],
    },
    
    # Common diseases
    "diseases": {
        "blight": ["early blight", "late blight", "fire blight", "bacterial blight"],
        "mildew": ["downy mildew", "powdery mildew", "mildiou"],
        "rust": ["rust", "leaf rust", "stem rust"],
        "rot": ["root rot", "fruit rot", "stem rot", "ta3fen"],
        "wilt": ["verticillium wilt", "fusarium wilt", "bacterial wilt", "dboul"],
        "leaf_spot": ["leaf spot", "cercospora", "septoria", "alternaria"],
        "mosaic": ["mosaic virus", "tobacco mosaic", "cucumber mosaic"],
        "olive_eye_peacock": ["peacock eye", "cycloconium", "3ayn el tawous", "عين الطاووس"],
    },
    
    # Common treatments and products
    "treatments": {
        "fungicides": ["fungicide", "anti-fungal", "mancozeb", "chlorothalonil", "copper"],
        "insecticides": ["insecticide", "pest control", "imidacloprid", "spinosad"],
        "herbicides": ["herbicide", "weed control", "glyphosate", "2,4-D"],
        "fertilizers": ["fertilizer", "NPK", "nitrogen", "phosphorus", "potassium"],
        "organic": ["organic", "biological", "natural", "bio"],
        "chemical": ["chemical", "synthetic", "systemic"],
    }
} 