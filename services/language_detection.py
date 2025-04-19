def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Supports: English, French, Arabic, Tunisian Arabic (Tounsi), Algerian Arabic (Jazairi)
    
    Args:
        text: The input text to analyze
    
    Returns:
        String identifier for the detected language ("english", "french", "arabic", "tounsi", or "jazairi")
    """
    import re
    
    # Check for empty text
    if not text or text.strip() == "":
        return "english"  # Default to English for empty text
    
    text_lower = text.lower().strip()
    total_length = len(text_lower)
    
    # Define patterns for different scripts
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    french_pattern = re.compile(r'[éèêëàâäôöùûüÿçœæ]')
    
    # Count occurrences of different scripts
    arabic_chars = len(re.findall(arabic_pattern, text))
    french_special_chars = len(re.findall(french_pattern, text))
    
    # Check for Arabizi numerals (common in Tounsi/Jazairi)
    arabizi_numerals = len([c for c in text_lower if c in ['3', '7', '9', '5', '2']])
    arabizi_ratio = arabizi_numerals / total_length if total_length > 0 else 0
    
    # Expanded Tounsi keywords based on dataset
    tounsi_keywords = [
        # Common Tunisian markers
        'fi', 'elli', 'tawa', 'chkoun', 'bech', 'mch', 'enti', 'inti', 'famma', 'barcha',
        'yezzi', 'chneya', 'lahné', 'tounes', 'mta3i', 'chnowa',
        
        # Agricultural terms
        'war9a', 'safra', 'nabta', 'siyam', 'rosh', 'n7i', 't9ta3', 'nadhf', 
        'bidhour', 'anzimet', 'ri', 'zira3a', 'salem', 'nheb', 'na3ref',
        
        # Olive-specific terms
        'zitoun', 'zitouna', '3ando', 'barsha', 'mardh', 'amrath',
        'chajra', 'ghars', 'ta9lim', 'trab', 'jdher', 'sa9', 'thmar',
        
        # Disease and treatment terms
        '3afej', 'maridh', 'dwa', 'dawé', '7achrat', 'mbidet', 'virus',
        'cycloconium', 'vertichillium', 'nwar', '9ar7a', 'ta3fen',
        
        # Common verbs and actions
        'a7reth', 'nzar3ou', 'na79ou', 'nfas5ou', 'n9al3ou', 'nrosh',
        'lazem', 'mta3', 'mte3', 'el', 'fil', 'wel', 'wil', 'bel', 'bil',
        
        # Weather and conditions
        'chems', 'mtar', 'rih', 'bard', 's5ana', 'rtoba', 'jfef',
        'chtè', 'sayf', '5rif', 'rbi3', 'ardh', 'trab', 'ma',
        
        # Time expressions
        'lyoum', 'ghodwa', 'ba3d', '9bal', 'wa9t', 'mawsem', 'sné',
        
        # Question words
        '3lech', 'kifech', 'winou', 'wa9teh', 'chbih', 'chnowa',
        
        # Common adjectives
        'behia', 'mriguel', '5ayeb', 'mli7', 'sghar', 'kbar',
        '5dher', 'yebis', 'jdid', '9dim', 'sa7i7', 'maridh'
    ]
    
    jazairi_keywords = [
        'kayen', 'wech', 'kach', 'makan', 'dzair', '3lach', 'kifach', 'chouf', 'ntaya', 'hna',
        'hadak', 'sahbi'
    ]
    
    # French keywords and phrases
    french_keywords = [
        'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cette', 'ces',
        'est', 'sont', 'être', 'avoir', 'faire', 'quelle', 'quel', 'quels', 'quelles',
        'et', 'ou', 'mais', 'donc', 'car', 'parce', 'pour', 'dans', 'sur', 'sous'
    ]
    
    french_phrases = [
        'qu est[- ]ce que',
        'est[- ]ce que',
        'il y a',
        'je suis',
        'c est',
        'je voudrais',
        'pouvez[- ]vous',
        'comment faire'
    ]
    
    # Count dialect keywords
    tounsi_count = sum(1 for word in text_lower.split() if word in tounsi_keywords)
    jazairi_count = sum(1 for word in text_lower.split() if word in jazairi_keywords)
    
    # Count French words and phrases
    french_word_count = sum(1 for word in text_lower.split() if word in french_keywords)
    french_phrase_count = sum(1 for pattern in french_phrases if re.search(pattern, text_lower))
    
    # Detect Arabizi-based dialects first (prioritize this check)
    if arabizi_ratio > 0.05 or tounsi_count > 0 or jazairi_count > 0:
        if tounsi_count > jazairi_count or any(word in text_lower for word in ['tounes', 'mta3i', 'chnowa', 'chnouma']):
            return "tounsi"
        elif jazairi_count > tounsi_count:
            return "jazairi"
        else:
            return "tounsi"  # Default to Tounsi for ambiguous Arabizi
    
    # Detect Standard Arabic (significant Arabic script)
    arabic_ratio = arabic_chars / total_length if total_length > 0 else 0
    if arabic_ratio > 0.15:
        return "arabic"
    
    # Detect French (significant French indicators)
    french_ratio = french_special_chars / total_length if total_length > 0 else 0
    if french_ratio > 0.05 or french_word_count >= 2 or french_phrase_count > 0:
        return "french"
    
    # Default to English for pure Latin script without strong indicators
    return "english"