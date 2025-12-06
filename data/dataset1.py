import pandas as pd
import re

extracted_df = pd.read_csv('data/extracted_text.csv')

columns = ['No', 'Nama Menu', 'Kategori', 'Kalori (kcal)', 'Sumber Karbohidrat', 'Bahan Utama / Pendamping', 'Deskripsi Singkat']
formatted_data = []

# Regex patterns for strict matching (start of word)
# \b matches word boundary.
food_patterns = {
    'Ayam': r'\bayam',
    'Sapi': r'\bsapi',
    'Ikan': r'\bikan|\bdori|\btuna',
    'Beef': r'\bbeef',
    'Chicken': r'\bchicken',
    'Rice': r'\brice|\bnasi',
    'Salad': r'\bsalad'
}

carb_patterns = [r'\bnasi', r'\brice', r'\bkentang', r'\bpotato', r'\bubi', r'\bpasta']

for index, row in extracted_df.iterrows():
    text = str(row['extracted_text'])
    lines = text.split('\n')
    
    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        if len(line_clean) < 4: continue
        
        # Check for bad words (spam filters)
        if any(x in line_lower for x in ['review', 'testi', 'diskon', 'promo', 'pendidikan', 'lowongan', 'hiring', 'jalan', 'jl.', 'no.', 'telp']):
            continue
            
        found_category = None
        
        # Check Categories
        if re.search(food_patterns['Ayam'], line_lower) or re.search(food_patterns['Chicken'], line_lower):
            found_category = 'Ayam'
        elif re.search(food_patterns['Sapi'], line_lower) or re.search(food_patterns['Beef'], line_lower):
            found_category = 'Sapi'
        elif re.search(food_patterns['Ikan'], line_lower): # Dori included here
            found_category = 'Ikan'
        elif re.search(food_patterns['Rice'], line_lower): # Rice dishes
            found_category = 'Lainnya'
            
        if found_category:
            # Extract Carbs
            found_carbs = []
            for p in carb_patterns:
                if re.search(p, line_lower):
                    # Map pattern back to readable word? Just use a default list for now
                    if 'nasi' in p or 'rice' in p: found_carbs.append('Nasi')
                    if 'kentang' in p or 'potato' in p: found_carbs.append('Kentang')
            
            carbs_str = ', '.join(set(found_carbs)) if found_carbs else "Nasi (Asumsi)"
            
            # Extract Calories
            cal_match = re.search(r'(\d+)\s*(kcal|kal)', line_lower)
            calories = int(cal_match.group(1)) if cal_match else 0
            
            formatted_data.append({
                'Nama Menu': line_clean,
                'Kategori': found_category,
                'Kalori (kcal)': calories,
                'Sumber Karbohidrat': carbs_str,
                'Bahan Utama / Pendamping': 'Sayur / Lauk Tambahan',
                'Deskripsi Singkat': f"Menu {line_clean} dari ekstrak Instagram."
            })

df_new = pd.DataFrame(formatted_data)
if not df_new.empty:
    df_new.drop_duplicates(subset=['Nama Menu'], inplace=True)
    df_new.insert(0, 'No', range(1, len(df_new) + 1))

df_new.to_csv('dataset_hasil_scan_instagram.csv', index=False)