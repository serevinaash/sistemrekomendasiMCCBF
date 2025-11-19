import pandas as pd
from io import StringIO

data = """user_id,kalori_target,kategori_lauk,sumber_karbo,deskripsi_preferensi,relevant_menus
1,400,Ayam,"nasi merah,jagung",tanpa santan rendah lemak panggang,"Chicken Teriyaki Grill,Chicken Blackpepper,Baked Chicken Honey,Chicken Herbs Grill & Egg Tortila,Chicken BBQ Grill,Chicken Grill Sambal Matah,Chicken Curry,Ayam Saus Mentega,Ayam Maranggi"
2,395,Ikan,"kentang,nasi merah",steamed tanpa santan saus mentega,"Dori Saus Mentega,Gulai Ikan,Dori Sambal Matah,Dori Sambal Ijo,Ikan Woku"
3,400,Sapi,"kentang,nasi merah",pedas gurih tumis cabe garam,"Sapi Cabe Garam,Rawon Surabaya,Bistik Sapi,Cah Sapi Buncis,Beef Bulgogi"
4,390,Ayam,"jagung,ubi",crispy renyah salad segar,"Bola Ayam Crispy,Chicken Patty Crispy,Chicken Ball Crispy,Chicken Crispy Saus Barbeque,Sempol Ayam Grill"
5,395,Ikan,"jagung,nasi putih",sambal pedas tumis sayur,"Dori Sambal Ijo,Dori Katsu Balado,Ikan Cabe Ijo,Dori Sambal Matah,Dori Cabe Garam,Dori Cabai Garam,Fish Katsu Saus Manis Pedas"
6,400,Ayam,"kentang,nasi merah",kuah soto tanpa santan bening,"Soto Ayam Spesial,Soto Ayam Kuning,Opor Ayam (Tanpa Santan),Chicken Tomyum,Soto Betawi"
7,395,Sapi,"nasi merah,kentang",saus teriyaki bulgogi korea,"Beef Yakiniku,Beef Bulgogi,Beef & Tofu Teriyaki,Beef Patty BBQ,Bola Daging Teriyaki"
8,400,Ayam,"jagung,nasi merah",bumbu serundeng tumis sayur tempe,"Ayam Serundeng,Ayam Serundeng & Sambal Korek,Ayam Maranggi,Ayam Balado"
9,390,Ikan,"kentang,ubi",saus telur asin gurih,"Dori Salted Egg,Chicken Salted Egg,Dori Saus Mentega"
10,395,Ayam,"nasi merah,kentang",sambal matah segar panggang,"Chicken Grill Sambal Matah,Dori Sambal Matah,Ayam Bakar Padang,Chicken BBQ Grill"
11,400,Sapi,"kentang,nasi merah",kuah rempah rawon soto,"Rawon Surabaya,Soto Bandung,Soto Betawi,Soto Daging Spesial"
12,395,Ayam,"jagung,nasi putih",tumis buncis ayam cincang,"Cah Buncis Ayam,Cah Ayam Buncis,Cah Ayam Jamur,Cah Brokoli Ayam,Tumis Ayam Paprika"
13,400,Ikan,"kentang,nasi merah",katsu renyah saus asam manis,"Dori Katsu Saus Asam Manis,Dori Katsu,Fish Katsu Saus Manis Pedas,Dori Chips Saus Asam Manis,Dori Katsu Balado,Dori Oat Saus Asam Manis"
14,390,Ayam,"ubi,kentang",bakar bumbu padang tanpa santan,"Ayam Bakar Padang,Ayam Bakar,Ayam Bakar (Bacem),Chicken BBQ Grill,Baked Chicken Honey"
15,395,Sapi,"jagung,nasi merah",tumis buncis lada hitam,"Buncis Sapi Lada Hitam,Cah Buncis Sapi Cincang,Brokoli Sapi Lada Hitam,Cah Sapi Buncis,Bola Daging Saus Lada Hitam"
16,400,Ayam,"nasi merah,kentang",bumbu woku pedas kemangi,"Chicken Woku,Chicken Ball Woku,Ikan Woku,Ayam Cabe Ijo"
17,395,Ikan,"jagung,nasi putih",stik ikan saus korea wijen,"Fish Stick Saus Korea,Fish Stick Saus Wijen,Dori Stick,Dori Oat Saus Wijen"
18,400,Ayam,"kentang,nasi merah",panggang madu saus honey,"Chicken Honey,Baked Chicken Honey,Egg Tortila & Baked Chicken Honey,Ayam Bakar (Bacem)"
19,390,Ayam,"jagung,nasi merah",rolade kukus tumis sayur,"Chicken Rolade,Rolade Ayam,Sempol Ayam Grill,Ayam Bakar,Cah Buncis Ayam"
20,395,Sapi,"kentang,nasi merah",bola daging saus teriyaki,"Bola Daging Teriyaki,Bola Daging Saus Lada Hitam,Bistik Bola Sapi,Beef Yakiniku,Beef & Tofu Teriyaki"
"""

df = pd.read_csv(StringIO(data))
output_path = "ground_truth_v3.csv"
df.to_csv(output_path, index=False)

print("File saved:", output_path)
