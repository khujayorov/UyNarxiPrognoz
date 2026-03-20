from pathlib import Path

path = Path('train_model.py')
text = path.read_text(encoding='utf-8')
needle = '            st.write(f"Bashorat qilingan qiymat: {prediction[0]:.2f}")\r\n'
if needle not in text:
    raise SystemExit('Needle not found, cannot patch')

replacement = needle + "\r\n" + "            # Natijani jadvalga chiqarish va yuklab olish\r\n" + \
"            result_df = pd.DataFrame({\r\n" + \
"                **{col: [val] for col, val in zip(df.columns[:-1], user_inputs)},\r\n" + \
"                df.columns[-1]: [prediction[0]],\r\n" + \
"            })\r\n" + \
"            st.write(\"Bashorat natijasi jadvali:\")\r\n" + \
"            st.dataframe(result_df)\r\n" + \
"\r\n" + \
"            csv_bytes = result_df.to_csv(index=False).encode('utf-8')\r\n" + \
"            st.download_button(\r\n" + \
"                label=\"Bashorat natijasini CSVga yuklab olish\",\r\n" + \
"                data=csv_bytes,\r\n" + \
"                file_name=\"bashorat_natijasi.csv\",\r\n" + \
"                mime=\"text/csv\",\r\n" + \
"            )\r\n" + \
"\r\n" + \
"            st.info(\"Agar ilova telefonda ochilgan bo'lsa, bu faylni kompyuterga yuborib ochishingiz mumkin.\")\r\n"

new_text = text.replace(needle, replacement)
path.write_text(new_text, encoding='utf-8')
print('Patch applied')
