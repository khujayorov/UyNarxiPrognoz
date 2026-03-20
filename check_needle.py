from pathlib import Path
path = Path('train_model.py')
text = path.read_text('utf-8')
needle = '            st.write(f"Bashorat qilingan qiymat: {prediction[0]:.2f}")\r\n        except Exception as e:'
print('found', needle in text)
print('index', text.find(needle))
